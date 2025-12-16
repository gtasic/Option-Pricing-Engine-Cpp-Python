import pandas as pd 
import numpy as np 
import supabase 
import os 
import sys 
from datetime import datetime, date, timezone
import backtest
import xgboost as xgb
from scipy.optimize import minimize
from dotenv import load_dotenv
sys.path.append("/workspaces/finance-/build")
from copy import deepcopy
import sabr 
import calibration as cal

load_dotenv()
supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
client_supabase = supabase.create_client(supabase_url, supabase_key)

asof = datetime.now(timezone.utc).date() 


class ComplexPortfolio:
    def __init__(self, cash_position,nav, quantity_assets, options_list, options_available, total_delta, total_gamma, total_vega, total_theta, total_rho, total_sigma):
        self.options_list = options_list  # Liste des options ouvertes dans le portefeuille
        self.options_available = options_available  # Options disponibles pour arbitrage
        self.nav = float(nav)
        self.cash_position = cash_position
        self.spot_price = None
        self.nav_yesterday = float(nav)
        self.transaction_costs = 0.001 # Frais de transaction fixes par opération
        self.quantity_assets = float(quantity_assets)
        self.total_delta =total_delta
        self.total_gamma = total_gamma
        self.total_vega = total_vega
        self.total_theta = total_theta
        self.total_rho = total_rho
        self.total_sigma = total_sigma
        self.daily_pnl = 0.0
     


    def df_for_ML(self,client_supabase,df, asof) : 
        S0 = df.iloc[-1]["close"]
        options = client_supabase.table("vol_surfaces").select("*").eq("asof",asof).execute()
        options = options.data
        df_final = pd.DataFrame(options)

        def prepare_ml_features(df_day, heston_params, sabr_params_by_maturity, spot_price):
            df = df_day.copy()
            
            df['moneyness'] = df['strike'] / spot_price
            df['log_moneyness'] = np.log(df['moneyness'])
            
            df['H_kappa'] = heston_params['kappa']
            df['H_theta'] = heston_params['theta']
            df['H_volvol'] = heston_params['sigma_v']
            df['H_rho'] = heston_params['rho']
            
            df['H_diff_term'] = df['iv'] - np.sqrt(df['H_theta'])


            cols = ['S_alpha', 'S_rho', 'S_nu', 'S_theoretical_iv', 'SABR_Edge']
            for c in cols:
                df[c] = np.nan

        
            for idx, row in df.iterrows():
                T = row['tenor']
                K = row['strike']
                
                available_maturities = list(sabr_params_by_maturity.keys())
                if not available_maturities:
                    continue 
                if T in sabr_params_by_maturity:
                    params = sabr_params_by_maturity[T]
                df.at[idx, 'S_alpha'] = params['alpha']
                df.at[idx, 'S_rho'] = params['rho']
                df.at[idx, 'S_nu'] = params['nu']
                try:
                    theo_iv = sabr.sabr_vol(K, spot_price, T, params['alpha'], 0.5, params['rho'], params['nu'])
                    df.at[idx, 'S_theoretical_iv'] = theo_iv
                except:
                    df.at[idx, 'S_theoretical_iv'] = row['iv'] 
            df['SABR_Edge'] = df['iv'] - df['S_theoretical_iv']
            
            return df
        
        processed_days = [] 
        unique_dates = df_final['asof'].unique()
        print(f"Démarrage de la construction du Dataset sur {len(unique_dates)} jours...")

        for date in unique_dates:
            df_day = df_final[df_final['asof'] == date]
            if df_day.empty: continue
                
         
            df_day["strike"] = df_day["moneyness"]*S0
            
            try:
                h_params = cal.calibrate_heston_to_surface(df_day, S0=S0, r=0.04)
                print("les paramètres heston pour le", date, "sont :", h_params)
            except Exception as e:
                print(f"⚠️ Fail Heston {date}: {e}")
                h_params = {'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.7}

            day_maturities = df_day['tenor'].unique()
            print(f"Les maturités disponibles pour le {date} sont : ", day_maturities)
            sabr_dict_day = {} 
            
            for T in day_maturities:
                df_slice = df_day[df_day['tenor'] == T]

                print(f"Calibrating SABR for maturity {T:.2f} on {date} with {len(df_slice)} options")
                if len(df_slice) < 4:
                    continue
                print(df_slice.columns)
                try:
                    
                    alpha, beta, rho, nu, err = sabr.calibrate_sabr_robust(
                        df_slice['strike'].values,
                        df_slice['iv'].values,
                        S0=S0,
                        T=T,
                        r=0.04,
                        q=0.0,
                        beta=0.5
                    )
                    
                    sabr_dict_day[T] = {
                        'alpha': alpha,
                        'beta' : 0.5,
                        'rho': rho,
                        'nu': nu,
                        'error': err
                    }
                except Exception as e :
                    print("Nous n'avons pas assez d'informations pour calibrer SABR pour la maturité", T, "le", date, "car :", e)
                    continue

            if sabr_dict_day: 
                print("On a calibré SABR pour le", date, "sur", len(sabr_dict_day), "maturités.")
                df_day_enriched = prepare_ml_features(df_day, h_params, sabr_dict_day, S0)
                print(df_day_enriched[['asof', 'iv', 'H_kappa', 'S_rho', 'SABR_Edge']].head())
                processed_days.append(df_day_enriched)

        if processed_days:
            df_train_ready = pd.concat(processed_days, ignore_index=True)
            print("✅ Dataset construit avec succès !")
            print(df_train_ready[['asof', 'iv', 'H_kappa', 'S_rho', 'SABR_Edge']].head())
        else:
            print("❌ Aucun jour n'a pu être traité.")
        df_train_ready = df_train_ready.dropna()
        df_train_ready.to_csv("df_for_ml.csv", index=False)
        df_train_ready["log_moneyness"] = np.log(df_train_ready["moneyness"])

        df["log_returns"] = np.log(df['close'] / df['close'].shift(1))
        df["hist_vol_10d"] = df["log_returns"].rolling(window=5).std() * np.sqrt(252)
        df_train_ready["hist_vol_10d"] = df["hist_vol_10d"].iloc[-1]

        return df_train_ready




    def update_options_open(self, client_supabase) : 
        
        if self.options_list.empty: 
            print("Aucune option n'a été ouverte dans le portefeuille.")

        reponse2 = client_supabase.table("simulation_params").select("*").eq("asof",datetime.now(timezone.utc).date().isoformat()).execute()
        df_params = pd.DataFrame(reponse2.data)

        if df_params.empty : 
            print("⚠️ Pas de données de marché (simulation_params) pour aujourd'hui.")
            return

        df_params[['gamma','vega','theta','rho','BS_price','MC_price','CRR_price']] = np.nan
        df_params = backtest.final_pd(df_params)
        df_params["prix"] =  (df_params["bid"] + df_params["ask"])/2

        for index, row in self.options_list.iterrows():
            df_option = df_params[df_params["contract_symbol"] == row["contract_symbol"]]
            if not df_option.empty:
                option_data = df_option.iloc[0]
                client_supabase.table("complex_portfolio_options").update({
                    "sigma": option_data["sigma"],
                    "delta": option_data["delta"],
                    "gamma": option_data["gamma"],
                    "vega": option_data["vega"],
                    "theta": option_data["theta"],
                    "rho": option_data["rho"],
                    "prix": option_data["prix"]
                }).eq("contract_symbol", row["contract_symbol"]).execute()

                self.options_list.at[index, "sigma"] = option_data["sigma"]
                self.options_list.at[index, "delta"] = option_data["delta"]
                self.options_list.at[index, "gamma"] = option_data["gamma"]
                self.options_list.at[index, "vega"] = option_data["vega"]
                self.options_list.at[index, "rho"] = option_data["rho"]
                self.options_list.at[index, "theta"] = option_data["theta"]
                self.options_list.at[index, "prix"] = option_data["prix"]
                
            else:                   
                print(f"L'option avec le symbole {row['contract_symbol']} n'a pas été trouvée dans les paramètres de simulation.")
          

    def close_position(self,supabase_client,S0):

        if self.options_list.empty :
            return
        

        asof_str = datetime.now(timezone.utc).date().isoformat()
        rows_to_drop = []
        response = supabase_client.table("portfolio_options").select("*").eq("statut", "open").execute()
        open_positions = response.data

        for index, row in self.options_list.iterrows():
            expiry_date = date.fromisoformat(row["expiry"])
            if expiry_date <= datetime.now(timezone.utc).date() :
                strike = float(row['strike'])
                quantity = float(row['quantity'])
                is_call = 'C' in row['contract_symbol']
                
                payoff_unit = max(0.0, S0-strike) if is_call else max(0.0, strike-S0)
                cash = payoff_unit*quantity

                cost = abs(cash) * self.transaction_costs
                net_cash = cash - cost

                self.cash_position +=net_cash  

                supabase_client.table("complex_portfolio_options").update({
                    "date_fermeture": asof_str,
                    "prix_fermeture": float(payoff_unit),
                    "statut": "closed"
                }).eq("contract_symbol", row["contract_symbol"]).execute() # On ferme la position en actualisant la valeur de statut à "closed" et la valeur de prix_fermeture au Payoff
                
                print(f"Fermeture Payoff de {row['contract_symbol']}: {payoff_unit:.2f} EUR/unité. Cash net: {net_cash:.2f}")
                rows_to_drop.append(index)
        if rows_to_drop:
            self.options_list = self.options_list.drop(rows_to_drop)

    def greeks_calcul(self, supabase_client):
        reponse = supabase_client.table("complex_portfolio_options").select("*").eq("statut","open").execute()
        reponse = reponse.data
        df_options = pd.DataFrame(reponse)
        if df_options.empty: 
            print("Aucune option n'a été ouverte dans le portefeuille.")
            return
        self.total_delta = (df_options["delta"] * df_options["quantity"]).sum()
        self.total_gamma = (df_options["gamma"] * df_options["quantity"]).sum()
        self.total_vega = (df_options["vega"] * df_options["quantity"]).sum()
        self.total_theta = (df_options["theta"] * df_options["quantity"]).sum()
        self.total_rho = (df_options["rho"] * df_options["quantity"]).sum()
        self.total_sigma = (df_options["sigma"] * df_options["quantity"]).sum()

    
    def delta_hedging(self, supabase_client, S0) : 

        target_asset_qty = -self.total_delta
        diff_quantity = target_asset_qty - self.quantity_assets

        if(abs(diff_quantity)>0.5):
            cost = diff_quantity*S0
            frais = abs(cost)*self.transaction_costs

            self.cash_position -= (cost+frais)
            self.quantity_assets+=diff_quantity

            self.total_delta += diff_quantity

            print(f"⚖️ Delta Hedging: Ajustement Spot de {diff_quantity:.2f} unités. Coût: {cost:.2f}")
        else:
            print("👌 Delta Hedging: Pas d'ajustement nécessaire.")

         

    def dataframe_for_ML(self, supabase_client): 
        reponse = backtest.final
        self.options_available = reponse
        return self.options_available
    

    def mispriced_options_with_ml(self,supabase_client, df):
        features = ["tenor","moneyness","iv",
            "hist_vol_10d","log_moneyness",
            "H_kappa","H_volvol","H_diff_term",
            "S_rho","S_nu","SABR_Edge"]
        model = xgb.XGBRegressor()
        model.load_model("volatility_model_v1.json") 
        y_pred = model.predict(df[features])
        df["predicted_iv"] = y_pred
        df["iv_diff"] = df["iv"] - df["predicted_iv"]
        options_available_sorted = df.sort_values(by="iv_diff", ascending=False)
        
        if df[df["iv_diff"] > 0].empty: 
            print("Aucune possibilité d'aribitrage détectée.")
        
        else :
            print("Options mispricées détectées pour arbitrage.")
        df.to_csv("final_pred.csv")
        return df

    def mispriced_options_without_ML(self,supabase_client,df):
        if self.options_list is not None and not getattr(self.options_list, 'empty', True):
            df = df[~df["contract_symbol"].isin(self.options_list["contract_symbol"]) ]
        options_sorted = df.sort_values("SABR_Edge", ascending=False).copy()
        return options_sorted  #En l'absence d'une quantité suffisante de données, on se concentre sur la différence de SABR 
        
    
    def new_options(self,options_sorted, client_supabase, asof,S0) : 
        df_existing_options = self.options_list
        options_sorted = options_sorted.copy()
        options_sorted["S0"] = S0
        options_sorted[['gamma','vega','theta','rho','BS_price','MC_price','CRR_price']] = np.nan

        # ensure columns expected by backtest.final_pd are present
        if "tenor" in options_sorted.columns and "T" not in options_sorted.columns:
            options_sorted = options_sorted.rename(columns={"tenor": "T"})
        if "S0" not in options_sorted.columns:
            options_sorted["S0"] = S0
        if "sigma" not in options_sorted.columns and "iv" in options_sorted.columns:
            options_sorted["sigma"] = options_sorted["iv"]

        try:
            reponse = backtest.final_pd(options_sorted)
        except Exception as e:
            print("Error computing greeks/prices with backtest.final_pd:", e)
            reponse = options_sorted.copy()

        if df_existing_options is not None and not getattr(df_existing_options, 'empty', True): 
            new_options = reponse[~reponse["contract_symbol"].isin(df_existing_options["contract_symbol"])].copy()
        else:
            new_options = reponse.copy()

        print("options_sorted shape:", options_sorted.shape)
        print("reponse shape:", reponse.shape)
        print("new_options shape:", new_options.shape)
        print(new_options[new_options["SABR_Edge"]>0].head())

       
        for i in range(min(2, len(new_options))):
            try:
                option = new_options.iloc[i]
                client_supabase.table("complex_portfolio_options").insert({"contract_symbol": option["contract_symbol"],
                                                                                "date_ouverture": asof.isoformat(),
                                                                                "expiry": option["expiry"] ,
                                                                                "quantity": 10,
                                                                                "strike" : option["strike"],
                                                                                "t" : option.get("tenor", option.get("T")),
                                                                                "prix": option["BS_price"],
                                                                                "delta": option["delta"],
                                                                                "sigma":option["iv"],
                                                                                "gamma":option["gamma"],
                                                                                'vega':option["vega"],
                                                                                "theta":option["theta"],
                                                                                "rho":option["rho"],
                                                                                "SABR_Edge": option["SABR_Edge"],
                                                                                "vol_ouverture": option["iv"],
                                                                                "comments": f"ouverte le {asof} pour une quantité {option.get('quantity',10)} pour un prix de {(option.get('price') if 'price' in option else (option.get('ask',0)+option.get('bid',0))/2)} avec un objectif de diff de vol de {option.get('SABR_Edge',0)}" ,
                                                                                "statut" : "open"}).execute() 
                
                print(f"Option intégré au portefeuille {option['contract_symbol']}")
                self.cash_position -= option["prix"]*option["quantity"]

                new_row = option.to_dict()
                new_row["quantity"] = 10
                new_row["prix"] = option["BS_price"]

                self.options_list = pd.concat([self.options_list, pd.DataFrame([new_row])], ignore_index = True)

            except Exception as e : 
                print(f"pas asser d'options mispricés différentes pour en sélectionner {i}, ", e)
        
    

    def multi_greeks_hedging(self, supabase_client, target_gamma = (50,200)):
        
        current_delta = self.total_delta
        current_gamma = self.total_gamma    
        current_vega = self.total_vega
        target_delta = 0.0
        min_gamma, max_gamma = target_gamma
        target_vega = 0.0
    
        hedge_options = self.options_list.copy()

        if min_gamma <= current_gamma <= max_gamma:
            print("Le gamma actuel est déjà dans la plage cible, pas besoin de rééquilibrer.")
            return 
        
        target_gamma = (min_gamma + max_gamma) / 2
        gamma_needed = target_gamma - current_gamma

    
    
        def objective(quantities):
            cost = sum(abs(q) * opt.CRR_price * 0.001 for q, opt in zip(quantities, hedge_options.itertuples()))
            return cost
        
        def constraint_delta(quantities):
            hedge_delta = sum(q * opt.delta for q, opt in zip(quantities, hedge_options.itertuples()))
            return abs(current_delta + hedge_delta - target_delta)
        
        def constraint_gamma(quantities):
            hedge_gamma = sum(q * opt.gamma for q, opt in zip(quantities, hedge_options.itertuples()))
            return abs(current_gamma + hedge_gamma - target_gamma)
        
        def constraint_vega(quantities):
            hedge_vega = sum(q * opt.vega for q, opt in zip(quantities, hedge_options.itertuples()))
            return abs(current_vega + hedge_vega - target_vega)
        
        constraints = [
            {'type': 'eq', 'fun': constraint_delta},
            {'type': 'eq', 'fun': constraint_gamma},
            {'type': 'eq', 'fun': constraint_vega}
        ]
        
        x0 = [0.0] * len(hedge_options)
        result = minimize(objective, x0, constraints=constraints, method='SLSQP')
        
        if result.success:
            adjustements = np.round(result.x)
            for i,adj in enumerate(adjustements) : 
                if abs(adj)>0: 
                    row = self.options_list.iloc[i]
                    cost = adj*row[i]
                    self.cash_position -=cost


                    new_quantity = self.options_list.at[i,"quantity"] +adj
                    self.options_list.at[i,"quantity"]= new_quantity

                    client_supabase.table("complex_portfolio_options").update({
                        "quantity" : new_quantity
                    }).eq("contract_symbol", row["contract_symbol"]).execute()

                    print(f" Gamma Hedge: Ajustement {row['contract_symbol']} de {adj}")
            
            self.greeks_calcul(supabase_client)

        
        

    def rebalance_portfolio(self,supabase_client, hedge_positions,final_quantities):
        for contract_symbol, quantity in hedge_positions.items():
            prix_option_response = supabase_client.table("simulation_params").select("BS_price").eq("contract_symbol", contract_symbol).execute()
            if not prix_option_response.data:
                print(f"Erreur: Prix de l'option non trouvé pour {contract_symbol}.")
                continue
            option_price = prix_option_response.data[0]['CRR_price']
            transaction_costs = abs(quantity) * option_price * self.transaction_costs
            self.cash_position -= quantity * option_price + transaction_costs
        self.greeks_calcul(supabase_client)  # Met à jour les grecs après le rééquilibrage
    
    def to_complex_portfolio(self, supabase_client, S0, asof) : 

        current_nav = self.calculate_nav(S0)
        daily_pnl = current_nav - self.nav_yesterday
        supabase_client.table("daily_complex_portfolio_pnl").insert({
            "asof" : asof.isoformat(),
            "buy_price_asset" : S0,
            "quantity_asset" : self.quantity_assets,
            "nav" : current_nav,
            "daily_pnl" : daily_pnl,
            "total_gamma" : self.total_gamma,
            "total_delta" : self.total_delta,
            "total_vega" : self.total_vega,
            "total_sigma" : self.total_sigma,
            "total_rho" : self.total_rho ,
            "cash_position" : self.cash_position

        }).execute()

    def calculate_nav(self,S0): 
        valeur_asset = self.quantity_assets*S0
        valeur_options = (self.options_list["prix"] * self.options_list["quantity"]).sum() if not self.options_list.empty else 0
        self.nav = self.cash_position + valeur_asset + valeur_options
        return self.nav
        

    @classmethod
    def portfolio_builder(cls, supabase_client):
        try :
            reponse = supabase_client.table("complex_daily_portfolio_pnl").select("*").execute()
            reponse = reponse.data

            last_state = reponse[-1]

            options = supabase_client.table("complex_portfolio_options").select("*").eq("statut", "open").execute()
            df_options = pd.DataFrame(options.data)

            if not reponse : 
                print("Nouveau portefeuille, pas d'historique")
                return cls(10000, 10000,0,df_options,None,0,0,0,0,0,0)
            return cls(
                cash_position = last_state.get("cash_position",0),
                nav = last_state.get("nav",0),
                quantity_assets = last_state.get("quantity_assets",0),
                options_list = df_options,
                options_available = None,
                total_delta = last_state.get("total_delta",0),
                total_gamma = last_state.get("total_gamma",0),
                total_vega = last_state.get("total_vega",0),
                total_theta = last_state.get("total_theta",0),
                total_rho = last_state.get("total_rho",0),
                total_sigma = last_state.get("total_sigma",0),
                  )
        except Exception as e : 
            print("On n'a pas d'informations")
            return cls(10000,10000, 0,pd.DataFrame(), None,0,0,0,0,0,0)
    
   


def strategy(client_supabase) : 
    asof = datetime.now(timezone.utc).date() 
    print(f"Démarrage du script pour le jour {asof}")

    try : 
        reponse = client_supabase.table("prices").select("*").execute()
        if not reponse.data : 
            print("Erreur : aucun prix trouvé")
            return 
        df_prices = pd.DataFrame(reponse.data)
        S0 = df_prices["close"].iloc[-1]
        print(f"Prix récupére pour {S0:.2f}")
    except Exception as e : 
        print("erreur lors de la récupération du Spot", e)
        return
    
    print("\n--- 1. Chargement du Portefeuille ---")
    try : 
        portfolio = ComplexPortfolio.portfolio_builder(client_supabase)
        print("portefeuille chargé")
        print(f"Cash: {portfolio.cash_position:.2f} | Actifs: {portfolio.quantity_assets:.2f} | NAV Veille: {portfolio.nav_yesterday:.2f}")
    except Exception as e:
        print(f"Crash critique au chargement du portefeuille : {e}")
        return
    
    print("\n--- 2. Mise à jour des prix (Mark-to-Market) ---")
    portfolio.update_options_open(client_supabase)

    print("\n--- 3. Fermeture des positions expirées ---")
    portfolio.close_position(client_supabase, S0)

    print("\n--- 4. Recherche d'opportunités (ML & Arbitrage) ---")
    try:
        df_ml_ready = portfolio.df_for_ML(client_supabase, df_prices, asof)
        
        if not df_ml_ready.empty:
            candidates = portfolio.mispriced_options_without_ML(client_supabase, df_ml_ready)
          
            portfolio.new_options(candidates, client_supabase, asof, S0)
        else:
            print("Pas de données de surface de vol (vol_surfaces) pour aujourd'hui. Pas de nouveaux trades.")
    except Exception as e:
        print(f"Erreur non-bloquante dans le module ML/Trading : {e}")

    print("\n--- 5. Gestion des Risques (Hedging) ---")

    portfolio.greeks_calcul(client_supabase)
    print(f"Grecs actuels -> Delta: {portfolio.total_delta:.2f} | Gamma: {portfolio.total_gamma:.2f}")

    portfolio.delta_hedging(client_supabase, S0)
    portfolio.greeks_calcul(client_supabase)

    portfolio.multi_greeks_hedging(client_supabase, target_gamma=(50, 200))

    print("\n--- 6. Sauvegarde et Fin ---")
    portfolio.to_complex_portfolio(client_supabase, S0, asof)
    print("Run terminé avec succès.")

if __name__ == "__main__":
    strategy(client_supabase)

    
