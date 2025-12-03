import supabase
import pandas as pd
import numpy as np
from datetime import date, datetime
from zoneinfo import ZoneInfo
UTC = ZoneInfo("UTC")
import os
from numpy import random
import backtest
from dotenv import load_dotenv
import sys
sys.path.append("/workspaces/finance-/build")
import finance
from copy import deepcopy
from scipy.stats import skew, kurtosis
import calibration as cal
import sabr
from scipy.optimize import minimize


load_dotenv()
supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")


supabase_client = supabase.create_client(supabase_url, supabase_key)


class Portfolio:
    def __init__(self, cash_balance=100000.0, current_nav=10000.0, transaction_costs_rate=0.001, quantity_assets=0.0, delta_total=0.0, total_gamma=0.0, total_vega=0.0, total_theta=0.0, total_rho=0.0,total_sigma = 0.0 ):
        self.options = []           # Liste des options ouvertes dans le portefeuille
        self.cash_balance = cash_balance # Capital initial du portefeuille
        self.current_nav = current_nav  # Valeur actuelle du portefeuille
        self.transaction_costs_rate = 0.001 # 0.1% de frais de transaction
        self.quantity_assets = quantity_assets  # Quantité de l'actif sous-jacent détenue
        self.delta_total = delta_total    # Delta total du portefeuille
        self.total_gamma = total_gamma    # Gamma total du portefeuille
        self.total_vega = total_vega    
        self.total_theta = total_theta
        self.total_rho = total_rho
        self.total_sigma = total_sigma



    def form_option_df(self, df_final): #Avec ce programme on va regarder dans les 25 options quotidiennes celles qui ont un potentiel d'arbitrage selon le ML étant donné que le tenor doit être bas on va reformer un df avec toutes les options disopnibles
        processed_days = [] 
        unique_dates = df_final['asof'].unique()


        for date in unique_dates:
            df_day = df_final[df_final['asof'] == date]
            if df_day.empty: continue
        
            S0 = df_day['close'].iloc[0] 
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
                df_day_enriched = ml.prepare_ml_features(df_day, h_params, sabr_dict_day, S0)
                print(df_day_enriched[['asof', 'iv', 'H_kappa', 'S_rho', 'SABR_Edge']].head())
                processed_days.append(df_day_enriched)

        if processed_days:
            df_train_ready = pd.concat(processed_days, ignore_index=True)
            print("✅ Dataset construit avec succès !")
            print(df_train_ready[['asof', 'iv', 'H_kappa', 'S_rho', 'SABR_Edge']].head())
        else:
            print("❌ Aucun jour n'a pu être traité.")
        df_train_ready = df_train_ready.dropna()

        self.option = df_train_ready
        return df_train_ready #Avec ce df il ne reste plus qu'à faire le ML pour identifier les opportunités d'arbitrage
    


    
    def identify_mispriced_options(self, options_universe, spot, historical_vols): #Avec ce programme on va regarder dans les 25 options lesquelles ont un potentiel d'arbitrage selon le ML

        opportunities = []
        
        for option in options_universe:
            implied_vol = option['iv']
            
       
            features_list = ["tenor","moneyness","iv",
            "hist_vol_10d","log_moneyness",
            "H_kappa","H_volvol","H_diff_term",
            "S_rho","S_nu","SABR_Edge"]
            features = np.array([[option[feat] for feat in features_list]])

            forecasted_realized_vol = self.ml_model.predict(features)
            
            vol_edge = forecasted_realized_vol - implied_vol
            
            if vol_edge > 0.02:  
                opportunities.append({
                    'option': option,
                    'implied_vol': implied_vol,
                    'forecasted_vol': forecasted_realized_vol,
                    'edge': vol_edge,
                    'expected_pnl': option['vega'] * vol_edge  # Théorique
                })
        
        opportunities.sort(key=lambda x: x['expected_pnl'], reverse=True)

        if opportunities:
            self.opportunities = opportunities
            print("The possible trades are :" , opportunities)
        else : 
            "No aribtrage opportunities found today."

        return opportunities  #Ici on a les options qui peuvent être edger aujourd'hui selon le ML
    
    def trade_arbitrage(self, opportunity): #On va acheter les deux options d'arbitrage qui ont le meilleur potentiel d'arbitrage
  
        option = opportunity['option']

        self.portfolio.buy_option(option, quantity=10)
        entry_price = option['mid']
        entry_iv = option['implied_vol']
        delta = option['delta'] * 10
        hedge_quantity = -delta
        self.portfolio.buy_stock(hedge_quantity)
        
        self.portfolio.log_position({
            'entry_date': datetime.now(),
            'option_symbol': option['contract_symbol'],
            'entry_price': entry_price,
            'entry_iv': entry_iv,
            'forecasted_rv': opportunity['forecasted_vol'],
            'edge': opportunity['edge'],
            'delta': option['delta'],
            'gamma': option['gamma'],
            'theta': option['theta'],
            'vega': option['vega'],
            'status': 'open'
        })
        
        print(f"✅ Bought {option['contract_symbol']}: IV={entry_iv:.2%}, "
              f"Forecast RV={opportunity['forecasted_vol']:.2%}, Edge={opportunity['edge']:.2%}")
    
    def compute_hedge_positions(self, target_delta=0, target_gamma=0, target_vega=0): # On va regarder chaque jour le gamma des positions pour voir comment au mieux hedger le portefeuille
        
        
        current_delta = self.portfolio.delta_total
        current_gamma = self.portfolio.total_gamma
        current_vega = self.portfolio.total_vega
        
        hedge_opts = self.options[:]
        
        def objective(quantities):
            cost = sum(abs(q) * opt.CRR_price * 0.001 for q, opt in zip(quantities, hedge_opts.itertuples()))
            return cost
        
        def constraint_delta(quantities):
            hedge_delta = sum(q * opt.delta for q, opt in zip(quantities, hedge_opts.itertuples()))
            return abs(current_delta + hedge_delta - target_delta)
        
        def constraint_gamma(quantities):
            hedge_gamma = sum(q * opt.gamma for q, opt in zip(quantities, hedge_opts.itertuples()))
            return abs(current_gamma + hedge_gamma - target_gamma)
        
        def constraint_vega(quantities):
            hedge_vega = sum(q * opt.vega for q, opt in zip(quantities, hedge_opts.itertuples()))
            return abs(current_vega + hedge_vega - target_vega)
        
        constraints = [
            {'type': 'eq', 'fun': constraint_delta},
            {'type': 'eq', 'fun': constraint_gamma},
            {'type': 'eq', 'fun': constraint_vega}
        ]
        
        x0 = [0.0] * len(hedge_opts)
        result = minimize(objective, x0, constraints=constraints, method='SLSQP')
        
        if result.success:
            hedge_positions = {
                opt.contract_symbol: qty 
                for opt, qty in zip(hedge_opts.itertuples(), result.x) if abs(qty) > 0.01
            }
            final_quantities = round(result.x)
            return hedge_positions, final_quantities
        else:
            return self._delta_hedge_only(), None
    



    def close_position(self, supabase_client):
        asof_date = datetime.now(UTC).date()
        asof_str = asof_date.isoformat()
        response = supabase_client.table("portfolio_options").select("*").eq("status", "open").execute()
        open_positions = response.data

        positions_to_close = [
            pos for pos in open_positions 
            if date.fromisoformat(pos['expiry']) <= asof_date 
        ]
        
        prix_actif_response = supabase_client.table("prices").select("close").eq("asof", asof_str).execute()
        if not prix_actif_response.data:
            print(f"Erreur: Prix S0 non trouvé pour {asof_str} pour le calcul du Payoff.")
            return
        S0 = prix_actif_response.data[0]['close']

        for pos in positions_to_close:
            is_call = 'C' in pos['contract_symbol'] 
            strike = pos['strike']
            quantity = pos['quantity']

            if is_call:
                payoff = max(0.0, S0 - strike)
            else: # Put
                payoff = max(0.0, strike - S0)
                
            prix_fermeture_cash = payoff * quantity
            
            cash_recu = prix_fermeture_cash * (1 - self.transaction_costs_rate)
            self.cash_balance += cash_recu
    #        CURRENT_NAV -= prix_fermeture_cash  # On retire la valeur des options fermées de la NAV
            supabase_client.table("portfolio_options").update({
                "date_fermeture": asof_str,
                "prix_fermeture": float(payoff),
                "status": "closed"
            }).eq("contract_symbol", pos["contract_symbol"]).execute() # On ferme la position en actualisant la valeur de statut à "closed" et la valeur de prix_fermeture au Payoff
            
            print(f"Fermeture Payoff de {pos['contract_symbol']}: {payoff:.2f} EUR/unité. Cash net: {cash_recu:.2f}")

    def actualiser_delta(self, client_supabase): #chaque jour on met à jour le delta de chaque option "open" dans le portefeuille pour recalculer le delta total du portefeuille et faire le delta hedging
        response = client_supabase.table("complex_portfolio_options").select("*").eq("status", "open").execute()
        open_positions = response.data
        if not open_positions:
            print("Aucune position ouverte trouvée dans le portefeuille.")
            return

        response2 = client_supabase.table("simulation_params").select("*").execute()
        simu_params = response2.data
        df_simu = pd.DataFrame(simu_params)
        df_simu = df_simu[df_simu['asof'] == datetime.now(UTC).date().isoformat()]

        for position in open_positions:
            df_option = df_simu[(df_simu['contract_symbol'] == position['contract_symbol'])]
            if not df_option.empty:
                new_delta = df_option.iloc[0]['delta']
                client_supabase.table("portfolio_options").update({
                    "delta": float(new_delta)
                }).eq(["contract_symbol"], position["contract_symbol"]).execute()
                print(f"Delta mis à jour pour {position['contract_symbol']}: {new_delta}")
            else:
                print(f"Aucune donnée de delta trouvée pour {position['contract_symbol']} à la date {datetime.now(UTC).date().isoformat()}")
    """
        for position in open_positions:
            response = client_supabase.table("simulation_params").select("delta").eq(["contract_symbol","asof"], [position["contract_symbol"],datetime.now(UTC).date().isoformat()]).eq("strike", position['strike']).execute()
            if response.data:
                new_delta = response.data[0]['delta'] 
                client_supabase.table("portfolio_options").update({
                    "delta": float(new_delta)
                }).eq("contract_symbol", position["contract_symbol"]).execute()
                print(f"Delta mis à jour pour {position['contract_symbol']}: {new_delta}")
            else:
                print(f"Aucune donnée de delta trouvée pour {position['contract_symbol']} à la date {datetime.now(UTC).date().isoformat()}")
    """



    def actualiser_options_open(self, client_supabase): #mettre à jour les greeks et prix des options chaque jour pour avoir un delta hedging cohérent
        
        response = client_supabase.table("complex_portfolio_options").select("*").eq("status", "open").execute()
        open_positions = response.data
        if not open_positions:
            print("Aucune position ouverte trouvée dans le portefeuille.")
            return
        
        response2 = client_supabase.table("simulation_params").select("*").eq("asof", datetime.now(UTC).date().isoformat()).execute()
        simu_params = response2.data
        df_simu = pd.DataFrame(simu_params)
     #   df_simu = df_simu[df_simu['asof'] == datetime.now(UTC).date().isoformat()]
        df_simu[['gamma','vega','theta','rho','BS_price','MC_price','CRR_price']] = np.nan
        df_simu = backtest.final_pd(df_simu)
        df_simu["prix"] = (df_simu["bid"] + df_simu["ask"])/2



        for position in open_positions:
            df_option = df_simu[df_simu['contract_symbol'] == position['contract_symbol']]
            if not df_option.empty:
                option_data = df_option.iloc[0]
                client_supabase.table("portfolio_options").update({
              #      "asof": datetime.now(UTC).date().isoformat(),
                    "sigma" : float(option_data['sigma']),
                    "delta": float(option_data['delta']),
                    "gamma": float(option_data['gamma']),
                    "vega": float(option_data['vega']),
                    "theta": float(option_data['theta']),
                    "rho": float(option_data['rho']),
                    "prix": float(option_data['BS_price'])
                }).eq("contract_symbol", position["contract_symbol"]).execute()
                print(f"Options mises à jour pour {position['contract_symbol']} le {datetime.now(UTC).date().isoformat()}")
            else:
                print(f"Aucune donnée trouvée pour {position['contract_symbol']} à la date {datetime.now(UTC).date().isoformat()}")


    def calcul_delta_portefeuille(self, supabase_client):
        response = supabase_client.table("complex_portfolio_options").select("*").eq("status", "open").execute()
        open_positions = response.data

        delta_options = sum(pos['delta'] * pos['quantity'] for pos in open_positions)
        self.total_gamma = sum(pos['gamma'] * pos['quantity'] for pos in open_positions)
        self.total_vega = sum(pos['vega'] * pos['quantity'] for pos in open_positions)
        self.total_theta = sum(pos['theta'] * pos['quantity'] for pos in open_positions)
        self.total_rho = sum(pos['rho'] * pos['quantity'] for pos in open_positions)
        self.total_sigma = sum(pos['sigma'] * pos['quantity'] for pos in open_positions)


        delta_actif = self.quantity_assets
        total_delta = delta_options + delta_actif
        print(f"Delta options: {delta_options:.4f}, Delta actif: {delta_actif:.4f}, Total: {total_delta:.4f}")
        
        return total_delta


    def achat_actif(self, supabase_client): #total delta du portefeuille est obtenu avec la fonction calcul_delta_portefeuille
        total_delta = self.calcul_delta_portefeuille(supabase_client)
        if abs(total_delta) < 0.01:
            print("Le portefeuille est déjà delta-neutre.")
            return

        prix_actif = supabase_client.table("prices").select().execute()  #à cahnger pour être sur de prendre le prix du jour
        if not prix_actif.data:
            print(f"Aucun prix de l'actif sous-jacent trouvé pour la date {datetime.now(UTC).date().isoformat()}.")
            return

        prix_actif = prix_actif.data[-1]['close']  # Simuler une légère variation du prix en fonction de la quantité détenue


        quantite_actif_achat = -(total_delta )  # Acheter ou vendre l'actif pour neutraliser le delta
        self.quantity_assets += quantite_actif_achat

        cout_transaction = quantite_actif_achat * prix_actif * (1+self.transaction_costs_rate) if quantite_actif_achat >0 else quantite_actif_achat * prix_actif * (1 - self.transaction_costs_rate)
        cout_total = abs(quantite_actif_achat) * prix_actif * (1 + self.transaction_costs_rate)

    #    CURRENT_NAV += quantite_actif_achat * prix_actif - cout_transaction  # Met à jour la valeur actuelle du portefeuille après l'achat/vente et les frais de transaction
        self.cash_balance-= cout_transaction  # Met à jour le cash disponible après l'achat/vente et les frais de transaction
        
        print(f"{'Achat' if quantite_actif_achat > 0 else 'Vente'} de {abs(quantite_actif_achat):.2f} unités")
        print(f"Position totale en actif: {self.quantity_assets:.2f}")


 

    def to_supabase_portfolio(self, supabase_client):
        date_today = datetime.now(UTC).date().isoformat()
        buy_price_asset = supabase_client.table("prices").select().execute().data[-1]['close'] 
        supabase_client.table("daily_complex_portfolio_pnl").insert({
            "asof": date_today,
            "nav": self.current_nav,
            "cash_balance": self.cash_balance,
            "quantity_assets": self.quantity_assets,
            "buy_price_asset": buy_price_asset,
            "total_sigma" : self.total_sigma,
            "total_delta": self.delta_total,
            "total_gamma": self.total_gamma,   
            "total_vega": self.total_vega,
            "total_theta": self.total_theta,
            "total_rho": self.total_rho
        }).execute()
        
        return
    @classmethod
    def load_from_supabase(cls, supabase_client):
        """Charge l'état du portefeuille depuis Supabase"""
        response = supabase_client.table("daily_complex_portfolio_pnl").select("*").order(
            "asof", desc=True
        ).limit(1).execute()
        
        if response.data:
            state = response.data[0]
            return cls(
                cash_balance=state['cash_balance'],
                current_nav=state['nav'],
                quantity_assets=state['quantity_assets'],
                delta_total=state.get('total_delta', 0.0)
            )
        else:
            return cls()





def run_daily_strategy(supabase_client, df_daily_choice):
    # 1. Charger l'état du portefeuille
    portfolio = Portfolio.load_from_supabase(supabase_client)
    
    # 2. Fermer les positions expirées
    portfolio.close_position(supabase_client)
    
    # 3. Actualiser les options existantes
    portfolio.actualiser_options_open(supabase_client)
    
    # 4. Ouvrir nouvelles positions
    portfolio.trade_arbitrage(portfolio.identify_mispriced_options( portfolio.form_option_df(df_final)))
    
    # 5. Delta hedging
    portfolio.achat_actif(supabase_client)
    
    # 6. Calculer NAV et PnL
    nav = portfolio.calculer_NAV_journalier(supabase_client)
    
    # 7. Enregistrer
    delta_total = portfolio.calcul_delta_portefeuille(supabase_client)
    portfolio.to_supabase_portfolio(supabase_client)

    portfolio.daily_pnl(supabase_client)

    #8. On calcule les métrics
    metrics = portfolio.metrics_portfolio(supabase_client)
    portfolio.to_metrics_portfolio(supabase_client, metrics)
    return portfolio

if __name__ == "__main__":
    df_complex = pd.DataFrame(supabase_client.table("vol_surfaces").select("*").execute().data)
    df_complex = df_complex[df_complex["asof"]== datetime.now(UTC).date().isoformat()]
    portfolio = run_daily_strategy(supabase_client, df_complex)