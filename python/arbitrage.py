import supabase
import pandas as pd
import numpy as np
from datetime import date, datetime, timezone
import os
from dotenv import load_dotenv
import sys
import xgboost as xgb  # Nécessaire pour charger le modèle
from scipy.optimize import minimize
sys.path.append("/workspaces/finance-/build")
import finance 
import backtest
import calibration as cal
import sabr

load_dotenv()
UTC = timezone.utc

# Configuration Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)

class Portfolio:
    def __init__(self, ml_model_path="volatility_model_v1.json", cash_balance=100000.0, current_nav=100000.0, quantity_assets=0.0):
        # État du portefeuille
        self.cash_balance = float(cash_balance)
        self.current_nav = float(current_nav)
        self.quantity_assets = float(quantity_assets)
        self.transaction_costs_rate = 0.001
        
        # Grecs globaux
        self.delta_total = 0.0
        self.total_gamma = 0.0
        self.total_vega = 0.0
        self.total_theta = 0.0
        self.total_rho = 0.0
        self.total_sigma = 0.0
        
        # Liste locale des positions ouvertes (pour le hedging)
        self.open_positions = [] 
        self.opportunities = []

        # Chargement du modèle ML
        self.ml_model = xgb.XGBRegressor()
        try:
            # Assure-toi d'avoir sauvegardé ton modèle auparavant avec model.save_model("xgb_model.json")
            if os.path.exists(ml_model_path):
                self.ml_model.load_model(ml_model_path)
                print("✅ Modèle ML chargé.")
            else:
                print("⚠️ Attention: Fichier modèle introuvable. Les prédictions échoueront.")
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle ML: {e}")


    def buy_option(self, option_data, quantity):
        price = 3
        cost = price * quantity * (1 + self.transaction_costs_rate)
        
        self.cash_balance -= cost
        print(f"💰 Achat Option: {quantity} x {option_data['contract_symbol']} @ {price:.2f}. Cash restant: {self.cash_balance:.2f}")

    def buy_stock(self, quantity=10):
        """Achat/Vente du sous-jacent pour le Delta Hedging"""
        # Récupération propre du prix spot (Dernier prix dispo)
        try:
            price_resp = supabase_client.table("prices").select("close").order("asof", desc=True).limit(1).execute()
            spot_price = price_resp.data[0]['close']
        except:
            print("❌ Erreur: Impossible de récupérer le prix Spot pour le hedging.")
            return

        cost = abs(quantity) * spot_price * self.transaction_costs_rate
        trade_amount = quantity * spot_price
        
        total_cash_impact = trade_amount + cost if quantity > 0 else trade_amount - cost # Achat = sortie cash, Vente = entrée cash
        
        self.quantity_assets += quantity
        self.cash_balance -= total_cash_impact
        
        action = "Achat" if quantity > 0 else "Vente"
        print(f"📉 Hedge Spot ({action}): {abs(quantity):.4f} unités @ {spot_price:.2f}. Impact Cash: {total_cash_impact:.2f}")

    def log_position(self, trade_info):
 
        for k, v in trade_info.items():
            if isinstance(v, (np.generic, np.ndarray)):
                trade_info[k] = float(v)
            if isinstance(v, datetime):
                trade_info[k] = v.isoformat()
                
        supabase_client.table("complex_portfolio_options").insert(trade_info).execute()

    # --- MÉTHODES ANALYTIQUES (ML & ARBITRAGE) ---


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
                continue # Pas de calibration SABR réussie ce jour là
                
            #closest_T = min(available_maturities, key=lambda x: abs(x - T))
            #params = sabr_params_by_maturity[closest_T]
            if T in sabr_params_by_maturity:
                params = sabr_params_by_maturity[T]
            
            # Remplissage
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

    def form_option_df(self, df_final):
        processed_days = [] 
        unique_dates = df_final['asof'].unique()


        for date in unique_dates:
            df_day = df_final[df_final['asof'] == date]
            if df_day.empty: continue
        
            global spot_price 
            S0 = spot_price
            df_day["strike"] = df_day["moneyness"]*S0
                
            try:
                print("On calibre Heston pour le", date)
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
                df_day_enriched = Portfolio.prepare_ml_features(df_day, h_params, sabr_dict_day, S0)
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
    

    def identify_mispriced_options(self, options_df, spot=None, historical_vols=None):
        opportunities = []
        
        # Conversion DataFrame -> Liste de dicts pour itérer facilement
        options_universe = options_df.to_dict('records')

        for option in options_universe:
            # Vérifie que toutes les features nécessaires sont là
            try:
                features_list = ["tenor", "moneyness", "iv", "hist_vol_10d", "log_moneyness",
                                 "H_kappa", "H_volvol", "H_diff_term", "S_rho", "S_nu", "SABR_Edge"]
                
                input_data = []
                for f in features_list:
                    val = option.get(f, 0.0) # 0.0 par défaut si manquant (dangereux mais évite le crash)
                    input_data.append(val)
                
                features = np.array([input_data])
                
                forecasted_rv = self.ml_model.predict(features)[0] # [0] car predict renvoie un array
                implied_vol = option['iv']
                
                vol_edge = forecasted_rv - implied_vol
                
                if vol_edge > 0.02: # Edge min de 2%
                    opportunities.append({
                        'option': option,
                        'implied_vol': implied_vol,
                        'forecasted_vol': forecasted_rv,
                        'edge': vol_edge,
                        'expected_pnl': option.get('vega', 0.1) * vol_edge 
                    })
            except Exception as e:
                # print(f"Skip option {option.get('contract_symbol')}: {e}")
                continue
        
        opportunities.sort(key=lambda x: x['expected_pnl'], reverse=True)
        return opportunities

    def trade_arbitrage(self, opportunities):
        """Exécute les meilleurs trades trouvés"""
        if not opportunities:
            print("∅ Aucune opportunité d'arbitrage aujourd'hui.")
            return

        # On prend la meilleure opportunité (Top 1)
        best_opp = opportunities[0]
        option = best_opp['option']
        
        print(f"⭐ Opportunité trouvée sur {option['contract_symbol']} (Edge: {best_opp['edge']:.2%})")

        # 1. Achat de l'option
        qty_option = 10
        self.buy_option(option, qty_option)
        
        # 2. Delta Hedge immédiat (Hedging simple à l'entrée)
        delta_per_unit = option.get('delta', 0.5) 
        total_delta_trade = delta_per_unit * qty_option
        self.buy_stock(-total_delta_trade) # Vente du sous-jacent si delta positif

        # 3. Log
        self.log_position({
            'entry_date': datetime.now(UTC),
            'contract_symbol': option['contract_symbol'],
            'expiry': option.get('expiry', '2025-12-31'), # Fallback
            'strike': option['strike'],
            'quantity': qty_option,
            'entry_price': option.get('mid', 0),
            'entry_iv': option['iv'],
            'forecasted_rv': float(best_opp['forecasted_vol']),
            'edge': float(best_opp['edge']),
            'delta': float(delta_per_unit),
            'vega': float(option.get('vega', 0)),
            'gamma': float(option.get('gamma', 0)),
            'theta': float(option.get('theta', 0)),
            'status': 'open'
        })

    # --- MÉTHODES DE GESTION (HEDGING & PNL) ---

    def actualiser_options_open(self, client_supabase):
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
                client_supabase.table("complex_portfolio_options").update({
                    "asof": datetime.now(UTC).date().isoformat(),
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



    def calcul_delta_portefeuille(self, client_supabase):
        # Recharger les positions fraîches
        self.actualiser_options_open(client_supabase)
        
        delta_options = sum(pos['delta'] * pos['quantity'] for pos in self.open_positions)
        
        # Mise à jour des autres grecs globaux
        self.total_gamma = sum(pos['gamma'] * pos['quantity'] for pos in self.open_positions)
        self.total_vega = sum(pos['vega'] * pos['quantity'] for pos in self.open_positions)
        
        total_delta = delta_options + self.quantity_assets
        self.delta_total = total_delta
        
        print(f"📊 Grecs Portefeuille: Δ {total_delta:.2f} | Γ {self.total_gamma:.2f} | ν {self.total_vega:.2f}")
        return total_delta

    def achat_actif_delta_hedge(self, client_supabase):
        total_delta = self.calcul_delta_portefeuille(client_supabase)
        
        # Seuil de tolérance pour éviter de sur-trader
        if abs(total_delta) < 0.5: 
            print("✅ Portefeuille Delta-Neutre (dans la tolérance).")
            return

        # On hedge tout le delta excédentaire
        qty_to_hedge = -total_delta
        self.buy_stock(qty_to_hedge)



    def optimize_greeks_hedging(self, available_options_df, target_gamma=0, target_vega=0):
        """
        Cherche des options dans le marché pour neutraliser Gamma et Vega
        tout en minimisant les coûts de transaction.
        """
        print("\n--- 🛡️ DÉMARRAGE HEDGING GAMMA/VEGA ---")
        
        # 1. Calcul des Grecs actuels
        self.calcul_delta_portefeuille(None) # Met à jour self.total_gamma, etc.
        current_gamma = self.total_gamma
        current_vega = self.total_vega
        
        # Si on est déjà proches de 0, on ne fait rien (pour économiser les frais)
        if abs(current_gamma) < 10 and abs(current_vega) < 100:
            print("✅ Portefeuille déjà stable (Gamma/Vega faibles). Pas de hedge nécessaire.")
            return

        # 2. Sélection des instruments de couverture (Hedge Instruments)
        # On ne va pas utiliser 1000 options, on en prend 5-10 liquides (ATM et OTM proches)
        # On suppose que available_options_df contient les colonnes 'gamma', 'vega', 'mid', 'ask', 'bid'
        hedge_instruments = available_options_df.head(10).copy() # Simplification: on prend les 10 premières liquides
        
        n_instr = len(hedge_instruments)
        x0 = np.zeros(n_instr) # Quantité initiale (0)

        # 3. Définition du problème d'optimisation
        
        # Fonction de Coût : On veut minimiser le Spread payé (Coût de transaction)
        def cost_function(quantities):
            # Coût = Quantité * (Ask - Mid) si achat, ou (Mid - Bid) si vente
            # Simplification : Coût = Quantité * Prix * Transaction_Rate
            cost = 0
            for q, (_, opt) in zip(quantities, hedge_instruments.iterrows()):
                price = opt['mid']
                cost += abs(q) * price * self.transaction_costs_rate
            return cost

        # Contrainte Gamma : Gamma_Portefeuille + Gamma_Hedge = Target
        def constraint_gamma(quantities):
            hedge_gamma = np.sum(quantities * hedge_instruments['gamma'].values)
            return (current_gamma + hedge_gamma) - target_gamma

        # Contrainte Vega : Vega_Portefeuille + Vega_Hedge = Target
        def constraint_vega(quantities):
            hedge_vega = np.sum(quantities * hedge_instruments['vega'].values)
            return (current_vega + hedge_vega) - target_vega

        constraints = [
            {'type': 'eq', 'fun': constraint_gamma},
            # {'type': 'eq', 'fun': constraint_vega} # On peut désactiver Vega si c'est trop dur à résoudre
        ]
        
        # Bornes : On ne veut pas acheter 1 million d'options. Limite à +/- 100 contrats par instrument.
        bounds = [(-100, 100) for _ in range(n_instr)]

        # 4. Résolution
        try:
            result = minimize(cost_function, x0, constraints=constraints, bounds=bounds, method='SLSQP')
            
            if result.success:
                print("⚡ Solution de Hedging trouvée !")
                # Exécution des trades
                quantities = np.round(result.x) # On arrondit à l'entier le plus proche
                
                for qty, (_, opt) in zip(quantities, hedge_instruments.iterrows()):
                    if abs(qty) >= 1.0: # On ne trade pas des poussières
                        self.buy_option(opt, qty) # Cette méthode gère le cash et le log
                        
                print("🛡️ Hedging Gamma terminé.")
            else:
                print("❌ L'optimiseur n'a pas trouvé de solution (Contraintes trop fortes).")
                
        except Exception as e:
            print(f"⚠️ Erreur Optimizer: {e}")

    def close_position(self, client_supabase):
        """Ferme les options arrivées à expiration"""
        today = datetime.now(UTC).date()
        
        # 1. Récupérer positions ouvertes
        resp = client_supabase.table("complex_portfolio_options").select("*").eq("status", "open").execute()
        
        # 2. Récupérer prix spot du jour
        price_resp = client_supabase.table("prices").select("close").order("asof", desc=True).limit(1).execute()
        if not price_resp.data: return
        S0 = price_resp.data[0]['close']

        for pos in resp.data:
            # Parsing date
            try:
                expiry_dt = datetime.strptime(pos['expiry'], "%Y-%m-%d").date()
            except:
                continue # Skip format date invalide

            if expiry_dt <= today:
                # Calcul Payoff
                is_call = 'C' in pos['contract_symbol']
                strike = pos['strike']
                payoff = max(0.0, S0 - strike) if is_call else max(0.0, strike - S0)
                
                cash_in = payoff * pos['quantity']
                self.cash_balance += cash_in
                
                # Update DB
                client_supabase.table("complex_portfolio_options").update({
                    "status": "closed",
                    "prix_fermeture": payoff,
                    "date_fermeture": today.isoformat()
                }).eq("id", pos['id']).execute() # Utiliser ID unique est plus sûr
                
                print(f"🏁 Expiration {pos['contract_symbol']}. Payoff: {cash_in:.2f}")

    def calculer_NAV_journalier(self, client_supabase):
        """Calcule la valeur liquidative (Net Asset Value)"""
        # NAV = Cash + Valeur Stocks + Valeur Options
        
        # 1. Valeur Stocks
        price_resp = client_supabase.table("prices").select("close").order("asof", desc=True).limit(1).execute()
        spot = price_resp.data[0]['close']
        valeur_stocks = self.quantity_assets * spot
        
        # 2. Valeur Options (Mark-to-Market)
        # On utilise le dernier prix connu ou BS_price
        valeur_options = 0
        if self.open_positions:
            valeur_options = sum(pos.get('prix_fermeture', 0) * pos['quantity'] if pos['status']=='closed' 
                                 else pos.get('entry_price', 0) * pos['quantity'] for pos in self.open_positions)
            # Note: Pour faire propre, il faudrait récupérer le prix actuel de l'option, pas le prix d'entrée.
        
        self.current_nav = self.cash_balance + valeur_stocks + valeur_options
        print(f"💵 NAV Estimée: {self.current_nav:.2f} (Cash: {self.cash_balance:.2f})")
        return self.current_nav

    def to_supabase_portfolio(self, client_supabase):
        """Sauvegarde l'état journalier"""
        today = datetime.now(UTC).date().isoformat()
        
        # On vérifie si on a déjà une entrée pour aujourd'hui pour éviter doublons (ou on update)
        try:
            client_supabase.table("daily_complex_portfolio_pnl").insert({
                "asof": today,
                "nav": self.current_nav,
                "cash_balance": self.cash_balance,
                "quantity_assets": self.quantity_assets,
                "total_delta": self.delta_total,
                "total_gamma": self.total_gamma,
                "total_vega": self.total_vega
            }).execute()
            print("💾 État journalier sauvegardé.")
        except Exception as e:
            print(f"Info sauvegarde: {e}")

    @classmethod
    def load_from_supabase(cls, client_supabase):
        """Factory : Charge le dernier état connu"""
        resp = client_supabase.table("daily_complex_portfolio_pnl").select("*").order("asof", desc=True).limit(1).execute()
        if resp.data:
            last = resp.data[0]
            print(f"🔄 Portefeuille chargé (Date: {last['asof']})")
            return cls(
                cash_balance=last['cash_balance'],
                current_nav=last['nav'],
                quantity_assets=last['quantity_assets']
            )
        else:
            print("✨ Nouveau portefeuille initialisé.")
            return cls()

def run_daily_strategy(client_supabase, df_market_data):
    print("\n--- 🚀 DÉMARRAGE STRATÉGIE ---")
    
    # 1. Chargement
    ptf = Portfolio.load_from_supabase(client_supabase)
    
    # 2. Gestion Expirations (Nettoyage)
    ptf.close_position(client_supabase)
    
    # 3. Trading Alpha (Nouveaux paris)
    df_opportunities = ptf.form_option_df(df_market_data)
    opps = ptf.identify_mispriced_options(df_opportunities)
    ptf.trade_arbitrage(opps)
    
    # --- 4. HEDGING AVANCÉ (GAMMA) ---
    # On lui donne les options disponibles aujourd'hui pour se couvrir
    ptf.optimize_greeks_hedging(df_opportunities, target_gamma=0)
    
    # --- 5. HEDGING FINAL (DELTA SPOT) ---
    # On finit par le Spot car c'est le moins cher pour nettoyer le Delta restant
    ptf.achat_actif_delta_hedge(client_supabase)
    
    # 6. Reporting
    ptf.calculer_NAV_journalier(client_supabase)
    ptf.to_supabase_portfolio(client_supabase)
    
    return ptf


if __name__ == "__main__":
    # Simulation d'un DataFrame d'options pour le test
    # Dans la réalité, tu charges ça depuis ta DB ou ton CSV
    try:
        price = supabase_client.table("prices").select("*").order("asof", desc=True).limit(1).execute()
        spot_price = price.data[0]['close']
        df_complex = pd.DataFrame(supabase_client.table("vol_surfaces").select("*").execute().data)
        df_complex = df_complex[df_complex["asof"]== datetime.now(UTC).date().isoformat()]
        df_complex["S0"] = spot_price
        portfolio = run_daily_strategy(supabase_client, df_complex)
    except FileNotFoundError:
        print("⚠️ Fichier 'df_train_ready.csv' manquant pour le test.")