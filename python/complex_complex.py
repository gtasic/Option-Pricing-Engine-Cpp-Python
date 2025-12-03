import pandas as pd 
import numpy as np 
import supabase 
import os 
import sys 
from datetime import datetime, date, timezone
import backtest
import xgboost as xgb

asof = datetime.now(timezone.utc).date() 

client_supabase = supabase.create_client("SUPABASE_URL", "SUPABASE_KEY")

class ComplexPortfolio:
    def __init__(self, cash_position, quantity_assets, options_list, options_available, delta_total, total_gamma, total_vega, total_theta, total_rho, total_sigma):
        self.options_list = options_list  # Liste des options dans le portefeuille
        self.options_available = options_available  # Options disponibles pour arbitrage
        self.sport_price = None
        self.cash_position = cash_position
        self.transaction_costs = 0.001 # Frais de transaction fixes par opération
        self.quantity_assets = quantity_assets
        self.delta_total =delta_total
        self.total_gamma = total_gamma
        self.total_vega = total_vega
        self.total_theta = total_theta
        self.total_rho = total_rho
        self.total_sigma = total_sigma


    def upadate_options_open(self, client_supabase) : 
        reponse = client_supabase.table("complex_portfolio_options").select("*").eq("status",open).execute()
        reponse = reponse.data
        df_options = pd.DataFrame(reponse)
        if not df_options: 
            print("Aucune option n'a été ouverte dans le portefeuille.")

        reponse2 = client_supabase.table("simulation_params").select("*").eq("asof",datetime.now(timezone.utc).date().isoformat()).execute()
        reponse2 = reponse2.data
        df_params = pd.DataFrame(reponse2)
        df_params[['gamma','vega','theta','rho','BS_price','MC_price','CRR_price']] = np.nan
        df_params = backtest.final_pd(df_params)
        df_params["prix"] =  (df_params["bid"] + df_params["ask"])/2

        for index, row in df_options.iterrows():
            df_option = df_params[df_params["contract_symbol"] == row["contract_symbol"]]
            if not df_option.empty:
                option_data = df_option.iloc[0]
                client.supabase.table("complex_portfolio_options").update({
                    "sigma": option_data["sigma"],
                    "delta": option_data["delta"],
                    "gamma": option_data["gamma"],
                    "vega": option_data["vega"],
                    "theta": option_data["theta"],
                    "rho": option_data["rho"],
                    "prix": option_data["prix"]
                }).eq("contract_symbol", row["id"]).execute()
            else:
                print(f"L'option avec le symbole {row['contract_symbol']} n'a pas été trouvée dans les paramètres de simulation.")
            

    def close_position(self,supabase_client):
        asof_date = datetime.now(timezone.utc).date()
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
            supabase_client.table("portfolio_options").update({
                "date_fermeture": asof_str,
                "prix_fermeture": float(payoff),
                "status": "closed"
            }).eq("contract_symbol", pos["contract_symbol"]).execute() # On ferme la position en actualisant la valeur de statut à "closed" et la valeur de prix_fermeture au Payoff
            
            print(f"Fermeture Payoff de {pos['contract_symbol']}: {payoff:.2f} EUR/unité. Cash net: {cash_recu:.2f}")


    def greeks_calcul(self, supabase_client):
        reponse = supabase_client.table("complex_portfolio_options").select("*").eq("status","open").execute()
        reponse = reponse.data
        df_options = pd.DataFrame(reponse)
        if df_options.empty: 
            print("Aucune option n'a été ouverte dans le portefeuille.")
            return
        self.delta_total = (df_options["delta"] * df_options["quantity"]).sum()
        self.total_gamma = (df_options["gamma"] * df_options["quantity"]).sum()
        self.total_vega = (df_options["vega"] * df_options["quantity"]).sum()
        self.total_theta = (df_options["theta"] * df_options["quantity"]).sum()
        self.total_rho = (df_options["rho"] * df_options["quantity"]).sum()
        self.total_sigma = (df_options["sigma"] * df_options["quantity"]).sum()

    
    def achat_vente_spot(self, supabase_client) : 
        prix_actif = supabase_client.table("prices").select("close").eq("asof",asof.isoformat()).execute()
        if not prix_actif.data:
            print(f"Erreur: Prix de l'actif non trouvé pour {asof.isoformat()}.")
            return
        S0 = prix_actif.data[0]['close']
        delta_needed = -self.delta_total + self.quantity_assets
        transaction_costs = abs(delta_needed) * S0 * self.transaction_costs_rate
        self.cash_position -= delta_needed * S0 + transaction_costs
        self.quantity_assets += delta_needed

        print(f"Achat/Vente Spot: {delta_needed} unités à {S0} EUR/unité. Coût des transactions: {transaction_costs} EUR.")


    def dataframe_for_ML(self, supabase_client): 
        reponse = bt.final_df
        self.options_available = reponse
        return self.options_available
    

    def mispriced_options(self,supabase_client):
        features = ["tenor","moneyness","iv",
            "hist_vol_10d","log_moneyness",
            "H_kappa","H_volvol","H_diff_term",
            "S_rho","S_nu","SABR_Edge"]
        model = xgb.XGBRegressor()
        model.load_model("volatility_model_v1.json") 
        y_pred = model.predict(self.options_available[features])
        self.options_available["predicted_iv"] = y_pred
        self.options_available["iv_diff"] = self.options_available["iv"] - self.options_available["predicted_iv"]
        options_available_sorted = self.options_available.sort_values(by="iv_diff", ascending=False)
        
        if options_available_sorted[options_available_sorted["iv_diff"] > 0].empty: 
            print("Aucune possibilité d'aribitrage détectée.")
        
        else :
            print("Options mispricées détectées pour arbitrage.")
        
        return options_available_sorted
    

    def multi_greeks_hedging(self, supabase_client):
        
        current_delta = self.delta_total
        current_gamma = self.total_gamma    
        current_vega = self.total_vega
        target_delta = 0.0
        target_gamma = 0.05
        target_vega = 0.0
    
        hedge_options = self.options_list.copy()
    
    
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
            hedge_positions = {
                opt.contract_symbol: qty 
                for opt, qty in zip(hedge_options.itertuples(), result.x) if abs(qty) > 0.01
            }
            final_quantities = round(result.x)
            return hedge_positions, final_quantities
        else:
            return self._delta_hedge_only(), None
        

    def rebalance_portfolio(self,supabase_client, hedge_positions,final_quantities):
        for contract_symbol, quantity in hedge_positions.items():
            prix_option_response = supabase_client.table("simulation_params").select("CRR_price").eq("contract_symbol", contract_symbol).execute()
            if not prix_option_response.data:
                print(f"Erreur: Prix de l'option non trouvé pour {contract_symbol}.")
                continue
            option_price = prix_option_response.data[0]['CRR_price']
            transaction_costs = abs(quantity) * option_price * self.transaction_costs_rate
            self.cash_position -= quantity * option_price + transaction_costs
        self.greeks_calcul(supabase_client)  # Met à jour les grecs après le rééquilibrage
        

    @classmethod
    def portfolio_builder(cls, supabase_client):
        reponse = supabase_client.table("complex_portfolio_pnl").select("*").execute()
        reponse = reponse.data
        df_portfolio = pd.DataFrame(reponse)
        return cls(
            cash_position = df_portfolio["cash_position"].iloc[-1],
            quantity_assets = df_portfolio["quantity_assets"].iloc[-1],
            options_list = None,
            options_available = None,
            delta_total = df_portfolio["delta_total"].iloc[-1],
            total_gamma = df_portfolio["total_gamma"].iloc[-1],
            total_vega = df_portfolio["total_vega"].iloc[-1],
            total_theta = df_portfolio["total_theta"].iloc[-1],
            total_rho = df_portfolio["total_rho"].iloc[-1],
            total_sigma = df_portfolio["total_sigma"].iloc[-1]
        )
    
if __name__ == "__main__" : 
    complex_portfolio = ComplexPortfolio.portfolio_builder(client_supabase)