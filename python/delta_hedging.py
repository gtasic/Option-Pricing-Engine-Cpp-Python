import supabase
import pandas as pd
import numpy as np
from datetime import date, datetime
from zoneinfo import ZoneInfo
UTC = ZoneInfo("UTC")
import os
from numpy import random
#changer le prix de cloture des options
import plotly.express as px
import plotly.graph_objects as go
import backtest
from dotenv import load_dotenv
import sys
sys.path.append("/workspaces/finance-/build")
import finance
from copy import deepcopy
from scipy.stats import skew, kurtosis

load_dotenv()
supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")


supabase_client = supabase.create_client(supabase_url, supabase_key)


class Portfolio:
    def __init__(self, cash_balance=100000.0, current_nav=10000.0, transaction_costs_rate=0.001, quantity_assets=0.0, delta_total=0.0, total_gamma=0.0, total_vega=0.0, total_theta=0.0, total_rho=0.0,total_sigma = 0.0 ):
        self.options = []           # Liste des options dans le portefeuille
        self.cash_balance = cash_balance # Capital initial du portefeuille
        self.current_nav = current_nav  # Valeur actuelle du portefeuille
        self.transaction_costs_rate = 0.001 # 0.1% de frais de transaction
        self.quantity_assets = quantity_assets  # Quantité de l'actif sous-jacent détenue
        self.delta_total = delta_total    # Delta total du portefeuille
        self.total_gamma = total_gamma
        self.total_vega = total_vega
        self.total_theta = total_theta
        self.total_rho = total_rho
        self.total_sigma = total_sigma




    def insrer_deux_options(self, supabase_client, df_daily_choice):
        print(df_daily_choice)
        if len(df_daily_choice) < 2:
            print("Pas assez d'options disponibles pour en sélectionner deux.")
            return None, None

        selected_options = df_daily_choice[df_daily_choice["asof"]== datetime.now(UTC).date()].sample(n=2, random_state=random.randint(0, 15))
        option1 = selected_options.iloc[0]
        option2 = selected_options.iloc[1]
        if option1['contract_symbol'] not in supabase_client.table("portfolio_options").select("contract_symbol").execute().data and option2['contract_symbol'] not in supabase_client.table("portfolio_options").select("contract_symbol").execute().data:
            print(f"Option 1 sélectionnée : {option1['contract_symbol']}, Strike : {option1['strike']}, T : {option1['T']}, Prix : {option1['mid']}")
            print(f"Option 2 sélectionnée : {option2['contract_symbol']}, Strike : {option2['strike']}, T : {option2['T']}, Prix : {option2['mid']}")



            cout_total_option1 = option1['mid'] * 10 * (1 + self.transaction_costs_rate)
            self.cash_balance -= cout_total_option1
            cout_total_option2 = option2['mid'] * 10 * (1 + self.transaction_costs_rate)
            self.cash_balance -= cout_total_option2
            cout_frais = option1['mid'] * 10 * self.transaction_costs_rate + option2['mid'] * 10 * self.transaction_costs_rate
            print(f"Coût total pour l'achat de 10 unités de chaque option (avec frais) : {cout_total_option1 + cout_total_option2:.2f} EUR")



            response = supabase_client.table("portfolio_options").insert([
                    {
                        "date_ouverture": datetime.now(UTC).date().isoformat(),
                        "contract_symbol": option1['contract_symbol'],
                        "expiry" : option1['expiry'].isoformat(),
                        "quantity": 10,
                        "strike": round(float(option1['strike']),2),
                        "T": option1['T'],
                        "prix": round(float(option1['mid']),2),
                        "sigma" : round(float(option1['sigma']),2),
                        "delta": round(float(option1['delta']),2),
                        "gamma": round(float(option1['gamma']),2),
                        "vega": round(float(option1['vega']),2),
                        "theta": round(float(option1['theta']),2),
                        "rho": round(float(option1['rho']),2),
                        "volume": int(option1['volume']) if not pd.isna(option1['volume']) else None,
                        "openInterest": int(option1['openInterest']) if not pd.isna(option1['openInterest']) else None,
                        "status": "open",
                        "comment": f"option achetée le  {datetime.now(UTC).date().isoformat()}"
                    }]).execute()
            supabase_client.table("trades_info").insert({
                "date_real": datetime.now(UTC).date().isoformat(),
                "transaction_cost": cout_total_option1,
                "description": f"Achat de 10 unités de l'option {option1['contract_symbol']} avec des frais de transaction de {self.transaction_costs_rate*100:.2f}%"
            }).execute()
        
            response = supabase_client.table("portfolio_options").insert([
                    {
                        "date_ouverture": datetime.now(UTC).date().isoformat(),
                        "contract_symbol": option2['contract_symbol'],
                      "expiry" : option2['expiry'].isoformat(),
                        "quantity": 10,
                        "strike": round(float(option2['strike']),2),
                        "T": option2['T'],
                        "prix": round(float(option2['mid']),2),
                        "sigma" : round(float(option2['sigma']),2),
                        "delta": round(float(option2['delta']),2),
                        "gamma": round(float(option2['gamma']),2),
                        "vega": round(float(option2['vega']),2),
                        "theta": round(float(option2['theta']),2),
                        "rho": round(float(option2['rho']),2),
                        "volume": int(option2['volume']) if not pd.isna(option2['volume']) else None,
                        "openInterest": int(option2['openInterest']) if not pd.isna(option2['openInterest']) else None,
                        "status": "open",
                        "comment": "option achetée le " + datetime.now(UTC).date().isoformat()

                    }]).execute()
            supabase_client.table("trades_info").insert({
                "date_real": datetime.now(UTC).date().isoformat(),
                "transaction_cost": cout_total_option2,
                "description": f"Achat de 10 unités de l'option {option2['contract_symbol']} avec des frais de transaction de {self.transaction_costs_rate*100:.2f}%"
            }).execute()


    """
    def transaction_costs_deux_options(supabase_client):
        global CURRENT_NAV  # c'est la valeur actuelle du portefeuille donc cash + valeur des options + valeur des actions de couverture
        global CASH_BALANCE # l'argent cash disponible dans le portefeuille, la trésorerie
        response = supabase_client.table("portfolio_options").select("*").execute()
        open_positions = response.data
        df = pd.DataFrame(open_positions)
        df = df[df["date_ouverture"] == datetime.now(UTC).date().isoformat()]
        open_positions = [pos for pos in open_positions if pos["date_ouverture"] == datetime.now(UTC).date().isoformat()]

        total_transaction_costs = 0.0
        for position in open_positions:
            prix = position['prix'] * position['quantity']    #On suppose qu'on achète au prix mid les options
            transaction_costs = prix * TRANSACTION_COSTS_RATE # Frais de transaction à l'achat
            total_transaction_costs += transaction_costs + prix
        supabase_client.table("trades_info").insert({
            "date_real": datetime.now(UTC).date().isoformat(),
            "transaction_cost": total_transaction_costs,
            "description": f"Frais de transaction pour l'ouverture de deux options {open_positions[0]['contract_symbol']} et {open_positions[1]['contract_symbol']} d'une quantité de {open_positions[0]['quantity']} et {open_positions[1]['quantity']} respectivement avec des frais de transaction de {TRANSACTION_COSTS_RATE*100:.2f}%"
        }).execute()
        CASH_BALANCE -= total_transaction_costs  #On met à jour le cash disponible après le cout total des transactions

        print(f"Frais de transaction pour l'ouverture de deux options : {total_transaction_costs:.2f} EUR")
        print(f"Valeur actuelle du portefeuille après frais de transaction : {CURRENT_NAV:.2f} EUR")
    """

    """
    def close_position(supabase_client, asof):
        global CURRENT_NAV
        response = supabase_client.table("portfolio_options").select("*").gt(asof , "expiry" ).execute()
        position = response.data
        if not position:
            print(f"Aucune position trouvée pour la date d'échéance {asof}.")
            return

        for pos in position:
            prix_fermeture = supabase_client.table("simu_param").select("mid").eq(["contract_symbol","asof"], [pos["contract_symbol"],datetime.now(UTC).date().isoformat()]).eq("strike", pos['strike']).execute()
            CURRENT_NAV += prix_fermeture * (1 - TRANSACTION_COSTS_RATE)* pos['quantity']
            prix_trade = prix_fermeture * (1 - TRANSACTION_COSTS_RATE)
            supabase_client.table("portfolio_options").update({
                "date_fermeture": datetime.now(UTC).date().isoformat(),
                "prix_fermeture": float(prix_fermeture),
                "status": "closed"
            }).eq("contract_symbol", pos["contract_symbol"]).execute()
        supabase_client.table("daily_trades").insert({
                "date": datetime.now(UTC).date().isoformat(),
                "transaction_costs": prix_trade ,
                "description": f"Fermeture de la position {pos['contract_symbol']}"
            }).execute()
    """

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
            frais = prix_fermeture_cash * self.transaction_costs_rate
            self.cash_balance += cash_recu
    #        CURRENT_NAV -= prix_fermeture_cash  # On retire la valeur des options fermées de la NAV
            supabase_client.table("portfolio_options").update({
                "date_fermeture": asof_str,
                "prix_fermeture": float(payoff),
                "status": "closed"
            }).eq("contract_symbol", pos["contract_symbol"]).execute() # On ferme la position en actualisant la valeur de statut à "closed" et la valeur de prix_fermeture au Payoff
            
            print(f"Fermeture Payoff de {pos['contract_symbol']}: {payoff:.2f} EUR/unité. Cash net: {cash_recu:.2f}")

    def actualiser_delta(self, client_supabase): #chaque jour on met à jour le delta de chaque option "open" dans le portefeuille pour recalculer le delta total du portefeuille et faire le delta hedging
        response = client_supabase.table("portfolio_options").select("*").eq("status", "open").execute()
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
        
        response = client_supabase.table("portfolio_options").select("*").eq("status", "open").execute()
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
        response = supabase_client.table("portfolio_options").select("*").eq("status", "open").execute()
        open_positions = response.data

        delta_options = sum(pos['delta'] * pos['quantity'] for pos in open_positions)
        self.total_gamma = sum(pos['gamma'] * pos['quantity'] for pos in open_positions)
        self.total_vega = sum(pos['vega'] * pos['quantity'] for pos in open_positions)
        self.total_theta = sum(pos['theta'] * pos['quantity'] for pos in open_positions)
        self.total_rho = sum(pos['rho'] * pos['quantity'] for pos in open_positions)
        self.total_sigma = sum(pos['sigma']  for pos in open_positions)/len(open_positions) if len(open_positions)>0 else 0.0


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
        frais_transaction = abs(quantite_actif_achat) * prix_actif * self.transaction_costs_rate
    #    CURRENT_NAV += quantite_actif_achat * prix_actif - cout_transaction  # Met à jour la valeur actuelle du portefeuille après l'achat/vente et les frais de transaction
        self.cash_balance-= cout_transaction  # Met à jour le cash disponible après l'achat/vente et les frais de transaction

        supabase_client.table("trades_info").insert({
        "date_real": datetime.now(UTC).date().isoformat(),
        "transaction_cost": cout_total,
        "description": f"Delta hedging: {'Achat' if quantite_actif_achat > 0 else 'Vente'} de {abs(quantite_actif_achat):.2f} unités à {prix_actif:.2f} EUR"
        }).execute()
        
        print(f"{'Achat' if quantite_actif_achat > 0 else 'Vente'} de {abs(quantite_actif_achat):.2f} unités")
        print(f"Position totale en actif: {self.quantity_assets:.2f}")

    def daily_pnl(self, supabase_client):
        response = supabase_client.table("daily_portfolio_pnl").select("*").order("asof", desc=True).limit(2).execute()
        values = response.data
        val = pd.DataFrame(values)
        asof = datetime.now(UTC).date().isoformat()

        if len(values) < 2:
            print("Pas assez de données pour calculer le PnL quotidien.")
            return None

        pnl = val['nav'].iloc[-1] - val['nav'].iloc[-2]
        print(f"PnL quotidien : {pnl:.2f} EUR")
        supabase_client.table("daily_portfolio_pnl").update({
            "daily_pnl": float(pnl)
        }).eq("asof", val['asof'].iloc[-1]).execute()
        return pnl




    def calculer_NAV_journalier(self, supabase_client):
        current_date_str = datetime.now(UTC).date().isoformat()
        
        prix_actif_response = supabase_client.table("prices").select().execute()
        if not prix_actif_response.data:
            print(f"Erreur: Prix S0 non trouvé pour {current_date_str}")
            return self.cash_balance # Retourne uniquement le cash si S0 est manquant
            
        prix_actif = prix_actif_response.data[0]['close']
        valeur_actions = self.quantity_assets * prix_actif

        response = supabase_client.table("portfolio_options").select("*").eq("status", "open").execute()
        open_positions = response.data
        valeur_options = 0.0

        for pos in open_positions:
            price_response = supabase_client.table("simulation_params").select().eq(
                "contract_symbol", 
                pos["contract_symbol"]
            ).execute()
            df_price = pd.DataFrame(price_response.data)
            df_price = df_price[df_price['asof'] == datetime.now(UTC).date().isoformat()]

            if not df_price.empty:
                prix_mid_D = (df_price.iloc[0]['bid'] + df_price.iloc[0]['ask']) / 2
                valeur_options += prix_mid_D * pos['quantity']
            else:
                print(f"Avertissement: Prix mid non trouvé pour {pos['contract_symbol']}")

        self.current_nav = valeur_actions + valeur_options + self.cash_balance # on ne rajoute pas le cash car on ne veut que la valeur des actifs
        print(f"NAV journalier au {current_date_str} : {self.current_nav:.2f} EUR")
        return self.current_nav



    # On peut aussi faire des graphiques de l'évolution du portefeuille avec toutes les métriques qui nous intéressent comme le pnl , NAV, Sharpe, RatioP&L cumulé, Sortino, max drawdown, avg gain/loss, turnover, VaR/CVaR.

    def metrics_portfolio(self, supabase_client) :
        df_met = supabase_client.table("daily_portfolio_pnl").select("*").execute().data
        df_portfolio = pd.DataFrame(df_met)
        df_portfolio['pnl_cumule'] = df_portfolio['daily_pnl'].cumsum()
        returns = df_portfolio['daily_pnl'][1:].values

        df_portfolio['rolling_max'] = df_portfolio['nav'].cummax()
        df_portfolio['drawdown'] = df_portfolio['nav'] - df_portfolio['rolling_max']
        df_portfolio['drawdown_pct'] = df_portfolio['drawdown'] / df_portfolio['rolling_max'].replace(0, 0.01)
        max_drawdown = df_portfolio['drawdown_pct'].min()
        avg_gain = df_portfolio[df_portfolio['daily_pnl'] > 0]['daily_pnl'].mean()
        avg_loss = df_portfolio[df_portfolio['daily_pnl'] < 0]['daily_pnl'].mean() 
        sharpe_ratio = (df_portfolio['daily_pnl'].mean() / df_portfolio['daily_pnl'].std()) * np.sqrt(252) if df_portfolio['daily_pnl'].std() != 0 else 0
        sortino_ratio = (df_portfolio['daily_pnl'].mean() / df_portfolio[df_portfolio['daily_pnl'] < 0]['daily_pnl'].std()) * np.sqrt(252) if df_portfolio[df_portfolio['daily_pnl'] < 0]['daily_pnl'].std() != 0 and not np.nan else 0
     #   turnover = df_portfolio['transaction_costs'].sum() / df_portfolio['nav'].mean() if df_portfolio['nav'].mean() != 0 else np.nan
        var_95 = np.percentile(df_portfolio['daily_pnl'], 5) if not np.nan else 0
        cvar_95 = df_portfolio[df_portfolio['daily_pnl'] <= var_95]['daily_pnl'].mean() if not np.nan else 0
 
        metrics = {
            "nav": df_portfolio['nav'].iloc[-1],
            "pnl": df_portfolio['daily_pnl'].iloc[-1],
            "max_drawdown": max_drawdown,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
       #     "turnover": turnover,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "kurtosis" :  kurtosis(returns) ,
            'skewness':   skew(returns) 
        }


        return metrics
    



    def metrics_portfolio(self, supabase_client):
        data = supabase_client.table("daily_portfolio_pnl").select("*").execute().data
        df = pd.DataFrame(data)
        
        df['daily_pnl'] = pd.to_numeric(df['daily_pnl'], errors='coerce').fillna(0)
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df['pnl_cumule'] = df['daily_pnl'].cumsum()
        df['rolling_max'] = df['nav'].cummax()
        df['drawdown'] = df['nav'] - df['rolling_max']
        df['drawdown_pct'] = df['drawdown'] / df['rolling_max'].replace(0, np.nan)        
        avg_gain = df[df['daily_pnl'] > 0]['daily_pnl'].mean()
        avg_loss = df[df['daily_pnl'] < 0]['daily_pnl'].mean()
        std_dev = df['daily_pnl'].std()        
        negative_returns = df[df['daily_pnl'] < 0]['daily_pnl']
        downside_std = negative_returns.std()

        if std_dev > 0:
            sharpe_ratio = (df['daily_pnl'].mean() / std_dev) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        if pd.notna(downside_std) and downside_std > 0:
            sortino_ratio = (df['daily_pnl'].mean() / downside_std) * np.sqrt(252)
        else:
            sortino_ratio = 0
        if len(df) > 5: 
            var_95 = np.nanpercentile(df['daily_pnl'], 5) 
            
            cvar_95 = df[df['daily_pnl'] <= var_95]['daily_pnl'].mean()
        else:
            var_95 = 0
            cvar_95 = 0    
        skewness = df['daily_pnl'].skew()
        kurtosis = df['daily_pnl'].kurtosis()

        return {
            "nav": df['nav'].iloc[-1],
            "pnl": df['daily_pnl'].iloc[-1],
            "max_drawdown": df['drawdown_pct'].min(),
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "var_95": var_95,
            "cvar_95": cvar_95 if pd.notna(cvar_95) else 0,
            "skewness": skewness if pd.notna(skewness) else 0,
            "kurtosis": kurtosis if pd.notna(kurtosis) else 0,
            "avg_gain": avg_gain if pd.notna(avg_gain) else 0,
            "avg_loss": avg_loss if pd.notna(avg_loss) else 0
        }



  
    def to_metrics_portfolio(self, supabase_client, metrics) :
        date_today = datetime.now(UTC).date().isoformat()
        supabase_client.table("portfolio_metrics").insert({
            "date" : date_today,
            "nav": metrics['nav'],
            "pnl": metrics['pnl'],
            "max_drawdown": metrics["max_drawdown"],
            "avg_gain": metrics["avg_gain"],
            "avg_loss": metrics["avg_loss"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
      #      "turnover": metrics["turnover"],
            "var_95": metrics["var_95"]  ,
            "cvar_95": metrics["cvar_95"]  , 
            "kurtosis" : metrics["kurtosis"], 
             'skewness':metrics["skewness"] 
        }).execute()




    def visualisation_portfolio(self, df_portfolio):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_portfolio['date'], y=df_portfolio['nav'], mode='lines+markers', name='NAV', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_portfolio['date'], y=df_portfolio['pnl_cumule'], mode='lines+markers', name='PnL Cumulé', line=dict(color='orange')))
        fig.update_layout(title='Évolution du Portefeuille', xaxis_title='Date', yaxis_title='Valeur (EUR)', legend_title='Légende')
        fig.show()
        return fig


    def to_supabase_portfolio(self, supabase_client):
        date_today = datetime.now(UTC).date().isoformat()
        buy_price_asset = supabase_client.table("prices").select().execute().data[-1]['close'] 
        supabase_client.table("daily_portfolio_pnl").insert({
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
        response = supabase_client.table("daily_portfolio_pnl").select("*").order(
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
    portfolio.insrer_deux_options(supabase_client, df_daily_choice)
    
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
    df_daily_choice = backtest.final
    portfolio = run_daily_strategy(supabase_client, df_daily_choice)