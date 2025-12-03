import supabase 
import os
import numpy as np
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import ML_sigma as ml
import backtest as bt
import calibration as cal
import sabr 
import delta_hedging as dh


class VolatilityArbitrageStrategy(dh.Portfolio):

    def __init__(self, portfolio, ml_model, option):
        self.Portfolio = dh.Portfolio()
        self.ml_model = ml_model 
        self.option = option


    def form_option_df(self, df_final): 
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
            df_train_ready.to_csv("df_train_ready.csv", index=False)

        self.option = df_train_ready
        return df_train_ready 
                
    def identify_mispriced_options(self, options_universe, spot, historical_vols):

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


        print("The possible trades are :" , opportunities)

        self.opportunities = opportunities
        return opportunities
    
    def execute_trade(self, opportunity):
  
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
            'vega': option['vega'],
            'status': 'open'
        })
        
        print(f"✅ Bought {option['contract_symbol']}: IV={entry_iv:.2%}, "
              f"Forecast RV={opportunity['forecasted_vol']:.2%}, Edge={opportunity['edge']:.2%}")
    
    def daily_rebalance(self):

        total_delta = self.portfolio.compute_total_delta()
        
        if abs(total_delta) > 0.05:
            hedge_adjustment = -total_delta
            self.portfolio.buy_stock(hedge_adjustment)
            
            cost = abs(hedge_adjustment) * self.current_spot * 0.001
            self.portfolio.cash -= cost
    
    def monitor_positions(self):

        for position in self.portfolio.open_positions:
            entry_date = position['entry_date']
            realized_vol_actual = self.compute_realized_vol_since(entry_date)
            
            forecast_error = realized_vol_actual - position['forecasted_rv']
            
            theoretical_pnl = position['vega'] * (realized_vol_actual - position['entry_iv'])
            actual_pnl = self.compute_position_pnl(position)
            
            print(f"Position {position['option_symbol']}:")
            print(f"  Entry IV: {position['entry_iv']:.2%}")
            print(f"  Forecasted RV: {position['forecasted_rv']:.2%}")
            print(f"  Actual RV: {realized_vol_actual:.2%}")
            print(f"  Theoretical P&L: {theoretical_pnl:.2f}€")
            print(f"  Actual P&L: {actual_pnl:.2f}€")

