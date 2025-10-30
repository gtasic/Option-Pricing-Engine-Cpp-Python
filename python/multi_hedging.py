from scipy.optimize import minimize
import supabase
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import backtest
supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_client = supabase.create_client(supabase_url, supabase_key)

class MultiGreeksHedger:
    def __init__(self, portfolio, available_options):
        self.portfolio = portfolio  # Les positions actuelles donc les portfolio_options open
        self.hedging_instruments = available_options  # Options du marché disopnibles donc backtest.final
    
    def compute_hedge_positions(self, target_delta=0, target_gamma=0, target_vega=0):
        
        
        current_delta = 0.25 #self.portfolio.delta_total
        current_gamma = 1.6  #self.portfolio.total_gamma
        current_vega = 5700 #self.portfolio.total_vega
        
        # Instruments de hedge (2-3 options ATM de maturités différentes)
        hedge_opts = self.hedging_instruments[:3]
        
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
            return hedge_positions, result.fun
        else:
            return self._delta_hedge_only(), None
    
    def _delta_hedge_only(self):
        quantity_spot = -self.portfolio.delta_total
        return {'SPOT': quantity_spot}, None
    

df_port = supabase_client.table("portfolio_options").select("*").execute()
df_portfolio = pd.DataFrame(df_port)

df_available_options = backtest.final

hedger = MultiGreeksHedger(df_portfolio, df_available_options)
hedge_trades, transaction_cost = hedger.compute_hedge_positions(
    target_delta=0, 
    target_gamma=0.05,  
    target_vega=0
)

print(f"Hedge positions: {hedge_trades}")
print(f"Transaction cost: {transaction_cost:.2f}€")