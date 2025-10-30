

import sys
sys.path.append("/workspaces/finance-/build")
import finance 
import supabase
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)

def stress_test_portfolio(portfolio, scenarios, options):

    results = {}
    
    for scenario_name, scenario_params in scenarios.items():
        # Choc du sous-jacent
        S_shocked = portfolio["buy_price_asset"] * (1 + scenario_params['spot_shock'])
        print("c est",S_shocked[1])
        
        # Choc de vol
        vol_shocked = 0.5 * (1 + scenario_params['vol_shock'])
        
        # Recalculer le portefeuille sous le scénario
        pnl_scenario = 0
        new_delta = 0
        new_gamma = 0
        new_vega = 0
        
        for option in options.iterrows():
            print(option[1]["strike"])
            # Nouveau prix sous scénario
            bs_shocked = finance.BS_parametres(
                S_shocked[1], 
                option[1]['strike'], 
                option[1]['T'], 
                0.04,   #le taux choisi
                vol_shocked
            )
            new_price = finance.call_price(bs_shocked)
            pnl_option = (new_price - option[1]['prix']) * option[1]['quantity']
            pnl_scenario += pnl_option
            
            new_delta += finance.call_delta(bs_shocked) * option[1]['quantity']
            new_gamma += finance.call_gamma(bs_shocked) * option[1]['quantity']
            new_vega += finance.call_vega(bs_shocked) * option[1]['quantity']
        
        # P&L du hedge en spot
        pnl_hedge = portfolio["quantity_assets"] * (S_shocked[1] - portfolio["buy_price_asset"])
        
        total_pnl = pnl_scenario + pnl_hedge
        
        results[scenario_name] = {
            'total_pnl': total_pnl,
            'pnl_pct': total_pnl / portfolio["nav"],
            'new_delta': new_delta,
            'new_gamma': new_gamma,
            'new_vega': new_vega,
            'still_delta_neutral': abs(new_delta) < 0.05
        }
    
    # Affichage
    print("\n=== Portfolio Stress Test Results ===\n")
    print(f"Current NAV: {portfolio["nav"].mean():,.0f}€\n")
    
    for scenario, res in results.items():
        status = "✓ Hedged" if res['still_delta_neutral'] else "✗ Exposed"
        print(f"{scenario}:")
        print(f"  P&L: {res['total_pnl'].mean():+,.0f}€ ({res['pnl_pct'].mean():+.2%})")
        print(f"  New Delta: {res['new_delta']:.3f} {status}")
        print()
    
    # Pire scénario
    worst_scenario = min(results.keys(), key=lambda s: results[s]['total_pnl'].mean())
    worst_loss = results[worst_scenario]['total_pnl'].mean()
    
    print(f"🚨 Worst-case scenario: {worst_scenario}")
    print(f"   Max loss: {worst_loss:,.0f}€ ({worst_loss/portfolio["nav"].mean():.2%} of NAV)\n")
    
    return results

# Définition des scénarios de manière ordonnée
stress_scenarios = {
    'Market Crash (-20%)': {'spot_shock': -0.20, 'vol_shock': 0.50},
    'Vol Spike': {'spot_shock': 0.0, 'vol_shock': 1.0},
    'Rally (+15%)': {'spot_shock': 0.15, 'vol_shock': -0.30},
    '2020 COVID (-30%, vol +100%)': {'spot_shock': -0.30, 'vol_shock': 1.0},
    'Flash Crash (-10% instant)': {'spot_shock': -0.10, 'vol_shock': 0.80}
}


portfolio = pd.DataFrame(supabase_client.table("daily_portfolio_pnl").select("*").execute().data)
options = pd.DataFrame(supabase_client.table("portfolio_options").select("*").execute().data)
stress_results = stress_test_portfolio(portfolio, stress_scenarios, options)