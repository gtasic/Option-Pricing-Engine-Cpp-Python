import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn
import os
import supabase
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta, timezone
import pandas as pd

asof = datetime.now(timezone.utc).date() 

supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_client = supabase.create_client(supabase_url,supabase_key)




def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    if abs(F - K) < 1e-6:  
        return alpha / (F ** (1 - beta)) * (1 + ((1-beta)**2/24 * alpha**2/(F**(2-2*beta)) + 
                                                    0.25 * rho * beta * nu * alpha / (F**(1-beta)) + 
                                                    (2 - 3*rho**2)/24 * nu**2) * T)
    else:  
        z = (nu / alpha) * (F * K)**((1-beta)/2) * np.log(F / K)
        x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        num = alpha
        denom = (F * K)**((1-beta)/2) * (1 + (1-beta)**2/24 * (np.log(F/K))**2 + 
                                          (1-beta)**4/1920 * (np.log(F/K))**4)
        
        term2 = 1 + ((1-beta)**2/24 * alpha**2/((F*K)**(1-beta)) + 
                     0.25 * rho * beta * nu * alpha / ((F*K)**((1-beta)/2)) + 
                     (2 - 3*rho**2)/24 * nu**2) * T
        
        return (num / denom) * (z / x_z) * term2

def calibrate_sabr(strikes, market_ivs, F, T):
    def objective(params):
        alpha, beta, rho, nu = params
        beta = 0.85
        model_ivs = [sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes]
        return np.sum((np.array(market_ivs) - np.array(model_ivs))**2)
    

    result = minimize(objective, 
                     x0=[0.2, 0.85, 0, 0.3],  # Guess initial
                     bounds=[(0.01, 1), (0, 1), (-0.99, 0.99), (0.01, 1)])
    
    return result.x



df=pd.DataFrame(supabase_client.table("simulation_params").select("*").execute().data)

maturity = set(df["T"])

for x in maturity : 
    df_matu = df[df["T"]==x]

    ivs = list(df_matu["sigma"])
    S0 = df_matu["S0"].iloc[0] -10
    strikes = list(df_matu["strike"])
    alpha, beta, rho, nu = calibrate_sabr(strikes, ivs, S0, x)

    print(f"SABR calibrated for {x*365:.1f} days: α={alpha:.3f}, β={beta:.3f}, ρ={rho:.3f}, ν={nu:.3f}")

    
strikes = np.array([90, 95, 100, 105, 110])
market_ivs = np.array([0.25, 0.22, 0.20, 0.21, 0.23])  # Smile
 
alpha, beta, rho, nu = calibrate_sabr(strikes, market_ivs, F=100, T=1.0)

print(f"SABR calibrated: α={alpha:.3f}, β={beta:.3f}, ρ={rho:.3f}, ν={nu:.3f}")



# Dans volatility_models.py

def sabr_calibration_with_diagnostics(market_data):
    results = {}
    
    for T in market_data['T'].unique():
        df_slice = market_data[market_data['T'] == T]
        strikes = df_slice['strike'].values
        market_ivs = df_slice['sigma'].values
        F = df_slice['S0'].iloc[0]  # Forward = spot si r=q=0
        
        # Calibration
        alpha, beta, rho, nu = calibrate_sabr(strikes, market_ivs, F, T)
        
        # Diagnostics
        model_ivs = [sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes]
        rmse = np.sqrt(np.mean((market_ivs - model_ivs)**2))
        
        # Check arbitrage-free (densité positive)
        strikes_fine = np.linspace(strikes.min(), strikes.max(), 100)
        ivs_fine = [sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes_fine]
        
        # Density check: d²C/dK² > 0
        call_prices = [fn.call_price(fn.BS_parametres(F, K, T, 0, iv)) for K, iv in zip(strikes_fine, ivs_fine)]
        second_deriv = np.gradient(np.gradient(call_prices, strikes_fine), strikes_fine)
        
        arbitrage_free = np.all(second_deriv > -1e-6)  # Tolérance numérique
        
        results[T] = {
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'nu': nu,
            'rmse': rmse,
            'arbitrage_free': arbitrage_free,
            'market_ivs': market_ivs,
            'model_ivs': model_ivs,
            "strikes": strikes
        }
        
        print(f"Maturity {T:.2f}Y: RMSE={rmse:.4f}, Arb-free={'✅' if arbitrage_free else '❌'}")
    
    return results

# Visualisation
def plot_sabr_fit(results):
    """Graphique de la qualité du fit"""
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    strike =[]
    model_iv =[]
    maturity = []
    for i, (T, res) in enumerate(results.items()):
        ax = axes[i] if len(results) > 1 else axes
        strike.append(res['strikes'])
        model_iv.append( res['model_ivs'])
        maturity.append(T)
        strikes = np.linspace(80, 120, 50)
        model_ivs = [sabr_implied_vol(100, K, T, res['alpha'], res['beta'], 
                                       res['rho'], res['nu']) for K in strikes]
        
        ax.plot(strikes, model_ivs, 'b-', label='SABR fit', linewidth=2)
        ax.scatter(res['strikes'], res['market_ivs'], color='red', 
                   label='Market', s=50, zorder=5)
        ax.set_title(f"T={T:.1f}Y (RMSE={res['rmse']:.4f})")
        ax.set_xlabel('Strike')
        ax.set_ylabel('Implied Vol')
        ax.legend()
        ax.grid(alpha=0.3)
  
    plt.tight_layout()
    plt.savefig('sabr_calibration_quality.png', dpi=150)
    print("📊 Saved: sabr_calibration_quality.png")

    model_iv[0] =[float(i) for i in model_iv[0]]
    strike[0] =[float(i) for i in strike[0]]
    fig = go.Figure(data=[go.Mesh3d(x=list(maturity), y=strike[0], z=model_iv[0], opacity=0.6, color='lightblue', )])
    fig.update_layout(scene=dict(
                    xaxis_title='Maturity',
                    yaxis_title='Strike',
                    zaxis_title='Implied Volatility'),
                  title='Volatility Surface')

    print("📊 Displayed: SABR Volatility Surface")
    return fig
plot_sabr_fit((sabr_calibration_with_diagnostics(df)))