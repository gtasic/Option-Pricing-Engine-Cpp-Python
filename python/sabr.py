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
import backtest
asof = datetime.now(timezone.utc).date() 

supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_client = supabase.create_client(supabase_url,supabase_key)




def sabr_vol(k, f, t, alpha, beta, rho, volvol):
   
    if k <= 0 or f <= 0: return 0.0
    epsilon = 1e-8
    log_fk = np.log(f / k)
    fk_beta = (f * k)**((1 - beta) / 2.0)
    z = (volvol / alpha) * fk_beta * log_fk
    
    if abs(z) < epsilon:
        x_z = 1.0
    else:
        z = min(max(z, -0.999), 0.999) 
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        
    term1 = alpha / fk_beta
    term2 = 1 + ( ((1 - beta)**2 / 24.0) * log_fk**2 + ((1 - beta)**4 / 1920.0) * log_fk**4 )
    numerator = term1 * (1 + ( ((1 - beta)**2 / 24.0) * alpha**2 / ((f * k)**(1 - beta)) +
                               (1 / 4.0) * (rho * beta * volvol * alpha) / fk_beta +
                               ((2 - 3 * rho**2) / 24.0) * volvol**2 ) * t)
    
    denominator = term2 * (x_z / z if abs(z) > epsilon else 1.0)
    
    return numerator / denominator

def calibrate_sabr_robust(strikes, market_ivs, S0, T, r=0.04, q=0.0, beta=0.5):
 
    strikes = np.array(strikes)
    market_ivs = np.array(market_ivs)
    
    F = S0 * np.exp((r - q) * T)
    

    idx_atm = (np.abs(strikes - F)).argmin()
    iv_atm = market_ivs[idx_atm]
    
    alpha_guess = iv_atm * (F**(1 - beta))
    
    x0 = [alpha_guess, 0.0, 0.3] 
    
    # Contraintes : Alpha > 0, -1 < Rho < 1, Nu > 0
    bounds = [(1e-4, None), (-0.999, 0.999), (1e-4, 5.0)]
    
    def objective(params):
        a, r_val, n = params
        model_ivs = [sabr_vol(k, F, T, a, beta, r_val, n) for k in strikes]
        
        # Erreur quadratique pondérée (on punit plus les erreurs ATM que OTM)
        # Poids = 1 / Vega ou simplement Gaussian autour de l'ATM
        # Ici simple: somme des carrés
        return np.sum((market_ivs - np.array(model_ivs))**2)

    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        alpha_opt, rho_opt, nu_opt = res.x
        return alpha_opt, beta, rho_opt, nu_opt, res.fun # On renvoie aussi l'erreur finale
    except Exception as e:
        print(f"Calibration failed: {e}")
        return alpha_guess, beta, 0, 0, 999.0


df2 = pd.DataFrame(supabase_client.table("simulation_params").select("*").execute().data)
df = backtest.final
maturity = set(df["T"])
r_rate = 0.045 # Taux sans risque actuel (ex: 4.5%)

print(f"{'Maturity':<10} | {'Alpha':<8} | {'Beta':<5} | {'Rho':<8} | {'Nu':<8} | {'Error (RMSE)':<10}")
print("-" * 65)

for x in sorted(maturity): 
    df_matu = df[df["T"]==x].sort_values("strike")

    ivs = df_matu["sigma"].values
    strikes = df_matu["strike"].values
    S0 = df_matu["S0"].iloc[0] 
    
    # On fixe Beta à 0.5 (Standard Equity) ou 1.0 (LogNormal)
    alpha, beta, rho, nu, err = calibrate_sabr_robust(strikes, ivs, S0, x, r=r_rate, beta=0.5)

    rmse = np.sqrt(err / len(strikes))
    print(f"{x*365:5.1f} days | {alpha:.4f}   | {beta:.1f}   | {rho:+.4f}   | {nu:.4f}   | {rmse:.2e}")
"""    
strikes = np.array([90, 95, 100, 105, 110])
market_ivs = np.array([0.25, 0.22, 0.20, 0.21, 0.23])  # Smile
 
alpha, beta, rho, nu = calibrate_sabr(strikes, market_ivs, F=100, T=1.0)

print(f"SABR calibrated: α={alpha:.3f}, β={beta:.3f}, ρ={rho:.3f}, ν={nu:.3f}")
"""


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
"""plot_sabr_fit((sabr_calibration_with_diagnostics(df)))
plot_sabr_fit((sabr_calibration_with_diagnostics(df_2)))"""