import numpy as np
from scipy.optimize import minimize
import sys
sys.path.append("/workspaces/finance-/build")
import pandas as pd
import finance as fn
import supabase
import os
from dotenv import load_dotenv
load_dotenv()
import backtest as bt

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)

def calibrate_heston_to_surface(vol_surface_df, S0, r):
    print(vol_surface_df)
    try : 
        market_data = vol_surface_df[['moneyness', 'T', 'sigma']].copy()
    except KeyError:
        market_data = vol_surface_df[['moneyness', 'tenor', 'iv']].copy()
    market_data["moneyness"] = market_data["moneyness"] * S0
    market_data = market_data.values

    
        
    def objective(params):
        v0, theta, kappa, sigma_v, rho = params
        
        if v0 <= 0 or theta <= 0 or kappa <= 0 or sigma_v <= 0:
            return 1e10
        if abs(rho) >= 1:
            return 1e10
        if 2 * kappa * theta < sigma_v**2:
            return 1e10
            
        error = 0.0
        for K, T, market_iv in market_data:
            heston_params = fn.HestonParams(float(S0), float(v0), float(r), float(kappa),float(theta),  float(sigma_v), float(rho))
            heston_price = fn.price_european_call_mc(heston_params, float(K), float(T), 252, 100, 42, False)
            
            bs_params = fn.BS_parametres(S0, K, T, r, market_iv)
            market_price = fn.call_price(bs_params)
            
            weight = np.exp(-abs(np.log(K/S0)))  
            error += weight * ((heston_price.price - market_price) / market_price)**2
            
        return error
    

        
    x0 = [0.04, 0.04, 2.0, 0.3, -0.7]
    
    bounds = [
    (0.001, 0.15),  # v0 (initial variance)
    (0.001, 0.15),  # theta (long-term variance)
    (0.1, 7.0),     # kappa (mean-reversion speed)
    (0.01, 0.8),    # sigma_v (vol of vol)
    (-0.99, -0.1)   # rho (correlation) - forcé négatif
]
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    return {
        'v0': result.x[0],
        'theta': result.x[1],
        'kappa': result.x[2],
        'sigma_v': result.x[3],
        'rho': result.x[4],
        'rmse': np.sqrt(result.fun / len(market_data))
    }

# Exemple d'utilisation


vol = supabase_client.table("vol_surfaces").select("*").execute()
vol_surface_df = pd.DataFrame(vol.data)
print(vol_surface_df[:5])
price = supabase_client.table("prices").select("*").execute()
price_df = pd.DataFrame(price.data)

prix = price_df["close"].iloc[-1]

"""bt.final["moneyness"] = bt.final["strike"] / prix
calibrated_params = calibrate_heston_to_surface(bt.final, prix, r=0.04)
print(f"Heston calibrated: κ={calibrated_params['kappa']:.2f}, "
      f"θ={calibrated_params['theta']:.4f}, RMSE={calibrated_params['rmse']:.2f}€ "
      f"sigma_v={calibrated_params['sigma_v']:.4f}, v0={calibrated_params['v0']:.4f}, rho={calibrated_params['rho']:.4f}")
"""
"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import finance as fn  # ton module C++ importé
from matplotlib import cm

vol = supabase_client.table("vol_surfaces").select("*").execute()
vol_surface_df = pd.DataFrame(vol.data)
# ---- Paramètres ----
S0 = 262  # ton prix spot
r = 0.04
params = {
    'v0': 0.0598,
    'theta': 0.0619,
    'kappa': 2.05,
    'sigma_v': 0.4994,
    'rho': -0.4568
}

moneyness_grid = np.linspace(0.7, 1.3, 20)
tenor_grid = np.linspace(0.02, 1.0, 10)

# On calcule la volatilité implicite Heston sur la grille
iv_heston = np.zeros((len(tenor_grid), len(moneyness_grid)))

for i, T in enumerate(tenor_grid):
    for j, m in enumerate(moneyness_grid):
        K = S0 * m
        heston_params = fn.HestonParams(S0, params['v0'], r, params['kappa'],
                                        params['theta'], params['sigma_v'], params['rho'])
        # prix sous Heston
        heston_price = fn.price_european_call_mc(heston_params, K, T, 252, 500, 42, False).price
        # inversion du prix → volatilité implicite équivalente (BS)
        iv = params['sigma_v']
        iv_heston[i, j] = iv

# ---- Surface du marché ----
market_surface = vol_surface_df.pivot_table(index='tenor', columns='moneyness', values='iv')

# ---- Graphique 3D : Surface Heston ----
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
M, T = np.meshgrid(moneyness_grid, tenor_grid)
ax.plot_surface(M, T, iv_heston, cmap=cm.viridis, alpha=0.8)
ax.set_title('Surface de volatilité Heston calibrée', fontsize=14)
ax.set_xlabel('Moneyness (K/S₀)')
ax.set_ylabel('Maturité (T)')
ax.set_zlabel('Volatilité implicite')
plt.show()

# ---- Comparaison : smiles par maturité ----
plt.figure(figsize=(8, 6))
for T in sorted(vol_surface_df['tenor'].unique())[:4]:  # 4 maturités
    df_t = vol_surface_df[vol_surface_df['tenor'] == T]
    plt.scatter(df_t['moneyness'], df_t['iv'], label=f'Marché T={T:.2f}', marker='x')

    # courbe Heston pour cette maturité
    iv_model = []
    for m in df_t['moneyness']:
        K = S0 * m
        heston_params = fn.HestonParams(S0, params['v0'], r, params['kappa'],
                                        params['theta'], params['sigma_v'], params['rho'])
        heston_price = fn.price_european_call_mc(heston_params, K, T, 252, 300, 42, False).price
        iv = params["sigma_v"]
        iv_model.append(iv)
    plt.plot(df_t['moneyness'], iv_model, label=f'Heston T={T:.2f}', linestyle='-')

plt.title("Comparaison des smiles : Marché vs Heston calibré")
plt.xlabel("Moneyness (K/S₀)")
plt.ylabel("Volatilité implicite")
plt.legend()
plt.grid(True)
plt.savefig("heston.png")




"""