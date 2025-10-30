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

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)

def calibrate_heston_to_surface(vol_surface_df, S0, r):
    market_data = vol_surface_df[['moneyness', 'tenor', 'iv']].copy()
    market_data["moneyness"] = market_data["moneyness"] * S0
    market_data = market_data.values

    
        
    def objective(params):
        v0, theta, kappa, sigma_v, rho = params
        
        if v0 <= 0 or theta <= 0 or kappa <= 0 or sigma_v <= 0:
            return 1e10
        if abs(rho) >= 1:
            return 1e10
        if 2 * kappa * theta <= sigma_v**2:
            return 1e10
            
        error = 0.0
        for K, T, market_iv in market_data:
            heston_params = fn.Heston_parametres(S0, v0, r, kappa,theta,  sigma_v, rho)
            heston_price = fn.price_call_european_mc(heston_params, K, T, M=252, N=10000, seed=42, anthetique=False)
            
            bs_params = fn.BS_parametres(S0, K, T, r, market_iv)
            market_price = fn.call_price(bs_params)
            
            weight = np.exp(-abs(np.log(K/S0)))  # Poids max pour ATM
            error += weight * (heston_price - market_price)**2
            
        return error
    
    # Guess initial (valeurs typiques)

        
    x0 = [0.04, 0.04, 2.0, 0.3, -0.7]
    
    # Optimisation avec contraintes
    bounds = [(0.001, 0.2), (0.001, 0.2), (0.1, 10), (0.01, 1.0), (-0.99, 0.99)]
    
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
print(vol_surface_df)
price = supabase_client.table("prices").select("*").execute()
price_df = pd.DataFrame(price.data)
prix = price_df["close"].iloc[-1]
calibrated_params = calibrate_heston_to_surface(vol_surface_df, prix, r=0.04)
print(f"Heston calibrated: κ={calibrated_params['kappa']:.2f}, "
      f"θ={calibrated_params['theta']:.4f}, RMSE={calibrated_params['rmse']:.2f}€")