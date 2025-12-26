import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import supabase
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta, timezone
import sabr 
import backtest as bt
import calibration as cal

asof = datetime.now(timezone.utc).date() 

supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_client = supabase.create_client(supabase_url,supabase_key)

np.random.seed(42)
df=pd.DataFrame(supabase_client.table("vol_surfaces").select("*").execute().data)
df_prix = pd.DataFrame(supabase_client.table("prices").select("*").execute().data)
df_prix["log_returns"] = np.log(df_prix['close'] / df_prix['close'].shift(1))
df_prix['hist_vol_10d'] = df_prix["log_returns"].rolling(window=5).std() * np.sqrt(252)
target=10
df_prix['target_realized_vol_10d'] = df_prix["log_returns"].rolling(window=target).std().shift(-target) * np.sqrt(252)




df_final = pd.merge(
    df,                             
    df_prix[["asof", "log_returns","hist_vol_10d", "target_realized_vol_10d","close"]],
    on="asof",                      
    how="left"      )               



#df_final = df_final.dropna()

df_final = df_final[df_final["iv"]>0.01]


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


processed_days = [] 
unique_dates = df_final['asof'].unique()

print(f"Démarrage de la construction du Dataset sur {len(unique_dates)} jours...")

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
        df_day_enriched = prepare_ml_features(df_day, h_params, sabr_dict_day, S0)
        print(df_day_enriched[['asof', 'iv', 'H_kappa', 'S_rho', 'SABR_Edge']].head())
        processed_days.append(df_day_enriched)

if processed_days:
    df_train_ready = pd.concat(processed_days, ignore_index=True)
    print("✅ Dataset construit avec succès !")
    print(df_train_ready[['asof', 'iv', 'H_kappa', 'S_rho', 'SABR_Edge']].head())
else:
    print("❌ Aucun jour n'a pu être traité.")
#df_train_ready = df_train_ready.dropna()
df_train_ready.to_csv("df_train_ready_good.csv", index=False)


features = ["tenor","moneyness","iv",
            "hist_vol_10d","log_moneyness",
            "H_kappa","H_volvol","H_diff_term",
            "S_rho","S_nu","SABR_Edge"]

target = 'target_realized_vol_10d'

df_final = pd.read_csv("df_train_ready_good.csv")
df_final = df_final.drop(["expiry", "delta"], axis=1)
df_final = df_final.dropna()
split_index = int(len(df_final) * 0.80)
X_train = df_final[features].iloc[:split_index]
y_train = df_final[target].iloc[:split_index]

X_test = df_final[features].iloc[split_index:]
y_test = df_final[target].iloc[split_index:]



model = xgb.XGBRegressor(
    n_estimators=1000,       
    learning_rate=0.01,      
    max_depth=3,
    min_child_weight=2,             
    subsample=0.7,           
    colsample_bytree=0.7 ,    
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train, 
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)


model.save_model("volatility_model_v1.json")


y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n--- Résultats ---")
print(f"RMSE (Écart moyen) : {rmse:.4f}")
print(f"MAE (Erreur absolue moyenne) : {mae:.4f}")


results = pd.DataFrame({'Actual_RV': y_test, 'Predicted_RV': y_pred}, index=X_test.index)
results['Market_IV'] = X_test['iv'] 
marge = 0.02 
results['contract_symbol'] = df_final.loc[X_test.index, 'contract_symbol']
results['Signal_Achat'] = results['Predicted_RV'] > (results['Market_IV'] + marge)
print(results)
xgb.plot_importance(model, max_num_features=10)
#plt.title("Qu'est-ce qui prédit le mieux la Volatilité Future ?")
#plt.savefig("ML_sigma.png")
results.to_csv("ML_sigma_results.csv", index=False)



