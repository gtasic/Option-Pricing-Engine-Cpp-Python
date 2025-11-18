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


asof = datetime.now(timezone.utc).date() 

supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_client = supabase.create_client(supabase_url,supabase_key)

np.random.seed(42)
df=pd.DataFrame(supabase_client.table("vol_surfaces").select("*").execute().data)
df_prix = pd.DataFrame(supabase_client.table("prices").select("*").execute().data)
df_prix["log_returns"] = np.log(df_prix['close'] / df_prix['close'].shift(1))
df_prix['hist_vol_10d'] = df_prix["log_returns"].rolling(window=10).std() * np.sqrt(252)
target=5
df_prix['target_realized_vol_10d'] = df_prix["log_returns"].rolling(window=target).std().shift(-target) * np.sqrt(252)


df_final = pd.merge(
    df,                             
    df_prix[["asof", "log_returns","hist_vol_10d", "target_realized_vol_10d"]],
    on="asof",                      
    how="left"      )               



df_final = df_final.dropna()


df_final = df_final[df_final["iv"]>0.01]
features = ['iv', 'moneyness', 'tenor', 'hist_vol_10d']
target = 'target_realized_vol_10d'

split_index = int(len(df_final) * 0.80)
X_train = df_final[features].iloc[:split_index]
y_train = df_final[target].iloc[:split_index]

X_test = df_final[features].iloc[split_index:]
y_test = df_final[target].iloc[split_index:]



model = xgb.XGBRegressor(
    n_estimators=1000,       
    learning_rate=0.05,      
    max_depth=5,             
    subsample=0.8,           
    colsample_bytree=0.8,    
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train, 
)


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

df_predict = pd.DataFrame([{"iv": 0.4,  'moneyness':0.9 ,'tenor':0.43 ,'hist_vol_10d': 0.2}, {"iv": 0.3,  'moneyness':0.7 ,'tenor':0.2 ,'hist_vol_10d': 0.2}])
print(model.predict(df_predict))