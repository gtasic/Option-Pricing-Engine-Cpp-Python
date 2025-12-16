import os
from dotenv import dotenv_values
# load .env manually to avoid load_dotenv find issues
env_path = '/workspaces/finance-/python/.env'
if os.path.exists(env_path):
    vals = dotenv_values(env_path)
    for k,v in vals.items():
        if v is not None:
            os.environ.setdefault(k, v)

import complex_complex
from datetime import datetime, timezone
from supabase import create_client
import pandas as pd

supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY')
print('SUPABASE_URL present?', bool(supabase_url))
print('SUPABASE_KEY present?', bool(supabase_key))
client = create_client(supabase_url, supabase_key)

cp = complex_complex.ComplexPortfolio.portfolio_builder(client)
prices = pd.DataFrame(client.table('prices').select('*').execute().data)
S0 = prices['close'].iloc[-1]
print('Latest S0:', S0)

# produce df_ML
try:
    df_ML = cp.df_for_ML(client, prices)
    print('df_ML shape:', df_ML.shape)
    print('df_ML columns:', df_ML.columns.tolist())
except Exception as e:
    print('Error building df_for_ML:', e)
    df_ML = None

if df_ML is not None:
    options_sorted = cp.mispriced_options_without_ML(client, df_ML)
    print('options_sorted shape:', options_sorted.shape)
    print('options_sorted columns:', options_sorted.columns.tolist())
    cp.new_options(options_sorted, client, datetime.now(timezone.utc).date(), S0)

