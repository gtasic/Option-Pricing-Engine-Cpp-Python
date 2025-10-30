from supabase import create_client, Client
import os
from dotenv import load_dotenv
import test
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd 
import numpy as np 
import logging
import backtest

load_dotenv()
supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")


supabase: Client = create_client(supabase_url, supabase_key)
print("Supabase client created successfully.")  

NYC = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

def df_to_records(df, date_cols= None, remove_dupli = True) :
    if df is None or df.empty:
        return []
    out = df.copy()

    if remove_dupli:
        initial_count = len(out)
        out = out.drop_duplicates()  
        final_count = len(out)
        if initial_count != final_count:
            print(f"⚠️  {initial_count - final_count} doublons EXACTS supprimés")

            
    for c in (date_cols or []):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c]).dt.date.astype(str)
    # NaN/inf -> None
    out = out.replace({np.nan: None})
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].apply(
                lambda x: None if (x is None or pd.isna(x)) else float(x)
            )
    return out.to_dict(orient="records")


def safe_upsert(table_name, records, batch_size=10):
    try:
        total_records = len(records)
        
        for i in range(0, total_records, batch_size):
            batch = records[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_records - 1) // batch_size + 1
            
            
            response = supabase.table(table_name).upsert(batch).execute()
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'upload vers {table_name}: {e}")
        print(f"Type d'erreur: {type(e).__name__}")
        
        if records:
            print(f"Échantillon de données: {records[0]}")
        
        return False

def test_connection():
   
    try:
        result = supabase.table("prices").select("*").limit(1).execute()
        return True
    except Exception as e:
        print(f"❌ Erreur de connexion Supabase: {e}")
        return False

def upload_daily_snapshot(asset_id, symbol, asof, r_flat=0.04):

    try:
        prices_df, quotes_df, vol_surface_df, sim_params_df = test.daily_snapshot_for_symbol(
            asset_id, symbol, asof, r_flat
        )
        prices_records = df_to_records(prices_df, date_cols=['asof'])
        quotes_records = df_to_records(quotes_df, date_cols=['asof', 'expiry'])
        vol_records = df_to_records(vol_surface_df, date_cols=['asof'])
        sim_records = df_to_records(sim_params_df, date_cols=['asof', 'expiry'])
  #      choice_records = df_to_records(backtest.final, date_cols= ["asof", "expiry"])
 #       options_pricing = df_to_records(visu.metrics, date_cols=["asof"])
        
        success_count = 0
        
        uploads = [
            ("prices", prices_records),
            ("option_quotes", quotes_records),
            ("vol_surfaces", vol_records),
            ("simulation_params", sim_records), 
      #      ("daily_choice", choice_records),
     #       ("option_pricing_metrics", options_pricing)
        ]
        
        for table_name, records in uploads:
            if safe_upsert(table_name, records):
                success_count += 1
            else:
                print(f"Échec de l'upload pour {table_name}")
        
        if success_count == len(uploads):
            print(f"🎉 Tous les uploads réussis! ({success_count}/{len(uploads)} tables)")
            return True
        else:
            print(f"⚠️  Upload partiel: {success_count}/{len(uploads)} tables réussies")
            return False
            
    except Exception as e:
        print(f"❌ Erreur générale lors du traitement: {e}")
        print(f"Type d'erreur: {type(e).__name__}")
        return False

def upload_multiple_symbols(symbols_config, asof, r_flat=0.04):
    results = {}
    
    for asset_id, symbol in symbols_config:
        print(f"\n{'='*50}")
        print(f"Traitement: {symbol}")
        print(f"{'='*50}")
        
        success = upload_daily_snapshot(asset_id, symbol, asof, r_flat)
        results[symbol] = success
        
        if success:
            print(f"✅ {symbol} terminé avec succès")
        else:
            print(f"❌ Échec pour {symbol}")
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n🏁 RÉSUMÉ FINAL:")
    print(f"   Réussis: {successful}/{total}")
    print(f"   Détails: {results}")
    
    return results

if __name__ == "__main__":
    asof_today = datetime.now(UTC).date()
    
    asset_id = 10101  
    symbol = "AAPL"
    to_sql_asset = df_to_records(test.to_asset_table(asset_id,symbol,"EQ"))
    supabase.table("assets").upsert(to_sql_asset).execute()
    # Upload simple
    success = upload_daily_snapshot(asset_id, symbol, asof_today, r_flat=0.04)
    
    if success:
        print(f"\n🎉 SUCCESS! Données de {symbol} uploadées avec succès!")
    else:
        print(f"\n💥 ÉCHEC! Problème lors de l'upload de {symbol}")

    

    """
    symbols_to_process = [
        (10101, "AAPL"),
        (10102, "MSFT"),
        (10103, "GOOGL"),
        (10104, "TSLA")
    ]
    
    results = upload_multiple_symbols(symbols_to_process, asof_today, r_flat=0.04)
    """


    
        
            
    