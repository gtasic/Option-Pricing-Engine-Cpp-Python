import supabase 
import pandas as pd 
from datetime import datetime, timedelta, timezone
import test
import sys
sys.path.append("/workspaces/finance-/build")
UTC = timezone.utc
import finance
from copy import deepcopy
from dotenv import load_dotenv
import os
import logging

load_dotenv()
supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")


supabase = supabase.create_client(supabase_url, supabase_key)

date_target = [7,30,60,90,180]
delta_target = [0.10,0.25,0.5,0.75, 0.9]

date_target_arbitrage = [7,14,21,30,40]  #les dates pour l'arbitrage complexe

asof = datetime.now(timezone.utc).date() 

def daily_choice_15(df_simulation, dte_target, delta_target, vol_min=50, oi_min=100):
    cand = df_simulation[(df_simulation['bid']>0) & (df_simulation['ask']>df_simulation['bid'])]
    cand = cand[(cand['volume']>=vol_min) | (cand['openInterest']>=oi_min)]
    cand = cand[(cand['T']*365 >= dte_target-10) & (cand['T']*365 <= dte_target+10)]
    if cand.empty:
        cand = df_simulation[(df_simulation['bid']>0) & (df_simulation['ask']>df_simulation['bid'])]
        cand = cand[(cand['volume']>=vol_min) | (cand['openInterest']>=oi_min)]
    cand = cand.assign(
       d_delta = (cand['delta'] - delta_target).abs(),
       spread = cand['ask'] - cand['bid'],
       mid = (cand['ask'] + cand['bid'])/2,
       spread_rel = (cand['ask'] - cand['bid']) / ((cand['ask'] + cand['bid'])/2 + 1e-12)
    )
    cand = cand.sort_values(['d_delta','spread_rel','openInterest','volume'], ascending=[True, True, False, False])
    if cand.empty:
        logging.warning(f"No candidates found for DTE target {dte_target} and delta target {delta_target}")
        return None  
    return cand.iloc[0]


def final_pd(df_simu) :
# on aurait pu vectoriser avec .apply() pour optimisation future (gain ~10x performance), c'est une fonctionnalité à implémenter 
# Fonctionne actuellement pour datasets <1000 lignes
    df_simu["r"] = 0.04
    for i in range(len(df_simu["S0"])) : 
        BS_para = finance.BS_parametres(df_simu["S0"].iloc[i],df_simu["strike"].iloc[i],
                                        df_simu["T"].iloc[i],df_simu["r"].iloc[i],df_simu["sigma"].iloc[i]) 
        MC_para = finance.MC_parametres(10000,10000,float(df_simu["S0"].iloc[i]),float(df_simu["strike"].iloc[i]),float(df_simu["T"].iloc[i]),float(df_simu["r"].iloc[i]),float(df_simu["sigma"].iloc[i]))
      #  MC_para.nb_simulations, MC_para.nb_paths, MC_para.S0, MC_para.K, MC_para.T, MC_para.r, MC_para.sigma = 10000,10000,df_simu["S0"].iloc[i],df_simu["strike"].iloc[i],df_simu["T"].iloc[i],df_simu["r"].iloc[i],df_simu["sigma"].iloc[i]
        CRR_para = finance.tree_parametres(df_simu["S0"].iloc[i],df_simu["strike"].iloc[i],df_simu["T"].iloc[i],df_simu["r"].iloc[i],df_simu["sigma"].iloc[i], 1000)
        
       # CRR_para.S0, CRR_para.K, CRR_para.T, CRR_para.r, CRR_para.sigma, CRR_para.N = df_simu["S0"].iloc[i],df_simu["strike"].iloc[i],df_simu["T"].iloc[i],df_simu["r"].iloc[i],df_simu["sigma"].iloc[i], 1000

        df_simu["gamma"].iloc[i] = finance.call_gamma(BS_para)
        df_simu["vega"].iloc[i] = finance.call_vega(BS_para)
        df_simu[ "theta"].iloc[i] = finance.call_theta(BS_para)
        df_simu[ "rho"].iloc[i] = finance.call_rho(BS_para)
        df_simu[ "BS_price"].iloc[i] = finance.call_price(BS_para) 
        df_simu[ "MC_price"].iloc[i] = finance.monte_carlo_call(MC_para)
        df_simu[ "CRR_price"].iloc[i] = finance.tree(CRR_para)

    return df_simu
   



def choix_df(date_target, delta_target) :         
    rows = []

    for x in date_target : 
        for y in delta_target : 
            choice = daily_choice_15(test.df_simu[test.df_simu["asof"]== asof], x, y)
            daily_dict = {}
            rows.append(choice)


    df_choix_15 = pd.DataFrame(rows, columns= [
        "asset_id", "asof", "contract_symbol", "expiry", "strike",
        "S0", "T", "r", "sigma","delta" ,"bid", "ask" , "openInterest" , "volume", "d_delta" , "spread" , "mid" , "spread_rel",
                 "gamma" , "vega" , "theta" , "rho" , 
        "BS_price" , "MC_price" , "CRR_price"    
    ])

    final_pd(df_choix_15).to_csv("data.csv")

    final = final_pd(df_choix_15)
    final = final.drop_duplicates(subset=['contract_symbol'])  #on enlève les doublons
    return final

final = choix_df(date_target, delta_target)  #tous les jours on implémente nos données dans notre table 

print(final)
dict_daily = supabase.table("portfolio_options").select("*").execute().data
dict_daily = pd.DataFrame(dict_daily)

for x in final["contract_symbol"] :
    if x in dict_daily["contract_symbol"].values :
        final = final[final["contract_symbol"] != x]

print(final)


for i in range(len(final)) :
    try :
        supabase.table("daily_choice").insert({
            "asset_id": int(final["asset_id"].iloc[i]),
            "asof": datetime.now(UTC).date().isoformat(),
            "contract_symbol": final["contract_symbol"].iloc[i],
            "expiry": (final["expiry"].iloc[i]).isoformat(),
            "strike": float(final["strike"].iloc[i]),
            "S0": float(final["S0"].iloc[i]),
            "T": float(final["T"].iloc[i]),
            "r": float(final["r"].iloc[i]),
            "sigma": float(final["sigma"].iloc[i]),
            "delta": float(final["delta"].iloc[i]),
            "bid": float(final["bid"].iloc[i]),
            "ask": float(final["ask"].iloc[i]),
            "openInterest": int(final["openInterest"].iloc[i]),
            "volume": int(final["volume"].iloc[i]),
            "gamma": float(final["gamma"].iloc[i]),
            "vega": float(final["vega"].iloc[i]),
            "theta": float(final["theta"].iloc[i]),
            "rho": float(final["rho"].iloc[i]),
            "BS_price": float(final["BS_price"].iloc[i]),
            "MC_price": float(final["MC_price"].iloc[i]),
            "CRR_price": float(final["CRR_price"].iloc[i])
        }
        ).execute()
    except Exception as e: 
        logging.error(f"Error inserting row {i} with contract_symbol {final['contract_symbol'].iloc[i]}: {e}")




