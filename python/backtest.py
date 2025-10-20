import supabase 
import pandas as pd 
from datetime import datetime, timedelta, timezone
import test
import csv
import sys
sys.path.append("/workspaces/finance-/build")

import finance
from copy import deepcopy
# Quand pas assez de données sur le marché américain risque de doublons sur les options choisies !!!! Attention 

date_target = [7,30,60,90,180]
delta_target = [0.25,0.5, 0.81]



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
        return None  
    return cand.iloc[0]


def final_pd(df_simu) :

    for i in range(len(df_simu["bid"])) : 
        BS_para = finance.BS_parametres(df_simu["S0"].iloc[i],df_simu["strike"].iloc[i],
                                        df_simu["T"].iloc[i],df_simu["r"].iloc[i],df_simu["sigma"].iloc[i]) 
        MC_para = finance.MC_parametres()
        MC_para.nb_simulations, MC_para.nb_paths, MC_para.S0, MC_para.K, MC_para.T, MC_para.r, MC_para.sigma = 10000,10000,df_simu["S0"].iloc[i],df_simu["strike"].iloc[i],df_simu["T"].iloc[i],df_simu["r"].iloc[i],df_simu["sigma"].iloc[i]
        CRR_para = finance.tree_parametres()
        
        CRR_para.S0, CRR_para.K, CRR_para.T, CRR_para.r, CRR_para.sigma, CRR_para.N = df_simu["S0"].iloc[i],df_simu["strike"].iloc[i],df_simu["T"].iloc[i],df_simu["r"].iloc[i],df_simu["sigma"].iloc[i], 1000

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
            choice = daily_choice_15(test.df_simu, x, y)
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