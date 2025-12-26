import numpy as np
from scipy import stats
import pandas as pd
import supabase
import os 
from dotenv import load_dotenv
import sys
load_dotenv()
sys.path.append("/workspaces/finance-/build")
import finance as fn



supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
client = supabase.create_client(supabase_url, supabase_key)

df_stats = pd.DataFrame(client.table("daily_choice").select('*').limit(5).execute().data)
df_graph = df_stats = pd.DataFrame(client.table("daily_choice").select('*').execute().data)


N = [1,5,10,50,100,200,500,1500,2000,3000]
prix_MC= []
prix_CRR = []
prix_Heston = []
prix_marche = []


"""for i in range(len(N)): 
    mc_p =[]
    mc_crr =[]
    mc_heston =[]
    for index, row in df_stats.iterrows() :
        prix_marche = (row["bid"]+row["ask"])/2
        mc = fn.monte_carlo_call(fn.MC_parametres(N[i],N[i],float(row["S0"]),float(row["strike"]),float(row["T"]),float(row["r"]),float(row["sigma"])))
        crr = fn.tree(fn.tree_parametres(float(row["S0"]),float(row["strike"]),float(row["T"]),float(row["r"]),float(row["sigma"]),N[i]))
        heston_params = fn.HestonParams(float(row["S0"]), float(0.04), float(0.04), float(1),float(0.04),  float(0.2), float(-0.5))
        heston_price = fn.price_european_call_mc(heston_params, float(row["strike"]), float(row["T"]), N[i], N[i], 42, False).price
        mc_p.append(abs(mc-prix_marche)/prix_marche)
        mc_crr.append(abs(crr-prix_marche)/prix_marche)
        mc_heston.append(abs(heston_price-prix_marche)/prix_marche)
    prix_MC.append(round(float(np.mean(mc_p)),4))
    prix_CRR.append(round(float(np.mean(mc_crr)),4))
    prix_Heston.append(round(float(np.mean(mc_heston)),4))
    print(f"C'est bon pour N = {N[i]}")


print(prix_MC)
print(prix_CRR)
print(prix_Heston)


"""




df_graph["moneyness"] = df_graph["S0"]/df_graph["strike"]
df_graph["MAE_BS"] = abs(df_graph["BS_price"]- (df_graph["bid"]+df_graph["ask"])/2)/(df_graph["bid"]+df_graph["ask"])/2
df_graph["MAE_MC"] = abs(df_graph["MC_price"]- (df_graph["bid"]+df_graph["ask"])/2)/(df_graph["bid"]+df_graph["ask"])/2
df_graph["MAE_CRR"] = abs(df_graph["CRR_price"]- (df_graph["bid"]+df_graph["ask"])/2)/(df_graph["bid"]+df_graph["ask"])/2
df_graph["Heston"] = np.nan
for index, row in df_graph.iterrows() : 
        heston_params = fn.HestonParams(float(row['S0']), 0.04, 0.04, 1,0.04, 0.2, -0.5)
        heston_price = fn.price_european_call_mc(heston_params, row["strike"], row["T"], 252, 100, 42, False)
        df_graph.loc[index,"Heston"] = heston_price.price

df_graph["MAE_Heston"] = abs(df_graph["Heston"]- (df_graph["bid"]+df_graph["ask"])/2)/(df_graph["bid"]+df_graph["ask"])/2


df_graph.to_csv("/workspaces/finance-/csv/compa.csv")

