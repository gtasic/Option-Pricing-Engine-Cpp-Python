import pandas as pd
import numpy as np
import backtest
from copy import deepcopy
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import test
import plotly
UTC = ZoneInfo("UTC")
import finance
import random
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from scipy import stats


df = backtest.final  # df avec les 15 options choisies chaque jour
df_total = test.df_simu  
df_total = df_total 
df_total[['gamma','vega','theta','rho','BS_price','MC_price','CRR_price']] = np.nan
print(df_total.head())
df_total = backtest.final_pd(df_total.sample(n=100))
df_total["mid"] = (df_total["bid"] + df_total["ask"])/2
df_total["spread"] = df_total["ask"] - df_total["bid"]
df_total["spread_rel"] = df_total["spread"] / (df_total["mid"] + 1e-12)
print(df_total.head())  #on a bien les greeks et les prix modèles pour toutes les options quotidiennement
maturite = [30,90,180]
delta = [0.25,0.5,0.81]

def change_df(df_ori) : 
    df = deepcopy(df_ori)
    for m in ['BS','MC','CRR']:
        df[f'err_{m}'] = df[f'{m}_price'] - df['mid']
        df[f'relerr_{m}'] = df[f'err_{m}'] / df['mid'].replace(0, np.nan)

    df['w_vega'] = df['vega'].fillna(0)
    df['w_vol']  = df['volume'].fillna(0)
    df['w_liq']  = 1 / df['spread_rel'].replace(0, np.nan).fillna(df['spread_rel'].median())
    return df

def weighted_rmse(arr_err, arr_w):
    w = arr_w.fillna(0).values
    e = arr_err.fillna(0).values
    denom = w.sum() if w.sum()>0 else len(e)
    return np.sqrt((w * e**2).sum() / denom)


def calcul_metrics(df) :
    metrics = {}
    asof_today = datetime.now(UTC).date()
    metrics['asof'] = asof_today
    for m in ['bs','mc','crr']:
        err = df[f'err_{m.upper()}']
        metrics[f'{m}_mae']  = float(err.abs().mean())
        metrics[f'{m}_rmse'] = float(np.sqrt((err**2).mean()))
        metrics[f'{m}_med']  = float(err.median())
        metrics[f'{m}_wrmse_vega'] = float(weighted_rmse(err, df['w_vega']))
        metrics[f'{m}_wrmse_vol']  = float(weighted_rmse(err, df['w_vol']))
    df_final = pd.DataFrame.from_dict([metrics])
    asof_today = datetime.now(UTC).date()
    df_final['asof'] = asof_today
    return(df_final)





def bucket_error(maturite , delta, df) : 
    for m in maturite : 
        for d in delta : 
            df_b = df[(df['T']*365 >= m-10) & (df['T']*365 <= m+10) & (df['delta'] >= d-0.1) & (df['delta'] <= d+0.1)]
            if not df_b.empty : 
                print(f"maturite {m} jours, delta {d} :")
                visualiser_erreur(df_b, f'png/m{m}_d{d}_model_vs_market_price.png')
                
                print("\n")


metrics =calcul_metrics(change_df(df))
print(metrics)
print("\n")
"""bucket_error(maturite, delta, change_df(df_total))  #avec cela on a les metrics par bucket chaque jour pour étudier l'évolution dans le temps
print("\n")

"""

def visualize_metrics(metrics):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    methods = ['bs', 'mc', 'crr']
    errors = ['mae', 'rmse', 'wrmse_vega', 'wrmse_vol']
    
    x = np.arange(len(errors))
    width = 0.2
    
    for i, method in enumerate(methods):
        y = [metrics[f'{method}_{err}'] for err in errors]
        ax.bar(x + i*width, y, width, label=method.upper())
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(errors)
    ax.set_ylabel('Error Value')
    ax.set_title('Option Pricing Errors by Method')
    ax.legend()
    
    plt.show()


def visualiser_erreur(df, nom_fichier='model_vs_market_price.png') :  
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    methods = ['BS', 'MC', 'CRR']
    colors = ['blue', 'orange', 'green']
    
    for method, color in zip(methods, colors):
        ax.scatter(df['T']*365, df[f'err_{method}'], alpha=0.5, label=method, color=color)
    
    max_price = max(df['mid'].max(), df[[f'err_{m}' for m in methods]].max().max())
    
    ax.set_xlabel('MATURITY (days)')
    ax.set_ylabel('Error Model Price')
    ax.set_title('Model Predictions BY MATURITY')
    ax.legend()
    
    plt.show()
    plt.savefig(f'{nom_fichier}')



def bucket_error(maturite , delta, df) : 
    for m in maturite : 
        for d in delta : 
            df_b = df[(df['T']*365 >= m-10) & (df['T']*365 <= m+10) & (df['delta'] >= d-0.1) & (df['delta'] <= d+0.1)]
            if not df_b.empty : 
                print(f"maturite {m} jours, delta {d} :")
                visualiser_erreur(df_b, f'png/m{m}_d{d}_model_vs_market_price.png')
            else : 
                print(f"maturite {m} jours, delta {d} : pas données")
                
                print("\n")



def visu_greeks(df) : 
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
    axs[0, 0].scatter(df_total['mid'], df_total['delta'], alpha=0.5)
    axs[0, 0].set_title('Delta vs Mid Price')
    axs[0, 0].set_xlabel('Mid Price')
    axs[0, 0].set_ylabel('Delta')
        
    axs[0, 1].scatter(df_total['mid'], df_total['gamma'], alpha=0.5, color='orange')
    axs[0, 1].set_title('Gamma vs Mid Price')
    axs[0, 1].set_xlabel('Mid Price')
    axs[0, 1].set_ylabel('Gamma')
        
    axs[1, 0].scatter(df_total['mid'], df_total['vega'], alpha=0.5, color='green')
    axs[1, 0].set_title('Vega vs Mid Price')
    axs[1, 0].set_xlabel('Mid Price')
    axs[1, 0].set_ylabel('Vega')
        
    axs[1, 1].scatter(df_total['mid'], df_total['theta'], alpha=0.5, color='red')
    axs[1, 1].set_title('Theta vs Mid Price')
    axs[1, 1].set_xlabel('Mid Price')
    axs[1, 1].set_ylabel('Theta')
        
    plt.tight_layout()
    plt.show()
    plt.savefig('greeks_vs_mid_price.png')


"""

visu_greeks(df_total)

"""
def etude_convergence_model_MCC_CRR(df, N_values=[50, 100,200, 500,750, 1000, 1500,3000,5000,7500]):
    all_results = []
    result_maes_mc = []
    result_maes_crr = []

    print("Début de l'étude de convergence...") 

    for N in N_values:
        print(f"Calcul pour N = {N}...") 
        
        crr_prices = []
        mc_prices = []

        for index, row in df.iterrows():
            CRR_para = finance.tree_parametres()
            CRR_para.S0, CRR_para.K, CRR_para.T, CRR_para.r, CRR_para.sigma, CRR_para.N = row['S0'], row['strike'], row['T'], row['r'], row['sigma'], N
            MC_para = finance.MC_parametres()
            MC_para.nb_simulations, MC_para.nb_paths, MC_para.S0, MC_para.K, MC_para.T, MC_para.r, MC_para.sigma = N, N, row['S0'], row['strike'], row['T'], row['r'], row['sigma']

            crr_p = finance.tree(CRR_para)
            mc_p = finance.monte_carlo_call(MC_para)
            
            crr_prices.append(crr_p)
            mc_prices.append(mc_p)

        temp_df = df.copy()
        temp_df['CRR_price'] = crr_prices
        temp_df['MC_price'] = mc_prices
        temp_df['err_CRR'] = temp_df['CRR_price'] - temp_df['mid']
        temp_df['err_MC'] = temp_df['MC_price'] - temp_df['mid']
        temp_df['N'] = N
        all_results.append(temp_df)
        err_mc = temp_df['err_MC']
        mae_mc = float(err_mc.abs().mean())
        result_maes_mc.append(mae_mc)
        print(f"MAE_mc pour N={N} : {mae_mc}") 

        err_crr = temp_df['err_CRR']
        mae_crr = float(err_crr.abs().mean())
        result_maes_crr.append(mae_crr)
        print(f"MAE_crr pour N={N} : {mae_crr}") 

    print("Fin des calculs, création du graphique...") 

    results_df = pd.concat(all_results)
    
    plt.figure(figsize=(14, 7))

    plt.plot(N_values, result_maes_mc, marker='o', label='MC Model Price Error', color='blue')
    plt.plot(N_values, result_maes_crr, marker='o', label='CRR Model Price Error', color='orange')

    plt.xlabel('Number of paths (N)')
    plt.ylabel('Model Price Error')
    plt.title('Convergence of MC and CRR Model Prices')
    plt.legend()
    
    plt.savefig('convergence_MC_CRR.png')
    plt.show()

    print("Graphique sauvegardé et affiché.")
    return results_df



maturite_bins = [0, 30, 90, 180, 365, 730] # Bins en jours (0-30j, 30-90j, etc.)
maturite_labels = ['0-30j', '30-90j', '90-180j', '180-365j', '365j+']
df_maturity = change_df(df_total)
df_maturity['maturity_bucket'] = pd.cut(df_maturity['T'] * 365, bins=maturite_bins, labels=maturite_labels)


df_long = pd.melt(
    df_maturity,
    id_vars=['maturity_bucket'],           
    value_vars=['err_BS', 'err_MC', 'err_CRR'],
    var_name='model',                     
    value_name='error'                     
)

def visualiser_erreur_boxplot(df_long, bucket_column='maturity_bucket'):
    """
    Crée un box plot comparant l'erreur des modèles pour chaque bucket.
    """
    plt.figure(figsize=(14, 8)) 

    sns.boxplot(
        data=df_long,
        x=bucket_column,  
        y='error',        
        hue='model'       
    )

    plt.title(f'Distribution de l\'Erreur des Modèles par {bucket_column}', fontsize=16)
    plt.xlabel('Bucket de Maturité', fontsize=12)
    plt.ylabel('Erreur de Prix (Modèle - Marché)', fontsize=12)
    plt.axhline(0, color='r', linestyle='--', linewidth=1.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('error_boxplot_by_maturity.png')
    plt.show()


#df_utile = test.df_simu[['S0','strike','T','r','sigma','mid']]


#etude_convergence_model_MCC_CRR(df_utile[df_utile["mid"]>1].sample(n=150, random_state=random.randint(0, 15)))




def analyze_model_risk(df_total, models=['BS', 'MC', 'CRR']):
    results = {}
    
    for model in models:
        df = df_total.copy()
        df['error'] = df[f'{model}_price'] - df['mid']
        df['rel_error'] = df['error'] / df['mid']
        df['moneyness'] = df['strike'] / df['S0']
        df['dte'] = df['T'] * 365
        
        # 1. Analyse par moneyness bucket
        df['money_bucket'] = pd.cut(df['moneyness'], 
                                     bins=[0, 0.9, 0.95, 1.05, 1.1, 2.0],
                                     labels=['Deep OTM', 'OTM', 'ATM', 'ITM', 'Deep ITM'])
        
        error_by_moneyness = df.groupby('money_bucket')['error'].agg(['mean', 'std', 'count'])
        
        # 2. Analyse par maturity bucket
        df['maturity_bucket'] = pd.cut(df['dte'], 
                                        bins=[0, 30, 90, 180, 365, 730],
                                        labels=['0-30d', '30-90d', '90-180d', '180-365d', '365d+'])
        
        error_by_maturity = df.groupby('maturity_bucket')['error'].agg(['mean', 'std', 'count'])
        
        t_stat, p_value = stats.ttest_1samp(df['error'].dropna(), 0)
        has_bias = p_value < 0.05
        
        X = df[['moneyness', 'dte']].dropna()
        y = df.loc[X.index, 'error'].dropna()
        X = X.sample(n= len(y))
        
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        
        results[model] = {
            'mean_error': df['error'].mean(),
            'mae': df['error'].abs().mean(),
            'rmse': np.sqrt((df['error']**2).mean()),
            "has_systematic_bias" : has_bias,
            'bias_direction': 'overpricing' if df['error'].mean() > 0 else 'underpricing',
            'error_by_moneyness': error_by_moneyness,
            'error_by_maturity': error_by_maturity,
            'r2_explainability': r2,  # % d'erreur expliquée par (K, T)
            'coef_moneyness': reg.coef_[0],
            'coef_maturity': reg.coef_[1]
        }
    
    best_model = min(results.keys(), key=lambda m: results[m]['mae'])
    
    print(f"\n=== Model Risk Analysis ===\n")
    
    for model, res in results.items():
        print(f"{model} Model:")
        print(f"error by mat {res["error_by_maturity"]}")
        print(f"error by mon {res["error_by_moneyness"]}")
        print(f"  MAE: {res['mae']:.2f}€  |  RMSE: {res['rmse']:.2f}€")
        print(f"  Bias: {res['bias_direction']} ({res['mean_error']:+.2f}€)")
        print(f"  Systematic? {'Yes (p<0.05)' if res['has_systematic_bias'] else 'No'}")
        print(f"  Error explained by (K,T): {res['r2_explainability']:.1%}\n")
    
    print(f"🏆 Best model: {best_model} (lowest MAE)")
    
    # 6. Recommandation
    print("\n💡 Recommendations:")
    if results['BS']['has_systematic_bias']:
        print("- BS shows systematic bias → Consider local vol model (Dupire)")
    if results['BS']['coef_maturity'] > 1.0:
        print("- Error increases with maturity → Stochastic vol (Heston) recommended")
    
    return results

# Utilisation
model_risk_results = analyze_model_risk(df_total)
