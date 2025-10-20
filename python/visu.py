import pandas as pd
import numpy as np
import backtest
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import test
import plotly
UTC = ZoneInfo("UTC")


from copy import deepcopy


df = backtest.final  # df avec les 15 options choisies chaque jour
df_total = test.df_simu  
df_total = df_total 
df_total[['gamma','vega','theta','rho','BS_price','MC_price','CRR_price']] = np.nan
print(df_total.head())
df_total = backtest.final_pd(df_total)
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






from matplotlib.widgets import CheckButtons

df_vizu = change_df(df_total)
market_prices = df_vizu['mid']
bs_errors = df_vizu['err_BS']
mc_errors = df_vizu['err_MC']
crr_errors = df_vizu['err_CRR']

fig, ax = plt.subplots(figsize=(10,5))
plt.subplots_adjust(left=0.1, right=0.8)  # espace pour les boutons

p_bs = ax.scatter(market_prices, bs_errors, label='BS', alpha=0.8)
p_mc = ax.scatter(market_prices, mc_errors, label='MC', alpha=0.8)
p_crr = ax.scatter(market_prices, crr_errors, label='CRR', alpha=0.8)

ax.set_xlabel("Market Mid Price")
ax.set_ylabel("Error Model Price")
ax.set_title("Model Predictions vs Market Prices")
ax.legend(loc='upper left')

# Axe pour les CheckButtons
rax = plt.axes([0.85, 0.4, 0.12, 0.15])  # x, y, width, height (fraction of figure)
labels = ['BS', 'MC', 'CRR']
visibility = [True, True, True]
check = CheckButtons(rax, labels, visibility)

def func(label):
    if label == 'BS':
        p_bs.set_visible(not p_bs.get_visible())
    elif label == 'MC':
        p_mc.set_visible(not p_mc.get_visible())
    elif label == 'CRR':
        p_crr.set_visible(not p_crr.get_visible())
    plt.draw()

check.on_clicked(func)

plt.show()
plt.savefig('interactive_model_vs_market_price.png')




visualiser_erreur(change_df(df_total), 'model_vs_market_price.png')