import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os 
import supabase

load_dotenv()
supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

supabase_client = supabase.create_client(supabase_url, supabase_key)

def performance_attribution(df_portfolio): #Dans ce ^rogramme nous allons analyser la performance de la stratégie de hedging
    
    df = df_portfolio.copy()
    
    # Calcul des contributions quotidiennes
    df['theta_contrib'] = -df['total_theta'] / 365  # Theta par jour
    df['gamma_contrib'] = 0.5 * df['total_gamma'] * (df['buy_price_asset'].pct_change()**2)
    df['vega_contrib'] = df['total_vega'] * 0.015  #df['vol_change']
    df['rho_contrib'] = df['total_rho'] * df['rate_change'] if 'rate_change' in df else 0
    
    # Contribution du delta (ne devrait pas contribuer si parfaitement hedgé)
    df['delta_contrib'] = df['total_delta'] * df['buy_price_asset'].pct_change()
    
    # Résiduel inexpliqué
    df['unexplained'] = (df['daily_pnl'] - 
                         (df['theta_contrib'] + df['gamma_contrib'] + 
                          df['vega_contrib'] + df['rho_contrib'] + df['delta_contrib']))
    
    # Agrégation sur la période
    total_pnl = df['daily_pnl'].sum()
    
    attribution = {
        'Theta Decay': df['theta_contrib'].sum(),
        'Gamma P&L': df['gamma_contrib'].sum(),
        'Vega P&L': df['vega_contrib'].sum(),
        'Rho P&L': df['rho_contrib'].sum(),
        'Delta Leakage': df['delta_contrib'].sum(),  # Devrait être ~0
        'Unexplained': df['unexplained'].sum(),
        'Total': total_pnl
    }
    
    # Visualisation
    print("\n=== Performance Attribution ===\n")
    print(f"Total P&L: {total_pnl:+,.2f}€\n")
    print("Breakdown:\n")
    
    for source, pnl in attribution.items():
        if source == 'Total':
            print(f"\n{'-'*40}")
            print(f"{'Total':<20} {pnl:>10,.2f}€")
        else:
            pct = pnl / total_pnl * 100 if total_pnl != 0 else 0
            bar = '█' * int(abs(pct) / 2)
            sign = '+' if pnl >= 0 else '-'
            print(f"{source:<20} {sign}{abs(pnl):>9,.2f}€  ({pct:>5.1f}%)  {bar}")
    
    # Insights
    print("\n💡 Key Insights:")
    
    gamma_pct = attribution['Gamma P&L'] / total_pnl * 100
    if gamma_pct > 50:
        print(f"  - Strategy is capturing realized vol effectively ({gamma_pct:.0f}% of P&L)")
    
    theta_pct = abs(attribution['Theta Decay']) / abs(total_pnl) * 100
    if theta_pct > 30:
        print(f"  - High theta decay ({theta_pct:.0f}%) - consider shorter DTE options")
    
    if abs(attribution['Delta Leakage']) > total_pnl * 0.1:
        print(f"  ⚠️  Delta hedging not perfect - consider more frequent rebalancing")
    
    if abs(attribution['Unexplained']) > total_pnl * 0.2:
        print(f"  ⚠️  {attribution['Unexplained']/total_pnl:.0%} P&L unexplained - check model assumptions")
    
    return attribution

# Graphique de l'attribution
def plot_attribution(attribution):  
    data = {k: v for k, v in attribution.items() if k not in ['Total', 'Unexplained']}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Waterfall chart
    sources = list(data.keys())
    values = list(data.values())
    cumulative = np.cumsum([0] + values)
    
    colors = ['green' if v > 0 else 'red' for v in values]
    ax1.bar(sources, values, color=colors, alpha=0.7)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_title('P&L Attribution by Greek', fontsize=14, fontweight='bold')
    ax1.set_ylabel('P&L (€)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart (valeurs absolues)
    abs_values = [abs(v) for v in values]
    ax2.pie(abs_values, labels=sources, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Contribution Breakdown (Absolute)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_attribution.png', dpi=150)
    plt.show()

# Utilisation
df_portfolio = pd.DataFrame(supabase_client.table("daily_portfolio_pnl").select("*").execute().data)

attribution = performance_attribution(df_portfolio)
plot_attribution(attribution)