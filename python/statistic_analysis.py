import numpy as np
from scipy import stats
import pandas as pd
import supabase
import os 
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
client = supabase.create_client(supabase_url, supabase_key)

def analyze_hedging_performance(df_pnl):
    returns = df_pnl['daily_pnl'][1:].values
    _, p_value_normality = stats.shapiro(returns)
    
    adf_stat, p_value_adf, _, _, critical_values, _ = adfuller(returns)

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    n = len(returns)
    sharpe = mean_return / std_return * np.sqrt(252)
    
    sharpe_ci = bootstrap_sharpe_ci(returns, n_bootstrap=10000)
    
    t_stat = mean_return / (std_return / np.sqrt(n))
    p_value_sharpe = 1 - stats.t.cdf(t_stat, df=n-1)
    
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    results = {
        'sharpe_ratio': sharpe,
        'sharpe_ci_lower': sharpe_ci[0],
        'sharpe_ci_upper': sharpe_ci[1],
        'p_value_profitable': p_value_sharpe,
        'is_normal': p_value_normality > 0.05,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'n_observations': n
    }
    
    return results

def bootstrap_sharpe_ci(returns, n_bootstrap=10000, alpha=0.05):
    """Bootstrap pour intervalle de confiance du Sharpe"""
    sharpe_bootstrap = []
    n = len(returns)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        sharpe = np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252)
        sharpe_bootstrap.append(sharpe)
    
    return np.percentile(sharpe_bootstrap, [alpha/2*100, (1-alpha/2)*100])

df_portfolio = client.table('daily_portfolio_pnl').select('*').execute().data
portfolio_df = pd.DataFrame(df_portfolio)
results = analyze_hedging_performance(portfolio_df)

print(f"""
=== Statistical Analysis of Delta Hedging Strategy ===

Sharpe Ratio: {results['sharpe_ratio']:.2f}
95% CI: [{results['sharpe_ci_lower']:.2f}, {results['sharpe_ci_upper']:.2f}]
P-value (profitable): {results['p_value_profitable']:.4f} {'✓ Significant' if results['p_value_profitable'] < 0.05 else '✗ Not significant'}

Returns Distribution:
- Normality: {'Yes' if results['is_normal'] else 'No (fat tails)'}
- Skewness: {results['skewness']:.2f} {'(left tail)' if results['skewness'] < 0 else '(right tail)'}
- Excess Kurtosis: {results['kurtosis']:.2f}

Risk Metrics:
- VaR (95%): {results['var_95']:.2f}€
- CVaR (95%): {results['cvar_95']:.2f}€

Sample Size: {results['n_observations']} days
""")