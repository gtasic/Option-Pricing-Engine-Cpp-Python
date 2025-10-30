import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import visu
def deep_dive_maturity_analysis(df_total):

    
    df = df_total.copy()
    
    # Créer les buckets de maturité
    df['dte'] = df['T'] * 365
    df['maturity_bucket'] = pd.cut(df['dte'], 
                                    bins=[0, 30, 90, 180, 365, 730],
                                    labels=['0-30d', '30-90d', '90-180d', 
                                           '180-365d', '365d+'])
    
    # Calculer les erreurs
    for model in ['BS', 'MC', 'CRR']:
        df[f'error_{model}'] = df[f'{model}_price'] - df['mid']
        df[f'abs_error_{model}'] = df[f'error_{model}'].abs()
    
    # === ANALYSE 1: Statistiques par bucket ===
    print("="*70)
    print("MATURITY BUCKET ANALYSIS - Statistical Summary")
    print("="*70)
    
    stats_by_bucket = {}
    
    for bucket in df['maturity_bucket'].cat.categories:
        df_bucket = df[df['maturity_bucket'] == bucket]
        
        if df_bucket.empty:
            continue
        
        stats_bucket = {}
        
        for model in ['BS', 'MC', 'CRR']:
            errors = df_bucket[f'error_{model}'].dropna()
            
            if len(errors) < 5:
                continue
            
            # Tests statistiques
            mean_err = errors.mean()
            std_err = errors.std()
            
            # Test de normalité
            _, p_normal = stats.shapiro(errors) if len(errors) < 5000 else (None, 0)
            
            # Test si erreur significativement différente de 0
            t_stat, p_value = stats.ttest_1samp(errors, 0)
            
            # Outliers (Tukey's method: Q1-1.5*IQR, Q3+1.5*IQR)
            q1, q3 = errors.quantile([0.25, 0.75])
            iqr = q3 - q1
            n_outliers = ((errors < q1 - 1.5*iqr) | (errors > q3 + 1.5*iqr)).sum()
            pct_outliers = n_outliers / len(errors) * 100
            
            stats_bucket[model] = {
                'mean': mean_err,
                'std': std_err,
                'mae': errors.abs().mean(),
                'bias_significant': p_value < 0.05,
                'p_value': p_value,
                'is_normal': p_normal > 0.05 if p_normal else False,
                'n_outliers': n_outliers,
                'pct_outliers': pct_outliers,
                'n_observations': len(errors)
            }
        
        stats_by_bucket[bucket] = stats_bucket
        
        # Affichage
        print(f"\n📊 {bucket}")
        print(f"   Observations: {len(df_bucket)}")
        
        for model in ['BS', 'MC', 'CRR']:
            if model in stats_bucket:
                s = stats_bucket[model]
                bias_indicator = "⚠️ BIASED" if s['bias_significant'] else "✓ Unbiased"
                print(f"\n   {model}:")
                print(f"      MAE: {s['mae']:.2f}€  |  Std: {s['std']:.2f}€")
                print(f"      Bias: {s['mean']:+.2f}€  {bias_indicator} (p={s['p_value']:.3f})")
                print(f"      Outliers: {s['n_outliers']} ({s['pct_outliers']:.1f}%)")
    
    # === ANALYSE 2: Régression erreur vs maturité ===
    print("\n" + "="*70)
    print("REGRESSION ANALYSIS: Error ~ Maturity")
    print("="*70 + "\n")
    
    from sklearn.linear_model import LinearRegression
    
    for model in ['BS', 'MC', 'CRR']:
        df_reg = df[['dte', f'error_{model}']].dropna()
        
        if df_reg.empty:
            continue
        
        X = df_reg[['dte']].values
        y = df_reg[f'error_{model}'].values
        
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        
        # Significativité du slope
        from scipy.stats import t as t_dist
        n = len(y)
        residuals = y - reg.predict(X)
        mse = np.sum(residuals**2) / (n - 2)
        se_slope = np.sqrt(mse / np.sum((X - X.mean())**2))
        t_stat = slope / se_slope
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n-2))
        
        print(f"{model} Model:")
        print(f"   Error = {intercept:.2f} + {slope:.4f} × Days_to_Expiry")
        print(f"   R²: {r2:.3f}  |  Slope p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            direction = "increases" if slope > 0 else "decreases"
            print(f"   ✓ Error {direction} significantly with maturity")
            print(f"   ⚠️ Model risk grows by {abs(slope):.2f}€ per day")
        else:
            print(f"   ✓ Error not significantly correlated with maturity")
        print()
    
    # === ANALYSE 3: Identification des seuils critiques ===
    print("="*70)
    print("CRITICAL THRESHOLDS IDENTIFICATION")
    print("="*70 + "\n")
    
    # Tester si erreur augmente significativement après 90j
    for model in ['BS', 'MC', 'CRR']:
        short_term = df[df['dte'] <= 90][f'abs_error_{model}'].dropna()
        long_term = df[df['dte'] > 90][f'abs_error_{model}'].dropna()
        
        if len(short_term) < 5 or len(long_term) < 5:
            continue
        
        # T-test
        t_stat, p_value = stats.ttest_ind(short_term, long_term)
        
        mae_short = short_term.mean()
        mae_long = long_term.mean()
        increase_pct = (mae_long - mae_short) / mae_short * 100
        
        print(f"{model} Model:")
        print(f"   MAE ≤90 days:  {mae_short:.2f}€")
        print(f"   MAE >90 days:  {mae_long:.2f}€")
        print(f"   Increase:      {increase_pct:+.1f}%  (p={p_value:.4f})")
        
        if p_value < 0.05:
            print(f"   🚨 ERROR SIGNIFICANTLY HIGHER after 90 days")
        print()
    
    # === ANALYSE 4: Lien avec liquidité ===
    print("="*70)
    print("LIQUIDITY IMPACT ANALYSIS")
    print("="*70 + "\n")
    
    # Hypothèse: Les outliers sont dans les options illiquides
    df['is_liquid'] = (df['volume'] > 100) & (df['openInterest'] > 200)
    
    for model in ['BS', 'MC', 'CRR']:
        liquid = df[df['is_liquid']][f'abs_error_{model}'].dropna()
        illiquid = df[~df['is_liquid']][f'abs_error_{model}'].dropna()
        
        if len(liquid) < 5 or len(illiquid) < 5:
            continue
        
        mae_liquid = liquid.mean()
        mae_illiquid = illiquid.mean()
        
        t_stat, p_value = stats.ttest_ind(liquid, illiquid)
        
        print(f"{model} Model:")
        print(f"   MAE (liquid):    {mae_liquid:.2f}€  (n={len(liquid)})")
        print(f"   MAE (illiquid):  {mae_illiquid:.2f}€  (n={len(illiquid)})")
        print(f"   Difference:      {mae_illiquid - mae_liquid:+.2f}€  (p={p_value:.4f})")
        
        if p_value < 0.05:
            print(f"   ✓ Illiquidity significantly increases pricing error")
        print()
    
    # === VISUALISATION: Graphique amélioré ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Boxplot original mais amélioré
    ax1 = axes[0, 0]
    df_melted = pd.melt(df, id_vars=['maturity_bucket'], 
                       value_vars=['error_BS', 'error_MC', 'error_CRR'],
                       var_name='model', value_name='error')
    
    sns.boxplot(data=df_melted, x='maturity_bucket', y='error', hue='model', ax=ax1)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_title('Error Distribution by Maturity Bucket', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pricing Error (€)', fontsize=12)
    ax1.set_xlabel('Maturity Bucket', fontsize=12)
    ax1.legend(title='Model')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. MAE par bucket (barplot)
    ax2 = axes[0, 1]
    mae_by_bucket = df.groupby('maturity_bucket')[['abs_error_BS', 'abs_error_MC', 'abs_error_CRR']].mean()
    mae_by_bucket.plot(kind='bar', ax=ax2, color=['steelblue', 'orange', 'green'], alpha=0.7)
    ax2.set_title('Mean Absolute Error by Maturity', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE (€)', fontsize=12)
    ax2.set_xlabel('Maturity Bucket', fontsize=12)
    ax2.legend(['BS', 'MC', 'CRR'])
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Scatterplot: Error vs DTE
    ax3 = axes[1, 0]
    ax3.scatter(df['dte'], df['abs_error_BS'], alpha=0.3, s=10, label='BS', color='steelblue')
    ax3.scatter(df['dte'], df['abs_error_MC'], alpha=0.3, s=10, label='MC', color='orange')
    ax3.scatter(df['dte'], df['abs_error_CRR'], alpha=0.3, s=10, label='CRR', color='green')
    
    # Régression linéaire superposée
    from sklearn.linear_model import LinearRegression
    X_plot = df['dte'].values.reshape(-1, 1)
    for model, color in [('BS', 'steelblue'), ('MC', 'orange'), ('CRR', 'green')]:
        y_plot = df[f'abs_error_{model}'].values
        mask = ~np.isnan(y_plot)
        if mask.sum() > 0:
            reg = LinearRegression().fit(X_plot[mask], y_plot[mask])
            ax3.plot(X_plot[mask], reg.predict(X_plot[mask]), color=color, 
                    linewidth=2, linestyle='--', alpha=0.7)
    
    ax3.set_title('Absolute Error vs Days to Expiry', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Days to Expiry', fontsize=12)
    ax3.set_ylabel('Absolute Error (€)', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Outliers par bucket
    ax4 = axes[1, 1]
    outlier_counts = []
    for bucket in df['maturity_bucket'].cat.categories:
        if bucket in stats_by_bucket and 'BS' in stats_by_bucket[bucket]:
            outlier_counts.append([
                stats_by_bucket[bucket]['BS']['pct_outliers'],
                stats_by_bucket[bucket]['MC']['pct_outliers'] if 'MC' in stats_by_bucket[bucket] else 0,
                stats_by_bucket[bucket]['CRR']['pct_outliers'] if 'CRR' in stats_by_bucket[bucket] else 0
            ])
        else:
            outlier_counts.append([0, 0, 0])
    
    outlier_df = pd.DataFrame(outlier_counts, 
                             columns=['BS', 'MC', 'CRR'],
                             index=df['maturity_bucket'].cat.categories)
    outlier_df.plot(kind='bar', ax=ax4, color=['steelblue', 'orange', 'green'], alpha=0.7)
    ax4.set_title('Outlier Percentage by Maturity', fontsize=14, fontweight='bold')
    ax4.set_ylabel('% of Outliers', fontsize=12)
    ax4.set_xlabel('Maturity Bucket', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend(['BS', 'MC', 'CRR'])
    
    plt.tight_layout()
    plt.savefig('png/maturity_deep_dive_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return stats_by_bucket

# Utilisation
visu.df_total.shape()

#maturity_stats = deep_dive_maturity_analysis(visu.df_total)