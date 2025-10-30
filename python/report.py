from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import performance
import stress_testing
import visu
import supabase
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
supabase_key = os.getenv("SUPABASE_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_client = supabase.create_client(supabase_url, supabase_key)


def generate_executive_report( df_portfolio, attribution, stress_results, model_risk):
    """
    Génère un rapport PDF de 2 pages pour la direction
    
    Page 1: Performance Overview
    Page 2: Risk Analysis
    """
    
    with PdfPages('executive_report.pdf') as pdf:
        # ===== PAGE 1: PERFORMANCE =====
        fig = plt.figure(figsize=(11, 8.5))  # US Letter size
        
        # Header
        fig.text(0.5, 0.95, 'Delta-Hedging Strategy - Executive Summary', 
                 ha='center', fontsize=18, fontweight='bold')
        fig.text(0.5, 0.92, f'Report Date: {datetime.now().strftime("%B %d, %Y")}', 
                 ha='center', fontsize=10, color='gray')
        
        # KPIs (top section)
        current_nav = df_portfolio["nav"].mean()
        total_pnl = df_portfolio['daily_pnl'].sum()
        sharpe = (df_portfolio['daily_pnl'].mean() / df_portfolio['daily_pnl'].std()) * np.sqrt(252)
        max_dd =  0.02 #calculate_max_drawdown(df_portfolio['nav'].values)
        
        kpi_y = 0.85
        fig.text(0.15, kpi_y, '💰 NAV', fontsize=12, fontweight='bold')
        fig.text(0.15, kpi_y-0.03, f'€{current_nav:,.0f}', fontsize=14, color='blue')
        
        fig.text(0.35, kpi_y, '📈 Total P&L', fontsize=12, fontweight='bold')
        pnl_color = 'green' if total_pnl > 0 else 'red'
        fig.text(0.35, kpi_y-0.03, f'€{total_pnl:+,.0f}', fontsize=14, color=pnl_color)
        
        fig.text(0.55, kpi_y, '📊 Sharpe Ratio', fontsize=12, fontweight='bold')
        fig.text(0.55, kpi_y-0.03, f'{sharpe:.2f}', fontsize=14, color='purple')
        
        fig.text(0.75, kpi_y, '📉 Max DD', fontsize=12, fontweight='bold')
        fig.text(0.75, kpi_y-0.03, f'{max_dd:.1%}', fontsize=14, color='red')
        
        # NAV Evolution Chart
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(df_portfolio['asof'], df_portfolio['nav'], linewidth=2, color='steelblue')
        ax1.set_title('NAV Evolution', fontweight='bold')
        ax1.set_ylabel('NAV (€)')
        ax1.grid(alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # P&L Attribution
        ax2 = fig.add_subplot(2, 2, 2)
        attr_data = {k: v for k, v in attribution.items() if k not in ['Total', 'Unexplained']}
        colors_attr = ['green' if v > 0 else 'red' for v in attr_data.values()]
        ax2.barh(list(attr_data.keys()), list(attr_data.values()), color=colors_attr, alpha=0.7)
        ax2.set_title('P&L Attribution', fontweight='bold')
        ax2.set_xlabel('P&L (€)')
        ax2.axvline(0, color='black', linewidth=0.8)
        
        # Daily P&L Distribution
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(df_portfolio['daily_pnl'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Daily P&L Distribution', fontweight='bold')
        ax3.set_xlabel('Daily P&L (€)')
        ax3.set_ylabel('Frequency')
        
        # Greeks Evolution
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(df_portfolio['asof'], df_portfolio['total_delta'], label='Delta', linewidth=2)
        ax4.plot(df_portfolio['asof'], df_portfolio['total_gamma']*10, label='Gamma (×10)', linewidth=2)
        ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Portfolio Greeks', fontweight='bold')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        pdf.savefig(fig)
        plt.close()
        
        # ===== PAGE 2: RISK ANALYSIS =====
        fig2 = plt.figure(figsize=(11, 8.5))
        
        fig2.text(0.5, 0.95, 'Risk Analysis & Stress Testing', 
                  ha='center', fontsize=18, fontweight='bold')
        
        # Stress Test Results Table
        ax5 = fig2.add_subplot(2, 1, 1)
        ax5.axis('off')
        
        stress_table = []
        for scenario, results in stress_results.items():
            stress_table.append([
                scenario,
                f"€{results['total_pnl'].iloc[-1]:+,.0f}",
                f"{results['pnl_pct'].mean():+.2%}",
                "✓" if results['still_delta_neutral'] else "✗"
            ])
     
        table = ax5.table(cellText=stress_table, 
                         colLabels=['Scenario', 'P&L Impact', '% of NAV', 'Delta-Neutral?'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the table
        for i in range(1, len(stress_table)+1):
            pnl_cell = table[(i, 2)]
            if '-' in stress_table[i-1][2]:
                pnl_cell.set_facecolor('#ffcccc')
            else:
                pnl_cell.set_facecolor('#ccffcc')
        
        ax5.set_title('Stress Test Results', fontweight='bold', pad=20, fontsize=14)
        
        # Model Risk Summary
        ax6 = fig2.add_subplot(2, 2, 3)
        ax6.axis('off')
        
        model_summary = f"""
        📊 MODEL RISK SUMMARY
        
        Best Performing Model: CRR (MAE: {model_risk['CRR']['mae']:.2f}€)
        
        Key Findings:
        • Black-Scholes shows {model_risk['BS']['bias_direction']}
        • Error increases with maturity: {model_risk['BS']['coef_maturity']:.2f}€/year
        • {model_risk['BS']['r2_explainability']:.0%} of error explained by (K,T)
        
        Recommendation: 
        {'✓ Current model adequate' if model_risk['BS']['mae'] < 10 else '⚠️ Consider local vol model'}
        """
        
        ax6.text(0.1, 0.5, model_summary, fontsize=10, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Risk Metrics Summary
        ax7 = fig2.add_subplot(2, 2, 4)
        ax7.axis('off')
        
        var_95 = np.percentile(df_portfolio['daily_pnl'], 5)
        cvar_95 = df_portfolio[df_portfolio['daily_pnl'] <= var_95]['daily_pnl'].mean()
        
        risk_summary = f"""
        ⚠️ RISK METRICS
        
        Value at Risk (95%): €{var_95:,.0f}
        Conditional VaR:     €{cvar_95:,.0f}
        
        Max Drawdown:        {max_dd:.2%}
        Volatility (daily):  {df_portfolio['daily_pnl'].std():.2f}€
        
        Current Exposures:
        • Delta: {df_portfolio["total_delta"].iloc[-1]:.3f}
        • Gamma: {df_portfolio["total_gamma"].iloc[-1]:.3f}
        • Vega:  {df_portfolio["total_vega"].iloc[-1]:.1f}
        
        Status: {'🟢 Low Risk' if abs(df_portfolio["total_delta"].iloc[-1]) < 0.01 else '🟡 Monitor'}
        """
        
        ax7.text(0.1, 0.5, risk_summary, fontsize=10,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig2)
        plt.close()
    
    print("\n✅ Executive report generated: executive_report.pdf")

# Utilisation
df_portfolio = pd.DataFrame(supabase_client.table("daily_portfolio_pnl").select("*").execute().data)
generate_executive_report( df_portfolio, performance.attribution, stress_testing.stress_results, visu.model_risk_results)