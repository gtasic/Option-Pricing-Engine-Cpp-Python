import streamlit as st
import pandas as pd
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn
import supabase 
import os
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
load_dotenv()
from scipy.interpolate import griddata
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, skew


supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)



st.title("Delta Hedging Strategy Analysis")
st.write("This page explores the implementation and performance of a delta hedging strategy for options trading."
         "Delta hedging involves adjusting the hedge position in the underlying asset to maintain a delta-neutral portfolio, thereby minimizing risk from price movements of the underlying asset.")

st.write("In this first portfolio, we just select two options a day among a universe of options with important volume and open interest. " \
"But we select it randomly, without any optimization to see the principle of delta hedging. " \
    "The delta of the options is calculated using the Black-Scholes model, and we hedge the total delta of the portfolio by buying/selling the underlying asset accordingly.")

portfolio_df = supabase_client.table("daily_portfolio_pnl").select("*").execute().data
portfolio_df = pd.DataFrame(portfolio_df)
portfolio_metrics = supabase_client.table("portfolio_metrics").select("*").execute().data
portfolio_metrics = pd.DataFrame(portfolio_metrics)


col1, col2, col3, col4 = st.columns(4)
current_nav = portfolio_df['nav'].iloc[-1]
daily_pnl = portfolio_df['daily_pnl'].iloc[-1]
sharpe = portfolio_metrics['sharpe_ratio'].iloc[-1]  
max_dd = portfolio_metrics['max_drawdown'].iloc[-1]  
total_delta = portfolio_df['total_delta'].iloc[-1]  

with col1:
    st.metric(
        label="💰 NAV",
        value=f"{current_nav:,.0f} €",
        delta=f"{daily_pnl:+.2f} €"
    )

with col2:
    st.metric(
        label="📈 Sharpe Ratio",
        value=f"{sharpe:.3f}",
        delta=f"{sharpe - 1.5:.2f}" if sharpe > 1.5 else None
    )

with col3:
    st.metric(
        label="📉 Max Drawdown",
        value=f"{max_dd:.1%}",
        delta=f"{max_dd - (-0.05):.1%}"
    )

with col4:
    st.metric(
        label="⚖️ Delta",
        value=f"{total_delta:.3f}",
        delta="Neutral ✅" if abs(total_delta) < 0.01 else "Rebalance ⚠️"
    )
st.write(f"These metrics are the key performance indicators of the delta hedging strategy. These are metrics for {portfolio_df['asof'].min()} to {portfolio_df['asof'].max()}.")

st.subheader("Portfolio over the time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_df['asof'], y=portfolio_df['nav'],
                         mode='markers',
                         name='Portfolio Value'))
fig.update_layout(title='Portfolio Value Over Time',
                   xaxis_title='Date',
                   yaxis_title='Portfolio Value (€)')
st.plotly_chart(fig, use_container_width=True)

st.write("The portfolio value graph shows how the value of the delta-hedged portfolio evolves over time. " \
"The nav evoloves step by step as new options are added to the portfolio and hedged. Globally, we can see " \
"an upward trend, indicating that the delta hedging strategy is effective in managing risk and generating" \
" returns over the period analyzed.")
st.write("However, our strategy is quite basic for now, as we just select two options a day randomly. " \
"More advanced selection and hedging techniques could further improve performance. So we have a positive pnl, because " \
"we don't take into account slippage and others frictions of the market. In a real trading environment, " \
"these factors could significantly impact the profitability of the delta hedging strategy.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Daily PnL Distribution")
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Histogram(x=portfolio_df['daily_pnl'], nbinsx=300))
    fig_pnl.update_layout(title='Daily PnL Distribution',
                          xaxis_title='Daily PnL (€)',
                          yaxis_title='Frequency',
                          xaxis_range=[-2000,2000])
    st.plotly_chart(fig_pnl, use_container_width=True)



with col2:
    st.subheader("Quantity hedged Over Time")
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=portfolio_df['asof'], y=portfolio_df['quantity_assets'],
                                   mode='markers',
                                   name='Hedeged Quantity'))
    fig_delta.update_layout(title='Quantity hedged Over Time',
                            xaxis_title='Date',
                            yaxis_title='Hedged Quantity')
    st.plotly_chart(fig_delta, use_container_width=True)    


st.write("The daily PnL distribution histogram illustrates the frequency of daily profit and loss values for the delta-hedged portfolio. " \
"We can observe that most daily PnL values cluster around zero, indicating that the delta hedging strategy effectively minimizes large fluctuations in daily returns. " \
"However, there are still occasional significant gains and losses, reflecting the inherent risks in options trading despite hedging efforts." )
st.write("The quantity of assets hedged over the time graph shows that the amount hedged decreases down to a floor level threshold that is the 40 days which is the maturity" \
"which is the most present in our portfolio." )

def get_open_positions():
    response = supabase_client.table("portfolio_options").select("*").eq("status", "open").execute()
    data = response.data
    df = pd.DataFrame(data)
    return df

st.title("📋 Current Positions")

col1, col2 = st.columns(2)
with col1:
    status_filter = st.selectbox("Status", ["All", "Open", "Closed"])
with col2:
    sort_by = st.selectbox("Sort by", ["Delta", "Gamma", "PnL", "Expiry"])

st.subheader("🔓 Open Options")
df_positions = get_open_positions()

df_display = df_positions[[
    'contract_symbol', 'strike', 'expiry', 'quantity', 
    'prix', 'delta', 'gamma', 'vega', 'theta'
]].copy()

def color_delta(val):
    color = 'green' if val > 0 else 'red'
    return f'color: {color}'

st.dataframe(
    df_display.style.applymap(color_delta, subset=['delta']),
    use_container_width=True
)
st.write("The open options table displays all currently active option positions in the delta-hedged portfolio. " \
"It includes key details such as the contract symbol, strike price, expiry date, quantity held, and the Greeks (delta, gamma, vega, theta) associated with each option. " \
"This information is crucial for understanding the risk profile of the portfolio and how each option contributes to the overall hedging strategy.")

st.subheader("🎯 Portfolio Greeks")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Δ Total", f"{df_positions['delta'].sum():.2f}")
col2.metric("Γ Total", f"{df_positions['gamma'].sum():.3f}")
col3.metric("ν Total", f"{df_positions['vega'].sum():.1f}")
col4.metric("Θ Total", f"{df_positions['theta'].sum():.1f}")
col5.metric("ρ Total", f"{df_positions['rho'].sum():.1f}")
st.write("The portfolio Greeks section summarizes the aggregate sensitivities of the delta-hedged portfolio to various risk factors. " \
"Delta (Δ) indicates the portfolio's sensitivity to changes in the underlying asset price, Gamma (Γ) measures the rate of change of delta, Vega (ν) reflects sensitivity to volatility changes, Theta (Θ) represents time decay, and Rho (ρ) indicates sensitivity to interest rate changes. " \
"These metrics are essential for assessing the effectiveness of the hedging strategy and understanding the portfolio's risk exposure.")

st.subheader("📊 Strike Distribution")
fig = go.Figure(data=[go.Bar(x=df_positions['strike'], y=df_positions['quantity'])])
st.plotly_chart(fig, use_container_width=True)
st.write("The strike distribution bar chart visualizes the quantity of options held at various strike prices within the delta-hedged portfolio. " \
"This distribution provides insights into the portfolio's exposure to different price levels of the underlying asset. "\
"A well-diversified strike distribution can help mitigate risks associated with price movements, while concentrations at specific strikes may indicate targeted strategies or potential vulnerabilities.")

st.subheader("PnL attribution by asof date")

def pnl_attribution(df_portfolio):
    dt = 1 / 252 
    d_price = df_portfolio["buy_price_asset"].diff().fillna(0)
    d_vol_avg = df_portfolio['total_sigma'].diff().fillna(0)*0.01
    df_portfolio["pnl_delta"] = df_portfolio["total_delta"] * d_price
    df_portfolio["pnl_gamma"] = 0.5 * df_portfolio["total_gamma"] * (d_price**2)
    df_portfolio["pnl_vega"] = df_portfolio["total_vega"] * d_vol_avg
    df_portfolio["pnl_theta"] = df_portfolio["total_theta"] * dt
    df_portfolio["explained_pnl"] = (df_portfolio["pnl_delta"] + 
                                     df_portfolio["pnl_gamma"] + 
                                     df_portfolio["pnl_vega"] + 
                                     df_portfolio["pnl_theta"])
    df_portfolio["pnl_residual"] = df_portfolio["daily_pnl"] - df_portfolio["explained_pnl"]
    return df_portfolio

st.dataframe(pnl_attribution(portfolio_df), use_container_width=True)

class PnLAttributionVisualizer:
    
    def __init__(self, theme='plotly_dark'):
        self.theme = theme
        self.colors = {
            'delta': '#1f77b4',    # Bleu
            'gamma': '#ff7f0e',    # Orange
            'vega': '#2ca02c',     # Vert
            'theta': '#d62728',    # Rouge
          #  'rho': '#9467bd',      # Violet
            'residual': '#8c564b', # Marron
            'total': '#e377c2'     # Rose
        }
    
    def waterfall_chart(self, attribution: dict, title: str = "Daily P&L Attribution") -> go.Figure:
        greeks = ['Delta', 'Gamma', 'Vega', 'Theta', 'Residual']
        values = [
            attribution['pnl_delta'],
            attribution['pnl_gamma'],
            attribution['pnl_vega'],
            attribution['pnl_theta'],
            attribution['pnl_residual']
        ]
        greeks.append('Total')
        values.append(attribution['pnl_total'])
        measure = ['relative'] * (len(greeks) - 1) + ['total']
        fig = go.Figure(go.Waterfall(
            name="P&L",
            orientation="v",
            measure=measure,
            x=greeks,
            y=values,
            text=[f"€{v:,.0f}" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ca02c"}},
            decreasing={"marker": {"color": "#d62728"}},
            totals={"marker": {"color": "#1f77b4"}}
        ))
        
        fig.update_layout(
            title={
                'text': f"{title}<br><sub>Date: {attribution.get('asof', 'N/A')}</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            yaxis_title="P&L (€)",
            showlegend=False,
            template=self.theme,
            height=500,
            font=dict(size=12)
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def stacked_area_chart(self, df: pd.DataFrame, title: str = "Cumulative P&L Attribution") -> go.Figure:
        fig = go.Figure()
        greeks_to_plot = [
            ('pnl_delta_cumul', 'Delta', self.colors['delta']),
            ('pnl_gamma_cumul', 'Gamma', self.colors['gamma']),
            ('pnl_vega_cumul', 'Vega', self.colors['vega']),
            ('pnl_theta_cumul', 'Theta', self.colors['theta']),
            ('pnl_residual_cumul', 'Residual', self.colors['residual'])
        ]
        
        for col, name, color in greeks_to_plot:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['asof'],
                    y=df[col],
                    mode='lines',
                    name=name,
                    stackgroup='one',
                    fillcolor=color,
                    line=dict(width=0.5, color=color),
                    hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Cumul P&L: €%{{y:,.2f}}<extra></extra>'
                ))
        fig.add_trace(go.Scatter(
            x=df['asof'],
            y=df['daily_pnl'],
            mode='markers',
            name='Total P&L',
            line=dict(color='white', width=3, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>Total P&L</b><br>Date: %{x}<br>€%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (€)",
            template=self.theme,
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def bar_chart_contribution(self, df: pd.DataFrame, title: str = "Total P&L Contribution by Greek") -> go.Figure:
        totals = {
            'Delta': df['pnl_delta'].sum(),
            'Gamma': df['pnl_gamma'].sum(),
            'Vega': df['pnl_vega'].sum(),
            'Theta': df['pnl_theta'].sum(),
            'Residual': df['pnl_residual'].sum()
        }
        
        greeks = list(totals.keys())
        values = list(totals.values())
        colors_list = [self.colors[g.lower()] for g in greeks]
        
        fig = go.Figure(go.Bar(
            x=greeks,
            y=values,
            marker_color=colors_list,
            text=[f"€{v:,.0f}" for v in values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Total P&L: €%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f"{title}<br><sub>Period: {df['asof'].iloc[0]} to {df['asof'].iloc[-1]}</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Greek",
            yaxis_title="Total P&L (€)",
            template=self.theme,
            height=500,
            showlegend=False
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    

    def heatmap_daily_attribution(self, df, title="Daily P&L Attribution Heatmap", n_days=15):
        df_subset = df.tail(n_days).copy()        
        greeks = ['pnl_delta', 'pnl_gamma', 'pnl_vega', 'pnl_theta', 'pnl_residual']
        greek_names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Residual']
        matrix = df_subset[greeks].T.values
        text_template = '%{z:.2s}€' 
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=df_subset['asof'],  # On utilise les dates filtrées
            y=greek_names,
            colorscale='RdBu',    # RdBu (Rouge-Bleu) est souvent plus lisible pour la Finance que RdYlGn
            reversescale=False,   # Rouge = Perte, Bleu = Gain (si RdBu standard)
            zmid=0,               # Le 0 est toujours blanc/neutre
            text=matrix,          # Les valeurs brutes pour le survol (hover)
            texttemplate=text_template, # Format court affiché dans la case
            textfont={"size": 12},
            colorbar=dict(title="P&L (€)")
        ))
        
        fig.update_layout(
            title={
                'text': f"{title} (Last {n_days} days)",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Date",
            yaxis_title="Greek Attribution",
            template=self.theme,
            height=500,

            xaxis=dict(
                type='category', 
                tickangle=-45                )
        )
        
        return fig
    
    def combined_dashboard(self, df: pd.DataFrame, latest_attribution: dict) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Latest Day Waterfall",
                "Cumulative P&L by Greek",
                "Total Contribution by Greek",
                "Daily Attribution Heatmap"
            ),
            specs=[
                [{"type": "waterfall"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        greeks = ['Delta', 'Gamma', 'Vega', 'Theta' 'Residual', 'Total']
        values = [
            latest_attribution['pnl_delta'],
            latest_attribution['pnl_gamma'],
            latest_attribution['pnl_vega'],
            latest_attribution['pnl_theta'],
            latest_attribution['pnl_residual'],
            latest_attribution['pnl_total']
        ]
        measure = ['relative'] * 6 + ['total']
        
        fig.add_trace(
            go.Waterfall(
                x=greeks, y=values, measure=measure,
                increasing={"marker": {"color": "#2ca02c"}},
                decreasing={"marker": {"color": "#d62728"}},
                totals={"marker": {"color": "#1f77b4"}},
                text=[f"€{v:.0f}" for v in values],
                textposition="outside"
            ),
            row=1, col=1
        )
        
        for greek, color in [('pnl_delta_cumul', self.colors['delta']),
                              ('pnl_vega_cumul', self.colors['vega']),
                              ('pnl_theta_cumul', self.colors['theta'])]:
            if greek in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['asof'], y=df[greek],
                        name=greek.replace('pnl_', '').replace('_cumul', '').title(),
                        line=dict(color=color, width=2)
                    ),
                    row=1, col=2
                )
        
        totals = {
            'Delta': df['pnl_delta'].sum(),
            'Vega': df['pnl_vega'].sum(),
            'Theta': df['pnl_theta'].sum(),
            'Gamma': df['pnl_gamma'].sum()
        }
        fig.add_trace(
            go.Bar(
                x=list(totals.keys()),
                y=list(totals.values()),
                marker_color=[self.colors[k.lower()] for k in totals.keys()]
            ),
            row=2, col=1
        )
        
        greeks_heat = ['pnl_delta', 'pnl_vega', 'pnl_theta']
        matrix = df[greeks_heat].T.values
        
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=df['asof'],
                y=['Delta', 'Vega', 'Theta'],
                colorscale='RdYlGn',
                zmid=0
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            template=self.theme,
            height=900,
            showlegend=True,
            title_text="P&L Attribution Dashboard",
            title_x=0.5
        )
        
        return fig



def create_all_visualizations():
    viz = PnLAttributionVisualizer(theme='plotly_dark')
    
    df_attribution = pnl_attribution(portfolio_df)
    latest = {
        'asof': df_attribution['asof'].iloc[-1],
        'pnl_delta': df_attribution['pnl_delta'].iloc[-1],
        'pnl_gamma': df_attribution['pnl_gamma'].iloc[-1],
        'pnl_vega': df_attribution['pnl_vega'].iloc[-1],
        'pnl_theta': df_attribution['pnl_theta'].iloc[-1],
      #  'pnl_rho': df_attribution['pnl_rho'].iloc[-1], --- IGNORE ---
        'pnl_residual': df_attribution['pnl_residual'].iloc[-1],
        'pnl_total': df_attribution['daily_pnl'].iloc[-1]
    }
    fig1 = viz.waterfall_chart(latest, "Daily P&L Attribution - Latest Day")
    print("✅ Waterfall chart sauvé")   
    fig2 = viz.stacked_area_chart(df_attribution, "Cumulative P&L Attribution Over Time")
    print("✅ Stacked area chart sauvé")
    fig3 = viz.bar_chart_contribution(df_attribution, "Total P&L Contribution by Greek")
    print("✅ Bar chart sauvé")
    fig4 = viz.heatmap_daily_attribution(df_attribution, "Daily P&L Attribution Heatmap")
    print("✅ Heatmap sauvé")    
    fig5 = viz.combined_dashboard(df_attribution, latest)
    return df_attribution, fig1, fig2, fig3, fig4, fig5

df_attribution, fig1, fig2, fig3, fig4, fig5 = create_all_visualizations()
st.subheader("Combined P&L Attribution Dashboard")
st.info("In this dashboard, we present a comprehensive overview of the P&L attribution for the delta-hedged portfolio. " \
"The dashboard includes a waterfall chart for the latest day, a stacked area chart showing cumulative P&L over time, a bar chart of total contributions by Greek, and a heatmap of daily attribution. " \
"These visualizations help in understanding how different Greeks contribute to the overall P&L and the effectiveness of the delta hedging strategy.")
st.plotly_chart(fig1)
st.info("This waterfall chart illustrates the breakdown of the portfolio's daily P&L into contributions from Delta, Gamma, Vega, Theta, and Residual components for the most recent trading day. " \
"It provides insights into which factors had the most significant impact on the portfolio's performance on that specific day.")
st.plotly_chart(fig2)
st.info("This stacked area chart shows the cumulative P&L over time, broken down by the contributions from each Greek. " \
"It helps visualize how the portfolio's performance has evolved over time and how different Greeks have contributed to that evolution.")
st.plotly_chart(fig3)
st.info("This bar chart displays the total P&L contribution by each Greek for the entire period. " \
"It highlights which Greeks have had the most significant impact on the portfolio's overall performance.")
st.plotly_chart(fig4)
st.info("This heatmap visualizes the daily P&L attribution for each Greek over the last 15 days. " \
"It allows for quick identification of patterns and trends in how each Greek has contributed to daily P&L.")

st.subheader("Conclusion")

st.write("The P&L attribution analysis provides valuable insights into the performance of the delta hedging strategy. " \
"By breaking down the daily P&L into contributions from Delta, Gamma, Vega, Theta, and Residual components, we can better understand the sources of profit and loss within the portfolio.   " \
"The waterfall chart for the latest day highlights the immediate impact of each Greek on the portfolio's performance, while the stacked area chart illustrates how these contributions accumulate over time. " \
"The bar chart summarizes the total contributions, allowing us to identify which Greeks have been most influential overall. " \
"Finally, the heatmap provides a visual representation of daily fluctuations in P&L attribution, revealing patterns that may inform future hedging strategies. " \
"Overall, this analysis underscores the importance of monitoring and managing the Greeks in a delta hedging strategy to optimize performance and mitigate risks.")
