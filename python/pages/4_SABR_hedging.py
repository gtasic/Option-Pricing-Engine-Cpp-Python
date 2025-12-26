import streamlit as st
import pandas as pd
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn
import supabase 
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
load_dotenv()
from scipy.interpolate import griddata
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, skew
import plotly.graph_objects as go
from scipy.interpolate import griddata


supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_client = supabase.create_client(supabase_url, supabase_key)



st.title("Delta Hedging Strategy Analysis")
st.write("This page explores the implementation and performance of a delta hedging strategy for options trading."
         "Delta hedging involves adjusting the hedge position in the underlying asset to maintain a delta-neutral portfolio, thereby minimizing risk from price movements of the underlying asset.")
st.subheader("Strategy Overview")
st.markdown("""In this portfolio, we implement a ddelta hedging strategy however we use a strategy that is a bit more complex than a simple delta hedge. 
            Indeed, for selcting options we use the SABR and  Heston models in order to represent more accurately the market dynamics and implied volatility surface.
            Then we use a ML model to predict the future volatility and adjust our hedging strategy accordingly.
            The main steps of the strategy are as follows:
            1. **Modeling the Implied Volatility Surface**: We use the SABR model to fit the implied volatility surface of options. This allows us to capture the volatility smile and skew observed in the market.
            2. **Predicting Future Volatility**: A machine learning model is trained on historical volatility data to predict future volatility levels. This prediction helps in selecting options that are expected to perform well under anticipated market conditions.
            3. **Option Selection**: Based on the predicted volatility and the fitted SABR model, we select options that are likely to provide optimal hedging characteristics.
            4. **Delta Hedging**: The portfolio is dynamically adjusted to maintain a delta-neutral position. This involves buying or selling the underlying asset as the delta of the options changes over time.
            5. **Performance Monitoring**: The performance of the hedging strategy is continuously monitored, with key metrics such as NAV, Sharpe ratio, and drawdown being tracked.
            This approach aims to enhance the effectiveness of delta hedging by incorporating advanced modeling techniques and predictive analytics.
            """)

portfolio_df = supabase_client.table("daily_complex_portfolio_pnl").select("*").execute().data
portfolio_df = pd.DataFrame(portfolio_df)
portfolio_metrics = supabase_client.table("portfolio_metrics").select("*").execute().data
portfolio_metrics = pd.DataFrame(portfolio_metrics)


col1, col2, col3, col4 = st.columns(4)
current_nav = portfolio_df['nav'].iloc[-1]
daily_pnl = portfolio_df['daily_pnl'].iloc[-1]
sharpe = portfolio_metrics['sharpe_ratio'].iloc[-1]  # Example Sharpe ratio
max_dd = portfolio_metrics['max_drawdown'].iloc[-1]  # Example max drawdown
total_delta = portfolio_df['total_delta'].iloc[-1]  # Example total delta
# We will have to implement the real values from our supabase database

with col1:
    st.metric(
        label="💰 NAV",
        value=f"{current_nav:,.0f} €",
        delta=f"{daily_pnl:+.2f} €"
    )

with col2:
    st.metric(
        label="📈 Sharpe Ratio",
        value=f"{sharpe:.2f}",
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


st.subheader("Portfolio over the time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_df['asof'], y=portfolio_df['nav'],
                         mode='markers',
                         name='Portfolio Value'))
fig.update_layout(title='Portfolio Value Over Time',
                   xaxis_title='Date',
                   yaxis_title='Portfolio Value (€)')
st.plotly_chart(fig, use_container_width=True)


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
    fig_delta.add_trace(go.Scatter(x=portfolio_df['asof'], y=portfolio_df['quantity_asset'],
                                   mode='markers',
                                   name='Hedeged Quantity'))
    fig_delta.update_layout(title='Quantity hedged Over Time',
                            xaxis_title='Date',
                            yaxis_title='Hedged Quantity')
    st.plotly_chart(fig_delta, use_container_width=True)    


def get_open_positions():
    response = supabase_client.table("complex_portfolio_options").select("*").eq("statut", "open").execute()
    data = response.data
    df = pd.DataFrame(data)
    return df

st.title("📋 Current Positions")

# Filtres
col1, col2 = st.columns(2)
with col1:
    status_filter = st.selectbox("Status", ["All", "Open", "Closed"])
with col2:
    sort_by = st.selectbox("Sort by", ["Delta", "Gamma", "PnL", "Expiry"])

# Table des positions
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

st.subheader("🎯 Portfolio Greeks")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Δ Total", f"{df_positions['delta'].sum():.2f}")
col2.metric("Γ Total", f"{df_positions['gamma'].sum():.3f}")
col3.metric("ν Total", f"{df_positions['vega'].sum():.1f}")
col4.metric("Θ Total", f"{df_positions['theta'].sum():.1f}")
col5.metric("ρ Total", f"{df_positions['rho'].sum():.1f}")

st.subheader("📊 Strike Distribution")
fig = go.Figure(data=[go.Bar(x=df_positions['strike'], y=df_positions['quantity'])])
st.plotly_chart(fig, use_container_width=True)

st.write("The strike distribution bar chart visualizes the quantity of options held at various strike prices within the delta-hedged portfolio. " \
"This distribution provides insights into the portfolio's exposure to different price levels of the underlying asset. "\
"A well-diversified strike distribution can help mitigate risks associated with price movements, while concentrations at specific strikes may indicate targeted strategies or potential vulnerabilities.")


df = pd.read_csv('/workspaces/finance-/csv/df_train_ready.csv')

df_daily = df.groupby('asof')[['H_kappa', 'H_volvol']].mean().reset_index()

fig_heston = px.scatter(
    df_daily[-20:], 
    x='H_kappa', 
    y='H_volvol',
    text = 'asof',
    size_max=20,
    title="<b>Dynamique des Régimes de Volatilité (Heston)</b><br><i>Comment la psychologie du marché évolue</i>"
)

fig_heston.add_trace(
    go.Scatter(
        x=df_daily['H_kappa'][-20:], 
        y=df_daily['H_volvol'][-20:], 
        mode='markers', 
        line=dict(color='royalblue', width=1.3, dash='dot'),
        name='Trajectoire'
    )
)

fig_heston.update_traces(textposition='top center', marker=dict(size=12, color='firebrick'))
fig_heston.update_layout(
    template='plotly_white',
    xaxis_title="Vitesse de retour au calme (Kappa)",
    yaxis_title="Nervosité du Marché (Vol of Vol)",
    font=dict(family="Arial", size=12)
)

conditions = [
    (df['SABR_Edge'] > 0.05),  # Opportunité Vente (Trop cher)
    (df['SABR_Edge'] < -0.05), # Opportunité Achat (Pas cher)
    (True)                     # Bruit
]
choices = ['Vente (Overpriced)', 'Achat (Underpriced)', 'Bruit (Noise)']
df['Opportunity'] = ['Vente (Overpriced)' if df['SABR_Edge'].iloc[i] > 0.05 else
                     'Achat (Underpriced)' if df['SABR_Edge'].iloc[i] < -0.05 else 'Bruit (Noise)' for i in range(len(df))]

fig_sabr = px.histogram(
    df, 
    x="SABR_Edge", 
    color="Opportunity", 
    nbins=120,
    title="<b>Distribution des Opportunités d'Arbitrage (SABR)</b><br><i>Détection des anomalies de prix > 5%</i>",
    color_discrete_map={
        'Bruit (Noise)': 'lightgray', 
        'Vente (Overpriced)': 'red', 
        'Achat (Underpriced)': 'green'
    }
)

fig_sabr.add_vline(x=0, line_dash="dash", line_color="yellow", annotation_text="Fair Price")
fig_sabr.update_layout(
    template='plotly_white',
    xaxis_title="Edge (Market IV - Model IV)",
    yaxis_title="Nombre d'Options",
    xaxis_range=[-0.3, 0.3],
    bargap=0.1
)


st.header("Visualizations")
st.write("The following visualizations illustrate the dynamics of volatility regimes and arbitrage opportunities in the options market based on the Heston and SABR models respectively.")  
st.plotly_chart(fig_heston)
st.plotly_chart(fig_sabr)
st.write("The first graph shows the evolution of volatility regimes over time, highlighting how market psychology changes. The second graph displays the distribution of arbitrage opportunities detected using the SABR model, with options priced significantly above or below their fair value.")    






df['Moneyness_Bin'] = pd.cut(df['moneyness'], bins=np.linspace(0.8, 1.2, 8))
df['Moneyness_Label'] = df['Moneyness_Bin'].apply(lambda x: f"{x.mid:.2f}")
df['Tenor_Bin'] = pd.cut(df['tenor'], bins=np.linspace(0.01, 0.10, 8)) 
df['Tenor_Label'] = df['Tenor_Bin'].apply(lambda x: f"{x.mid:.2f}Y")

heatmap_matrix = df.pivot_table(
    index='Tenor_Label', 
    columns='Moneyness_Label', 
    values='SABR_Edge', 
    aggfunc='mean'
)

# --- CRÉATION DU GRAPHIQUE PLOTLY ---
fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_matrix.values,
    x=heatmap_matrix.columns,
    y=heatmap_matrix.index,
    colorscale='RdBu_r', # Rouge = Cher (Vente), Bleu = Pas cher (Achat)
    zmid=0,              # Centre la couleur sur 0 (Prix Juste)
    text=np.round(heatmap_matrix.values, 4), # Affiche la valeur au survol
    texttemplate="%{text}",                  # Affiche la valeur dans la case
    textfont={"size": 10},
    hoverongaps=False
))

fig_heatmap.update_layout(
    title="<b>Heatmap des Opportunités d'Arbitrage</b><br><i>Où se cache l'Alpha ? (Rouge = Vendre, Bleu = Acheter)</i>",
    xaxis_title="Moneyness (Strike / Spot)",
    yaxis_title="Maturité (Années)",
    template='plotly_white',
    height=600,
    width=900
)

st.plotly_chart(fig_heatmap)

fig_skew = px.scatter(
    df, 
    x="tenor", 
    y="S_rho", 
    color="S_nu", 
    size="iv",    
    trendline="lowess",
    trendline_options=dict(frac=0.8),
    title="<b>Structure à Terme du Skew (SABR Rho)</b><br><i>Dissipation de la peur à travers le temps</i>",
    labels={
        "tenor": "Maturité (Années)",
        "S_rho": "Intensité du Skew (Corrélation Spot-Vol)",
        "S_nu": "Convexité (Nu)",
        "iv": "Vol Implicite"
    },
    color_continuous_scale="Viridis", 
    opacity=0.7
)

fig_skew.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey'))) # Bordure des points

fig_skew.add_hline(y=-0.9, line_dash="dot", line_color="red", annotation_text="Zone de Crash Risk")
fig_skew.add_hline(y=-0.3, line_dash="dot", line_color="green", annotation_text="Zone de Normalité")

fig_skew.update_layout(
    template='plotly_white',
    height=600,
    width=900,
    hovermode="closest" )

st.plotly_chart(fig_skew)


df = pd.read_csv('/workspaces/finance-/csv/df_train_ready.csv')


x_range = np.linspace(df['moneyness'].min(), df['moneyness'].max(), 30)
y_range = np.linspace(df['tenor'].min(), df['tenor'].max(), 30)
X, Y = np.meshgrid(x_range, y_range)


Z_model = griddata(
    (df['moneyness'], df['tenor']), 
    df['S_theoretical_iv'], 
    (X, Y), 
    method='cubic' 
)

Z_edge = griddata(
    (df['moneyness'], df['tenor']), 
    df['SABR_Edge'], 
    (X, Y), 
    method='cubic'
)


fig_calibration = go.Figure()

fig_calibration.add_trace(go.Surface(
    z=Z_model, x=X, y=Y,
    colorscale='Viridis',
    opacity=0.8,
    name='SABR Model',
    showscale=False
))

fig_calibration.add_trace(go.Scatter3d(
    x=df['moneyness'],
    y=df['tenor'],
    z=df['iv'],
    mode='markers',
    marker=dict(
        size=4,
        color='red',
        symbol='circle',
        opacity=0.9
    ),
    name='Market Data'
))

fig_calibration.update_layout(
    title="<b>Calibration Quality: Market Data vs SABR Surface</b><br><i>Les points rouges sont le marché, le drapé est le modèle</i>",
    scene=dict(
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Maturity (Years)',
        zaxis_title='Implied Volatility',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) # Angle de vue
    ),
    template='plotly_white',
    height=700
)

fig_alpha = go.Figure(data=[go.Surface(
    z=Z_edge, x=X, y=Y,
    colorscale='RdBu_r', # Rouge = Vente, Bleu = Achat
    cmin=-0.05, cmax=0.05, # On sature les couleurs à +/- 5% pour bien voir les zones
    opacity=0.9
)])

fig_alpha.add_trace(go.Surface(
    z=np.zeros_like(Z_edge), x=X, y=Y,
    colorscale=[[0, 'white'], [1, 'white']],
    opacity=0.3,
    showscale=False,
    name='Fair Price'
))

fig_alpha.update_layout(
    title="<b>Alpha Topography: 3D Map of Volatility Edge</b><br><i>Peaks = Overpriced (Sell), Valleys = Underpriced (Buy)</i>",
    scene=dict(
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Maturity (Years)',
        zaxis_title='Edge (Market - Model)',
        zaxis=dict(range=[-0.1, 0.1]), # On zoome sur l'écart
    ),
    template='plotly_white',
    height=700
)


st.plotly_chart(fig_calibration)
st.plotly_chart(fig_alpha)











st.subheader("PnL attribution by asof date")


#Pnl attribution
def pnl_attribution(df_portfolio):
    dt = 1 / 252 
    d_price = df_portfolio["buy_price_asset"].diff().fillna(0)
    

    
    d_vol_avg = df_portfolio["total_sigma"].diff().fillna(0)
    
    # -------------------------------------

    # Delta
    df_portfolio["pnl_delta"] = (df_portfolio["total_delta"]+df_portfolio["quantity_asset"]) * d_price
    
    # Gamma
    df_portfolio["pnl_gamma"] = 0.5 * df_portfolio["total_gamma"] * (d_price**2)
    
    # Vega : On utilise la variation de la VOL MOYENNE
    df_portfolio["pnl_vega"] = df_portfolio["total_vega"] * d_vol_avg
    
    # Theta (Rappel correction précédente)
    df_portfolio["pnl_theta"] = df_portfolio["total_theta"] * dt

    # Résultat
    df_portfolio["pnl_brut"] = df_portfolio["daily_pnl"] + df_portfolio["frais"]
    df_portfolio["explained_pnl"] = (df_portfolio["pnl_delta"] + 
                                     df_portfolio["pnl_gamma"] + 
                                     df_portfolio["pnl_vega"] + 
                                     df_portfolio["pnl_theta"])
    
    df_portfolio["pnl_residual"] = df_portfolio["pnl_brut"] - df_portfolio["explained_pnl"]
    
    return df_portfolio
st.dataframe(pnl_attribution(portfolio_df), use_container_width=True)

"""
Visualisations P&L Attribution
Waterfall, Stacked Area, Bar Charts pour l'analyse des Greeks
"""


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
        
        # Ajouter le total
        greeks.append('Total')
        values.append(attribution['pnl_total'])
        
        # Déterminer le type de chaque barre (relative ou total)
        measure = ['relative'] * (len(greeks) - 1) + ['total']
        
        # Créer le waterfall
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
        
        # Ajouter une ligne à zéro
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def stacked_area_chart(self, df: pd.DataFrame, title: str = "Cumulative P&L Attribution") -> go.Figure:
        """
        Graphique en aires empilées montrant l'attribution cumulative
        
        Parfait pour voir l'évolution sur plusieurs jours
        """
        
        fig = go.Figure()
        
        # Ajouter chaque Greek comme une aire
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
        
        # Ajouter la ligne du total
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
        """
        Bar chart montrant la contribution totale de chaque Greek
        
        Parfait pour un résumé sur toute la période
        """
        
        # Calculer les totaux
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
        # 1. FILTRAGE : On ne garde que les n derniers jours
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
        
        # 1. Waterfall du dernier jour
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
        
        # 2. Cumulative par Greek
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
        
        # 3. Bar chart des totaux
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
        
        # 4. Heatmap (simplifié)
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
    
    # 1. Waterfall du dernier jour
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
    
    # 2. Stacked area cumulatif
    fig2 = viz.stacked_area_chart(df_attribution, "Cumulative P&L Attribution Over Time")
    print("✅ Stacked area chart sauvé")
    
    # 3. Bar chart des contributions
    fig3 = viz.bar_chart_contribution(df_attribution, "Total P&L Contribution by Greek")
    print("✅ Bar chart sauvé")
    
    # 4. Heatmap
    fig4 = viz.heatmap_daily_attribution(df_attribution, "Daily P&L Attribution Heatmap")
    print("✅ Heatmap sauvé")
    
    # 5. Dashboard combiné
    fig5 = viz.combined_dashboard(df_attribution, latest)
  
    
    return df_attribution, fig1, fig2, fig3, fig4, fig5

df_attribution, fig1, fig2, fig3, fig4, fig5 = create_all_visualizations()
st.subheader("Combined P&L Attribution Dashboard")
st.info("ttttt")
st.plotly_chart(fig1)
st.info("ttttt")
st.plotly_chart(fig2)
st.info("ttttt")
st.plotly_chart(fig3)
st.info("ttttt")
st.plotly_chart(fig4)
