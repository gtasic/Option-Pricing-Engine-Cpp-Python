import streamlit as st
import pandas as pd 
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn
import plotly.express as px
import plotly.graph_objects as go
import numpy as np




st.title("Interface for Financial Backtesting and Option Pricing")
st.write("This application serves as an interface to present the functionalities of my financial project developed in plural languages including Python, C++ and SQL.")
st.write("This application allows users to perform backtesting of trading strategies and price options using various models based on the scripts I developped.")
st.write("The backtesting logic is implemented in my project, while the option pricing and analysis functionalities are handled in chat.py using OpenAI's GPT-4 model.")
st.header("models Module")
st.write("The models pages includes various option pricing models such as Blacl-Scholes, Binomial Tree, Monte-Carlo Simulation and Heston Model to compare performant of different models on option pricing" \
"by modifying parameters such as volatility, interest rates, time to maturity and underlying asset price.")

st.header("Backtesting Module")
st.write("The backtesting module utilizes historical market data to evaluate trading strategies. It includes functions for selecting options based on criteria such as days to expiration and delta values.")

st.write("The models pages includes various option pricing models such as Blacl-Scholes, Binomial Tree, Monte-Carlo Simulation and Heston Model to compare performant of different models on option pricing" \
"by modifying parameters such as volatility, interest rates, time to maturity and underlying asset price.")


# Chargement
df = pd.read_csv('/workspaces/finance-/csv/df_train_ready.csv')

df_daily = df.groupby('asof')[['H_kappa', 'H_volvol']].mean().reset_index()

fig_heston = px.scatter(
    df_daily, 
    x='H_kappa', 
    y='H_volvol',
    text='asof', # Affiche la date à côté du point
    size_max=20,
    title="<b>Dynamique des Régimes de Volatilité (Heston)</b><br><i>Comment la psychologie du marché évolue</i>"
)

fig_heston.add_trace(
    go.Scatter(
        x=df_daily['H_kappa'], 
        y=df_daily['H_volvol'], 
        mode='lines', 
        line=dict(color='royalblue', width=2, dash='dot'),
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
fig_heston.write_html("heston.html") # Sauvegarde: fig_heston.write_html("heston_interactive.html")

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
    bargap=0.1
)


st.header("Visualizations")
st.write("The following visualizations illustrate the dynamics of volatility regimes and arbitrage opportunities in the options market based on the Heston and SABR models respectively.")  
st.plotly_chart(fig_heston)
st.plotly_chart(fig_sabr)
st.write("The first graph shows the evolution of volatility regimes over time, highlighting how market psychology changes. The second graph displays the distribution of arbitrage opportunities detected using the SABR model, with options priced significantly above or below their fair value.")    




df = pd.read_csv('/workspaces/finance-/csv/df_train_ready.csv')

# --- PRÉPARATION DES DONNÉES POUR LA HEATMAP ---
# On doit "binner" (grouper) les données pour créer une grille
# 1. Binning du Moneyness (De Put OTM à Call OTM)
df['Moneyness_Bin'] = pd.cut(df['moneyness'], bins=np.linspace(0.8, 1.2, 8))
# On formate les labels pour que ce soit propre
df['Moneyness_Label'] = df['Moneyness_Bin'].apply(lambda x: f"{x.mid:.2f}")

# 2. Binning des Maturités (Tenor)
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


# --- PRÉPARATION ---
# On utilise Plotly Express qui gère très bien les scatter plots avec trendlines

fig_skew = px.scatter(
    df, 
    x="tenor", 
    y="S_rho", 
    color="S_nu", # On ajoute une dimension : la couleur = la convexité (Vol-of-Vol)
    size="iv",    # On ajoute une dimension : la taille = le niveau de peur (IV)
    trendline="lowess", # Ajoute une courbe de tendance lissée (régression locale)
    trendline_options=dict(frac=0.8),
    title="<b>Structure à Terme du Skew (SABR Rho)</b><br><i>Dissipation de la peur à travers le temps</i>",
    labels={
        "tenor": "Maturité (Années)",
        "S_rho": "Intensité du Skew (Corrélation Spot-Vol)",
        "S_nu": "Convexité (Nu)",
        "iv": "Vol Implicite"
    },
    color_continuous_scale="Viridis", # Échelle de couleur pro
    opacity=0.7
)

# Personnalisation pour un look "Hedge Fund"
fig_skew.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey'))) # Bordure des points

# Ajout de zones d'annotation (Lignes de niveau)
fig_skew.add_hline(y=-0.9, line_dash="dot", line_color="red", annotation_text="Zone de Crash Risk")
fig_skew.add_hline(y=-0.3, line_dash="dot", line_color="green", annotation_text="Zone de Normalité")

fig_skew.update_layout(
    template='plotly_white',
    height=600,
    width=900,
    hovermode="closest" # Le tooltip montre le point le plus proche
)

st.plotly_chart(fig_skew)


import plotly.graph_objects as go
from scipy.interpolate import griddata

# Chargement
df = pd.read_csv('/workspaces/finance-/csv/df_train_ready.csv')

# --- PRÉPARATION DES DONNÉES 3D ---
# Pour tracer une surface 3D, il faut une grille (Meshgrid).
# Comme tes données sont des points épars, on va les interpoler pour créer un drapé lisse.

# 1. Création de la grille régulière
x_range = np.linspace(df['moneyness'].min(), df['moneyness'].max(), 30)
y_range = np.linspace(df['tenor'].min(), df['tenor'].max(), 30)
X, Y = np.meshgrid(x_range, y_range)

# 2. Interpolation pour la Surface Théorique (SABR)
# On imagine que 'S_theoretical_iv' est la surface lisse
Z_model = griddata(
    (df['moneyness'], df['tenor']), 
    df['S_theoretical_iv'], 
    (X, Y), 
    method='cubic' # ou 'linear' si cubic fait des vagues bizarres
)

# 3. Interpolation pour le Edge (Alpha)
Z_edge = griddata(
    (df['moneyness'], df['tenor']), 
    df['SABR_Edge'], 
    (X, Y), 
    method='cubic'
)

# --- GRAPHIQUE 1 : MARKET (Points) vs MODEL (Surface) ---

fig_calibration = go.Figure()

# A. La Surface du Modèle (Le Drapé Lisse)
fig_calibration.add_trace(go.Surface(
    z=Z_model, x=X, y=Y,
    colorscale='Viridis',
    opacity=0.8,
    name='SABR Model',
    showscale=False
))

# B. Les Points du Marché (La Réalité)
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

# --- GRAPHIQUE 2 : LA TOPOGRAPHIE DU PROFIT (3D Edge) ---

fig_alpha = go.Figure(data=[go.Surface(
    z=Z_edge, x=X, y=Y,
    colorscale='RdBu_r', # Rouge = Vente, Bleu = Achat
    cmin=-0.05, cmax=0.05, # On sature les couleurs à +/- 5% pour bien voir les zones
    opacity=0.9
)])

# Ajout d'un plan "Zéro" transparent pour voir le niveau neutre
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
