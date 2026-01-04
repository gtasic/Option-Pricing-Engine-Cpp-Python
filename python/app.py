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
st.header("Models Module")
st.write("The models pages includes various option pricing models such as Black-Scholes, Binomial Tree, Monte-Carlo Simulation and Heston Model to compare performant of different models on option pricing" \
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




