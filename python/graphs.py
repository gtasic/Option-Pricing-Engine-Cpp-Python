import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Chargement
df = pd.read_csv('df_train_ready.csv')

# --- GRAPHIQUE 1 : HESTON REGIME (Interactif) ---
# On veut voir l'évolution jour après jour.
# On prend une moyenne par jour pour n'avoir qu'un point par jour (le régime du jour)
df_daily = df.groupby('asof')[['H_kappa', 'H_volvol']].mean().reset_index()

fig_heston = px.scatter(
    df_daily, 
    x='H_kappa', 
    y='H_volvol',
    text='asof', # Affiche la date à côté du point
    size_max=20,
    title="<b>Dynamique des Régimes de Volatilité (Heston)</b><br><i>Comment la psychologie du marché évolue</i>"
)

# On ajoute une ligne pour relier les jours (La trajectoire)
fig_heston.add_trace(
    go.Scatter(
        x=df_daily['H_kappa'], 
        y=df_daily['H_volvol'], 
        mode='lines+markers', 
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


# --- GRAPHIQUE 2 : SABR ALPHA (Histogramme coloré) ---
# On va colorier les opportunités (Edge > 2% ou < -2%)

# Création des catégories pour la couleur
conditions = [
    (df['SABR_Edge'] > 0.05),  # Opportunité Vente (Trop cher)
    (df['SABR_Edge'] < -0.05), # Opportunité Achat (Pas cher)
    (True)                     # Bruit
]
choices = ['Vente (Overpriced)', 'Achat (Underpriced)', 'Bruit (Noise)']
df['Opportunity'] = ['Vente (Overpriced)' if df['SABR_Edge'].iloc[i] > 0.05 else
                     'Achat (Underpriced)' if df['SABR_Edge'].iloc[i] < -0.05 else 'Bruit' for i in range(len(df))]

fig_sabr = px.histogram(
    df, 
    x="SABR_Edge", 
    color="Opportunity", 
    nbins=40,
    title="<b>Distribution des Opportunités d'Arbitrage (SABR)</b><br><i>Détection des anomalies de prix > 5%</i>",
    color_discrete_map={
        'Bruit (Noise)': 'lightgray', 
        'Vente (Overpriced)': 'red', 
        'Achat (Underpriced)': 'green'
    }
)

fig_sabr.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Fair Price")
fig_sabr.update_layout(
    template='plotly_white',
    xaxis_title="Edge (Market IV - Model IV)",
    yaxis_title="Nombre d'Options",
    bargap=0.1
)
fig_sabr.write_html("sabr_opportunity_histogram.html")