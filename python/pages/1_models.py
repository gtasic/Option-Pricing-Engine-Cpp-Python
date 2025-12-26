import streamlit as st
import pandas as pd
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import griddata
import numpy as np


st.title("Interface for Financial Backtesting and Option Pricing")
st.write("We are going to explore pricing options using various models in this section. We have different models" \
"implemented such as Black-Scholes, Binomial Tree, and Monte Carlo simulations and Heston. You can select the model you want to use" \
    " and input the necessary parameters to get the option prices.")

st.header("Option Pricing Models")
st.write("Select an option pricing model from the dropdown below to see its details and parameters.")
model_options = ["Black-Scholes", "Binomial Tree", "Monte Carlo Simulation", "Heston Model"]
selected_model = st.selectbox("Choose a model:", model_options)
if selected_model == "Black-Scholes":
    st.subheader("Black-Scholes Model")
    st.write("The Black-Scholes model is a mathematical model for pricing an options contract. "
             "It assumes that the price of the underlying asset follows a geometric Brownian motion with constant volatility and interest rate.")
    st.write("Parameters needed:")
    st.write("- Current stock price (S)")
    st.write("- Strike price (K)")
    st.write("- Time to expiration (T)")
    st.write("- Risk-free interest rate (r)")
    st.write("- Volatility of the underlying asset (σ)")
elif selected_model == "Binomial Tree":
    st.subheader("Binomial Tree Model")
    st.write("The Binomial Tree model is a discrete-time model for option pricing. "
             "It models the possible price movements of the underlying asset over time using a binomial lattice.")
    st.write("Parameters needed:")
    st.write("- Current stock price (S)")
    st.write("- Strike price (K)")
    st.write("- Time to expiration (T)")
    st.write("- Risk-free interest rate (r)")
    st.write("- Volatility of the underlying asset (σ)")
    st.write("- Number of time steps (N)")
elif selected_model == "Monte Carlo Simulation":    
    st.subheader("Monte Carlo Simulation")
    st.write("Monte Carlo Simulation is a statistical technique used to model the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables.")
    st.write("Parameters needed:")
    st.write("- Current stock price (S)")
    st.write("- Strike price (K)")
    st.write("- Time to expiration (T)")
    st.write("- Risk-free interest rate (r)")
    st.write("- Volatility of the underlying asset (σ)")
    st.write("- Number of simulations (M)")
elif selected_model == "Heston Model":
    st.subheader("Heston Model")
    st.write("The Heston model is a mathematical model that describes the evolution of the volatility of an underlying asset. "
             "It assumes that the volatility of the asset is stochastic and follows its own random process.")
    st.write("Parameters needed:")
    st.write("- Current stock price (S)")
    st.write("- Strike price (K)")
    st.write("- Time to expiration (T)")
    st.write("- Risk-free interest rate (r)")
    st.write("- Initial volatility (v0)")
    st.write("- Long-term variance (θ)")
    st.write("- Rate of mean reversion (κ)")
    st.write("- Volatility of volatility (σv)")
    st.write("- Correlation between asset and volatility (ρ)")
st.write("Use the sidebar to input the parameters for the selected model and compute the option price.")
st.sidebar.header("Input Parameters")
st.sidebar.subheader(f"Parameters for {selected_model}")
if selected_model == "Black-Scholes":
    S = st.sidebar.number_input("Current stock price (S):", value=100.0)
    K = st.sidebar.number_input("Strike price (K):", value=100.0)
    T = st.sidebar.number_input("Time to expiration (T in years):", value=1.0)
    r = st.sidebar.number_input("Risk-free interest rate (r):", value=0.05)
    sigma = st.sidebar.number_input("Volatility of the underlying asset (σ):", value=0.2)
    if st.sidebar.button("Calculate Option Price"):
        st.write("Calculating option price using Black-Scholes model...")
        # Here you would call your pricing function and display the result
        valeur = fn.call_price(fn.BS_parametres(S,K,T,r,sigma))
        st.write(f"Option Price: {valeur:.2f} €")
elif selected_model == "Binomial Tree": 
    S = st.sidebar.number_input("Current stock price (S):", value=100.0)
    K = st.sidebar.number_input("Strike price (K):", value=100.0)
    T = st.sidebar.number_input("Time to expiration (T in years):", value=1.0)
    r = st.sidebar.number_input("Risk-free interest rate (r):", value=0.05)
    sigma = st.sidebar.number_input("Volatility of the underlying asset (σ):", value=0.2)
    N = st.sidebar.number_input("Number of time steps (N):", value=100)
    if st.sidebar.button("Calculate Option Price"):
        st.write("Calculating option price using Binomial Tree model...")
        tree_param = fn.tree_parametres( S, K, T, r, sigma, N)
        #tree_param.S0, tree_param.K, tree_param.T, tree_param.r, tree_param.sigma, tree_param.N = S, K, T, r, sigma, N
        valeur = fn.tree(tree_param)
        st.write(f"Option Price: {valeur:.2f} €")
elif selected_model == "Monte Carlo Simulation":
    S = st.sidebar.number_input("Current stock price (S):", value=100.0)
    K = st.sidebar.number_input("Strike price (K):", value=100.0)
    T = st.sidebar.number_input("Time to expiration (T in years):", value=1.0)
    r = st.sidebar.number_input("Risk-free interest rate (r):", value=0.05)
    sigma = st.sidebar.number_input("Volatility of the underlying asset (σ):", value=0.2)
    M = st.sidebar.number_input("Number of simulations (M):", value=10000)
    if st.sidebar.button("Calculate Option Price"):
        st.write("Calculating option price using Monte Carlo Simulation...")
        mc_para = fn.MC_parametres(M, M, S, K, T, r, sigma)
        #mc_para.nb_simulations, mc_para.nb_paths, mc_para.S0, mc_para.K, mc_para.T, mc_para.r, mc_para.sigma = M, M, S, K, T, r, sigma
        valeur = fn.monte_carlo_call(mc_para)
        st.write(f"Option Price: {valeur:.2f} €")
elif selected_model == "Heston Model":
    S = st.sidebar.number_input("Current stock price (S):", value=100.0)
    K = st.sidebar.number_input("Strike price (K):", value=100.0)
    T = st.sidebar.number_input("Time to expiration (T in years):", value=1.0)
    r = st.sidebar.number_input("Risk-free interest rate (r):", value=0.05)
    v0 = st.sidebar.number_input("Initial volatility (v0):", value=0.04)
    theta = st.sidebar.number_input("Long-term variance (θ):", value=0.04)
    kappa = st.sidebar.number_input("Rate of mean reversion (κ):", value=1.0)
    sigma_v = st.sidebar.number_input("Volatility of volatility (σv):", value=0.2)
    rho = st.sidebar.number_input("Correlation between asset and volatility (ρ):", value=-0.5)
    if st.sidebar.button("Calculate Option Price"):
        st.write("Calculating option price using Heston model...")
        st.write("Option Price: [Calculated Value]")
        heston_params = fn.HestonParams(S,v0,r,kappa,theta,sigma_v,rho)
        heston_price = fn.price_european_call_mc(heston_params, K, T, 252, 100, 42, False)
        st.write(f"Option price: {heston_price.price:.2f}€")
    
st.write("\n\n\n\n\n")


st.header("Performance of different models")
st.write("The models presented earlier are different types. They all aim to determine a price of an " \
"option based on different parameters. Some are analitycs like Black-Scholes, other are iterative like Monte-Carlo which is a method for resolving " \
"Black-Scholes in a more precise way by averaging solutions near the point of start.")

simu_df = pd.DataFrame({
    "N" :  [1,5,10,50,100,200,500,1500,2000,3000],
    "MAE_mc" : [0.8552, 0.4397, 0.5202, 0.334, 0.1591, 0.1792, 0.0772, 0.0448, 0.0764, 0.038],

"MAE_crr" : [0.0943, 0.0848, 0.0441, 0.0453, 0.0429, 0.0443, 0.0435, 0.0434, 0.0433, 0.0433],
"MAE_heston" : [0.6366, 2.5587, 0.7996, 0.207, 0.2854, 0.2796, 0.3567, 0.3415, 0.3536, 0.3347]
})
st.subheader("Mean Absolute Error (MAE) Comparison between Monte Carlo, Binomial Tree Models and Heston model")
#st.dataframe(simu_df)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(simu_df['N'], simu_df['MAE_mc'], 
            label='Monte Carlo (Stochastique)', 
            marker='o', linestyle='--', color='blue')
            
  
ax.plot(simu_df['N'], simu_df['MAE_crr'], 
            label='Arbre Binomial (Déterministe)', 
            marker='x', linestyle='-', color='red')

ax.plot(simu_df['N'], simu_df['MAE_heston'], 
            label='Heston (Stochastique)', 
            marker='+', linestyle='-', color='green')

ax.set_title("Évolution de l'Erreur Absolue Moyenne (MAE) selon N")
ax.set_xlabel("Nombre de pas / Simulations (N)")
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel("log(MAE (Erreur))")
ax.legend()
ax.grid(True, alpha=0.3)
    
st.pyplot(fig)
    
st.info("""
**1. La limite structurelle du CRR (Biais) :**
L'arbre binomial converge très vite (ligne plate dès $N=100$), ce qui le rend excellent pour le pricing rapide. Cependant, il atteint un **plancher d'erreur** (~0.043) qu'il ne parvient pas à franchir. C'est le **biais de discrétisation** : l'approximation par arbre garde un écart résiduel avec le modèle continu, même avec beaucoup de pas.

**2. La victoire asymptotique du Monte Carlo (Variance) :**
Bien que plus bruité au début (oscillations en $1/\sqrt{N}$), le Monte Carlo ne s'arrête jamais de converger. Le graphique montre un point de bascule vers N=3000
 où **le Monte Carlo devient plus précis que le CRR** (0.038 vs 0.043). Cela démontre que le Monte Carlo est un **estimateur sans biais** : avec une puissance de calcul suffisante, son erreur tend théoriquement vers 0 absolu.

**Conclusion :** Pour la rapidité, privilégier le **CRR**. Pour l'exactitude théorique absolue (sans biais), le **Monte Carlo** l'emporte à long terme.
""")


temps_df = pd.DataFrame({
    "N" :  [1,5,10,50,100,200,500,1500,2000,3000,5000,10000,20000],
    "temps_mc" : [4.0899e-05,2.8806e-05,3.167e-05,4.3507e-05,6.3612e-05,0.000449952,0.000245896,0.000309433,0.00231009,0.00169539,0.00146274,0.00288691,0.0054079],

"temps_crr" : [1.396e-05,5.951e-06,1.5606e-05,4.7364e-05,0.000156551,0.000819743,0.0026593,0.00372411,0.00730946,0.0381681,0.0896268,0.3416,1.49752],
"temps_heston" : [0.000969,0.002742,0.005349,.009907, 0.022831,0.477294, 0.068287,0.096268,0.145442,0.199295,0.477191, 0.931244, 1.905265 ]
})
st.subheader("Runtime Comparison between Monte Carlo and Binomial Tree Models")
#st.dataframe(temps_df)
time, ax = plt.subplots(figsize=(10, 6))
ax.plot( temps_df['N'],  temps_df['temps_mc'], 
            label='Monte Carlo (Stochastique)', 
            marker='o', linestyle='--', color='blue')
            
  
ax.plot(temps_df['N'], temps_df['temps_crr'], 
            label='Arbre Binomial (Déterministe)', 
            marker='x', linestyle='-', color='red')

ax.plot(temps_df['N'], temps_df['temps_heston'], 
            label='Heston (Stochastique)', 
            marker='+', linestyle='-', color='green')

ax.set_title("Évolution de runtime selon N")
ax.set_xlabel("Nombre de pas / Simulations (N)")
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel("log(Runtime)")
ax.legend()
ax.grid(True, alpha=0.3)
    
st.pyplot(time)

st.markdown("### Analyse de la Performance : Le mur de la complexité")

st.info("""
**1. L'avantage structurel du Monte Carlo (Scalabilité Linéaire) :**
Le temps de calcul du Monte Carlo croît de manière **linéaire** ($\mathcal{O}(N)$). Pour doubler la précision, il faut doubler le temps. Cette propriété rend le Monte Carlo très robuste pour les calculs intensifs ($N > 10,000$) ou les dimensions élevées, car le coût reste maîtrisé.

**2. L'explosion du coût du CRR (Complexité Quadratique) :**
Bien que nous ayons optimisé la mémoire en $\mathcal{O}(N)$, le *temps de calcul* de l'arbre reste **quadratique** ($\mathcal{O}(N^2)$) à cause des boucles imbriquées nécessaires pour parcourir les nœuds.
* Concrètement : Si on double $N$, le temps de calcul est multiplié par **4**.
* Conséquence : La courbe de runtime du CRR "décolle" verticalement pour les grands $N$, le rendant inutilisable pour une finesse extrême.

**Conclusion Technique :** Le **CRR** est imbattable sur les temps courts ($< 10ms$), mais le **Monte Carlo** est le seul choix viable pour la haute performance à grande échelle.
""")


# Chargement
df = pd.read_csv('/workspaces/finance-/csv/compa.csv')


# 1. Création de la grille régulière
x_range = np.linspace(0.75, 1.25, 30)
y_range = np.linspace(df['T'].min(), df['T'].max(), 30)
X, Y = np.meshgrid(x_range, y_range)

# 2. Interpolation pour la Surface Théorique (SABR)
# On imagine que 'S_theoretical_iv' est la surface lisse
Z_model = griddata(
    (df['moneyness'], df['T']), 
    df['MAE_BS'], 
    (X, Y), 
    method='cubic' # ou 'linear' si cubic fait des vagues bizarres
)


fig_calibration = go.Figure()

# A. La Surface du Modèle (Le Drapé Lisse)
fig_calibration.add_trace(go.Surface(
    z=Z_model, x=X, y=Y,
    colorscale='Viridis',
    opacity=0.8,
    name='BS',
    showscale=False
))


fig_calibration.update_layout(
    title="<b>Evaluation de la performance du modèle de BS",
    scene=dict(
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Maturity (Years)',
        zaxis_title='MAE (BS)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) # Angle de vue
    ),
    template='plotly_white',
    height=700
)
#st.plotly_chart(fig_calibration)



x_range = np.linspace(0.75, 1.25, 30)
y_range = np.linspace(df['T'].min(), df['T'].max(), 30)
X, Y = np.meshgrid(x_range, y_range)

# 2. Interpolation pour la Surface Théorique (SABR)
# On imagine que 'S_theoretical_iv' est la surface lisse
Z_model = griddata(
    (df['moneyness'], df['T']), 
    df['MAE_CRR'], 
    (X, Y), 
    method='cubic' # ou 'linear' si cubic fait des vagues bizarres
)


fig_calibration_crr = go.Figure()

# A. La Surface du Modèle (Le Drapé Lisse)
fig_calibration_crr.add_trace(go.Surface(
    z=Z_model, x=X, y=Y,
    colorscale='Viridis',
    opacity=0.8,
    name='CRR',
    showscale=False
))


fig_calibration_crr.update_layout(
    title="<b>Evaluation de la performance du modèle de CRR",
    scene=dict(
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Maturity (Years)',
        zaxis_title='MAE (CRR)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) # Angle de vue
    ),
    template='plotly_white',
    height=700
)
#st.plotly_chart(fig_calibration_crr)



df_mc = df[df['MAE_MC'] < 0.2]

x_range = np.linspace(0.75, 1.25, 30)
y_range = np.linspace(df_mc['T'].min(), df_mc['T'].max(), 30)
X, Y = np.meshgrid(x_range, y_range)


Z_model = griddata(
    (df_mc['moneyness'], df_mc['T']), 
    df_mc['MAE_MC'], 
    (X, Y), 
    method='linear' # ou 'linear' si cubic fait des vagues bizarres
)


fig_calibration_mc = go.Figure()

# A. La Surface du Modèle (Le Drapé Lisse)
fig_calibration_mc.add_trace(go.Surface(
    z=np.maximum(Z_model, 0),x=X, y=Y,
    colorscale='Viridis',
    opacity=0.8,
    name='MC',
    showscale=False
))


fig_calibration_mc.update_layout(
    title="<b>Evaluation de la performance du modèle de MC",
    scene=dict(
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Maturity (Years)',
        zaxis_title='MAE (MC)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) # Angle de vue
    ),
    template='plotly_white',
    height=700
)
#st.plotly_chart(fig_calibration_mc)


x_range = np.linspace(0.75, 1.25, 30)
y_range = np.linspace(df['T'].min(), df['T'].max(), 30)
X, Y = np.meshgrid(x_range, y_range)

# 2. Interpolation pour la Surface Théorique (SABR)
# On imagine que 'S_theoretical_iv' est la surface lisse
Z_model = griddata(
    (df['moneyness'], df['T']), 
    df['MAE_Heston'], 
    (X, Y), 
    method='linear' # ou 'linear' si cubic fait des vagues bizarres
)


fig_calibration_h = go.Figure()

# A. La Surface du Modèle (Le Drapé Lisse)
fig_calibration_h.add_trace(go.Surface(
    z=Z_model, x=X, y=Y,
    colorscale='Viridis',
    opacity=0.8,
    name='Heston',
    showscale=False
))


fig_calibration_h.update_layout(
    title="<b>Evaluation de la performance du modèle de Heston",
    scene=dict(
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Maturity (Years)',
        zaxis_title='MAE (Heston)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) # Angle de vue
    ),
    template='plotly_white',
    height=700
)
#st.plotly_chart(fig_calibration_h)


st.markdown("Comparaison Globale des Modèles")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Black-Scholes")
    # key="bs_chart" assure l'unicité
    st.plotly_chart(fig_calibration, use_container_width=True, key="bs_chart")

with col2:
    st.subheader("2. Heston (Stochastique)")
    st.plotly_chart(fig_calibration_h, use_container_width=True, key="heston_chart")

col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Monte Carlo")
    st.plotly_chart(fig_calibration_mc, use_container_width=True, key="mc_chart")

with col4:
    st.subheader("4. Arbre CRR")
    st.plotly_chart(fig_calibration_crr, use_container_width=True, key="crr_chart")


st.markdown("""
### Analyse Comparative des Surfaces d'Erreur

Cette visualisation 3D expose les faiblesses structurelles et numériques de chaque modèle en cartographiant l'**Erreur Absolue Moyenne (MAE)** en fonction de la **Moneyness** ($S/K$) et de la **Maturité** ($T$).

#### 1. Black-Scholes : La limite du "Monde Plat"
* **Observation :** La surface forme une "cuvette" ou un "U". L'erreur est faible à la monnaie (Moneyness $\\approx 1$) mais explose dès qu'on s'éloigne vers les options *In-The-Money* ou *Out-Of-The-Money*.
* **Interprétation Financière :** C'est la signature classique du **Smile de Volatilité**. Le marché intègre des probabilités de krach (queues de distribution épaisses) que l'hypothèse de volatilité constante ($\sigma$) de Black-Scholes ne peut pas capturer.

#### 2. CRR & Monte Carlo : L'impact Numérique
* **Observation :** Ces surfaces ressemblent globalement à celle de Black-Scholes (car ils convergent vers la même théorie), mais avec des artefacts spécifiques.
    * **CRR (Arbre) :** On observe parfois des "vagues" ou instabilités sur les très courtes maturités ($T \\to 0$). C'est un effet de discrétisation lorsque la grille de l'arbre n'est pas assez fine pour capturer la courbure du payoff.
    * **Monte Carlo :** La surface est plus "rugueuse" ou bruitée, reflétant la variance statistique inhérente à la méthode.

#### 3. Heston : L'apport de la Volatilité Stochastique
* **Observation :** La nappe est remarquablement plus **plate** et "basse" (valeurs de MAE proches de 0 partout). Les ailes (bords du graphique) sont corrigées.
* **Conclusion :** En modélisant la variance comme un processus aléatoire corrélé au prix, le modèle de Heston parvient à "fitter" le smile du marché. Il capture la dynamique réelle des prix là où Black-Scholes échoue structurellement.
""")