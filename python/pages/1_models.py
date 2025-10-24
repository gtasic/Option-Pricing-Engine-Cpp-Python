import streamlit as st
import pandas as pd
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn


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
        # Here you would call your pricing function and display the result
        tree_param = fn.tree_parametres()
        tree_param.S0, tree_param.K, tree_param.T, tree_param.r, tree_param.sigma, tree_param.N = S, K, T, r, sigma, N
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
        mc_para = fn.MC_parametres()
        mc_para.nb_simulations, mc_para.nb_paths, mc_para.S0, mc_para.K, mc_para.T, mc_para.r, mc_para.sigma = M, M, S, K, T, r, sigma
        valeur = fn.monte_carlo_call(mc_para)
        st.write(f"Option Price: {valeur:.2f} €")
        # Here you would call your pricing function and display the result
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
        # Here you would call your pricing function and display the result
        st.write("Option Price: [Calculated Value]")
