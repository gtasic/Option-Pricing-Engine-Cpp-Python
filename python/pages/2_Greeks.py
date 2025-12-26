import streamlit as st
import pandas as pd 
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn
import numpy as np
import supabase 
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
sys.path.append("/workspaces/finance-/build")
import sabr
from scipy.interpolate import griddata

load_dotenv()
supabase_url  = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")


supabase_client = supabase.create_client(supabase_url, supabase_key)
st.title("Option Greeks Analysis")



st.write("One of the key points of the project is to calculate the Greeks for options, which are essential for risk management and hedging strategies."
          "The Greeks measure the sensitivity of the option's price to various factors such as changes in the underlying asset price, volatility, time decay, and interest rates."
          "In the Greeks and Models section, we provide detailed explanations of each Greek and how they are calculated using different models."
          "Understanding the Greeks helps traders and investors make informed decisions about their options positions and manage their risk effectively.")

st.image("/workspaces/finance-/python/png/greeks_vs_mid_price.png", caption= 'Option Greeks Overview')
st.markdown("Figure: This infographic provides an overview of the primary Greeks used in options trading. Each Greek measures a different aspect of risk associated with an options position, helping traders to understand how their portfolios may respond to changes in market conditions.")
st.write("Delta (Δ) measures the sensitivity of the option's price to changes in the underlying asset's price. A delta of 0.5 means that for every 1 increase in the underlying asset, the option's price is expected to increase by 0.50.\n")
st.write("Gamma (Γ) measures the rate of change of delta with respect to changes in the underlying asset's price. It helps traders understand how delta will change as the underlying price moves.")
st.write("Vega (ν) measures the sensitivity of the option's price to changes in the volatility of the underlying asset. A higher vega indicates that the option's price is more sensitive to changes in volatility.")
st.write("Theta (Θ) measures the sensitivity of the option's price to the passage of time, also known as time decay. Options lose value as they approach expiration, and theta quantifies this effect.")
st.write("Rho (ρ) measures the sensitivity of the option's price to changes in interest rates. It indicates how much the option's price will change for a 1% change in interest rates.")
st.write("By understanding and monitoring these Greeks, traders can better manage their options portfolios and implement effective hedging strategies to mitigate risk.")
st.write("In the Current Positions section, users can view their open options positions along with their associated Greeks."
         "This section provides a comprehensive overview of the portfolio's risk exposure and helps users make informed decisions about their trading strategies."
         "The aggregated Greeks for the entire portfolio are also displayed, giving users a quick snapshot of their overall risk profile.")



df = pd.DataFrame(supabase_client.table("simulation_params").select("*").lt("sigma", 0.5).execute().data)
df = df.dropna()
df['moneyness'] = df['strike'] / df['S0'] 
df = df[df['moneyness'] < 1.3]
df = df[df['moneyness'] > 0.7]




x_range = np.linspace(df['T'].min(), df['T'].max(), 30)
y_range = np.linspace(df['moneyness'].min(), df['moneyness'].max(), 30)
X, Y = np.meshgrid(x_range, y_range)

# 2. Interpolation pour la Surface Théorique (SABR)
# On imagine que 'S_theoretical_iv' est la surface lisse
Z_model = griddata(
    (df['T'], df['moneyness']), 
    df['sigma'], 
    (X, Y), 
    method='linear' # ou 'linear' si cubic fait des vagues bizarres
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
    title="<b>Naive Volatility Surface Visualization</b>",
    scene=dict(
        xaxis_title='Maturity (Years)',
        yaxis_title='Moneyness',
        zaxis_title='Implied Volatility',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)) # Angle de vue
    ),
    template='plotly_white',
    height=700
)

st.plotly_chart(fig_calibration)



st.write("The 3D graph above represents the volatility surface, which is a three-dimensional plot showing the implied volatility of options across different strike prices and maturities."
         "The X-axis represents the time to maturity of the options, the Y-axis represents the strike prices, and the Z-axis represents the implied volatility."
         "Traders and analysts use the volatility surface to understand market expectations of future volatility and to identify potential mispricings in options."
         "The shape of the volatility surface can provide insights into market sentiment, with common patterns such as 'volatility smiles' "
         "or 'volatility skews' indicating varying levels of risk perception among market participants.")

  
st.subheader("Implied Volatility Surface Analysis")

st.markdown("""
The **Volatility Surface** represents the "risk map" anticipated by the market. 
Unlike the Black-Scholes model, which assumes a constant volatility ($\sigma$), we observe here a complex surface that varies along two major axes:
""")

# Creating two columns to structure the explanation
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Moneyness Axis: The Skew")
    st.info("""
    **Observation:** The curve is neither flat nor perfectly symmetrical.
    
    * **The 'Smirk' Profile:** For Equities, we typically observe higher volatility for *Out-Of-The-Money* Puts (Moneyness < 1) compared to Calls (Moneyness > 1).
    * **Financial Interpretation:** This reflects **"Crash Fear"**. Investors are willing to pay a premium (higher implied vol) to hedge against sudden market drops via Puts. This creates the characteristic downward slope (skew) to the right.
    """)

with col2:
    st.markdown("#### 2. Maturity Axis")
    st.info("""
    **Observation:** The shape of the curve evolves with the time horizon ($T$).
    
    * **Short Term ($T \\to 0$):** The curvature is very pronounced (steep 'V' or 'U' shape). The market reacts violently to immediate uncertainties (Earnings, Fed announcements).
    * **Long Term ($T \\to 2$):** The surface flattens out.
    * **Financial Interpretation:** This is the **Mean Reversion** effect. Over the long run, shocks are smoothed out, and volatility tends to revert to its historical average.
    """)

st.success("""
**Conclusion for Modeling:**
The existence of this surface demonstrates the limitations of the constant volatility hypothesis (Black-Scholes). 
To price options accurately across this surface, it is mandatory to implement Local or Stochastic Volatility models (such as **Heston** or **SABR**) capable of reproducing both the *Skew* and the *Term Structure flattening*.
""")

