import streamlit as st
import pandas as pd 
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn


erreur = "/workspaces/finance-/python/error_boxplot_by_maturity.png"

st.title("Interface for Financial Backtesting and Option Pricing")
st.write("This application allows users to perform backtesting of trading strategies and price options using various"
            " models based on the scripts I developped.")
st.write("The backtesting logic is implemented in my project, while the option pricing and analysis functionalities are handled in chat.py using OpenAI's GPT-4 model.")

st.header("Models Comparison Module")
st.write("The models used in this project for option pricing include Black-Scholes, Binomial Tree, Monte Carlo Simulation, and Heston Model."
         " Each model has its own assumptions and parameters that influence the pricing of options. You can explore the details of each model in the Models section."
         "We are going differences between these models and their pricing outputs in the Models Comparison section.")

st.image(erreur, caption= 'Error Boxplot by Maturity')
st.markdown("Figure: This boxplot illustrates the distribution of pricing errors across different maturities for the various models implemented. It helps in visualizing which models perform better at certain maturities and highlights any outliers in the pricing errors.")

st.write("In this chart we can see that precision varies significantly with maturity. Short-term options tend to have lower pricing errors across all models, while long-term options exhibit higher variability in errors. This suggests that the models may need adjustments or additional factors to accurately price long-term options."
         "Moreover, between the models, we can see that there are not much differencies in terms of errors for low maturity but as maturity increases" \
         "Monte Carlo models seem to perform worse compared to the others, likely due to the increased uncertainty and complexity in simulating long-term price paths.")

st.write("We can compare two models in particular : the Monte Carlo model and the Binomial Tree model." 
         "As they are both based on different numerical methods to approximate option prices, they can yield different results under the same market conditions."
         "Whereas Black-Scholes gives an analitical solution for European options, both Monte Carlo and Binomial Tree methods are numerical approaches that can be used for a wider range of option types but may introduce approximation errors."
            "The Binomial Tree model constructs a discrete-time lattice of possible underlying asset prices, allowing for the valuation of American options and other path-dependent features."
            "On the other hand, the Monte Carlo simulation relies on generating a large number of random price paths for the underlying asset to estimate the option's expected payoff.")

simu_df = pd.DataFrame({
    "N" : [10,25,50,100,200,500,750,1000,1500,3000,5000,7500],
    "MAE_mc" : [15.432844,12.699282,11.699282,8.192795,7.119168,5.961601,5.954711,6.322372,5.783767,5.358887,5.432844,5.120945],
      "MAE_crr" : [9.640285e+126,9.640285e+126,9.640285e+126,7.728346e+254,1.352739e+244,3.249037,3.248539,3.248497,3.248517,3.248321,3.248228,3.248332]
})
st.subheader("Mean Absolute Error (MAE) Comparison between Monte Carlo and Binomial Tree Models")
st.dataframe(simu_df)

st.markdown("""
The table above presents the Mean Absolute Error (MAE) for both the Monte Carlo (MAE_mc) and Binomial Tree (MAE_crr) models across different numbers of simulations or steps (N).
- **N**: Represents the number of simulations (for Monte Carlo) or steps (for Binomial Tree).
- **MAE_mc**: Mean Absolute Error for the Monte Carlo model. As N increases, the MAE generally decreases, indicating improved accuracy with more simulations.
- **MAE_crr**: Mean Absolute Error for the Binomial Tree model. The values appear to be extremely large for lower N, suggesting potential instability or inaccuracies in the model at those levels. However, as N increases, the MAE stabilizes to a more reasonable range.
From the data, we can observe that the Monte Carlo model shows a consistent decrease in MAE as the number of simulations increases, demonstrating its convergence towards more accurate pricing. In contrast, the Binomial Tree model exhibits very high MAE values at lower N, which may indicate issues with model implementation or sensitivity to the number of steps. As N increases, the Binomial Tree model's MAE stabilizes, suggesting that it becomes more reliable with a higher number of steps.
""")

st.write("One of the key points of the project is to calculate the Greeks for options, which are essential for risk management and hedging strategies."
          "The Greeks measure the sensitivity of the option's price to various factors such as changes in the underlying asset price, volatility, time decay, and interest rates."
          "In the Greeks and Models section, we provide detailed explanations of each Greek and how they are calculated using different models."
          "Understanding the Greeks helps traders and investors make informed decisions about their options positions and manage their risk effectively.")

st.image("/workspaces/finance-/python/png/greeks_vs_mid_price.png", caption= 'Option Greeks Overview')
st.markdown("Figure: This infographic provides an overview of the primary Greeks used in options trading. Each Greek measures a different aspect of risk associated with an options position, helping traders to understand how their portfolios may respond to changes in market conditions.")
st.write("Delta (Δ) measures the sensitivity of the option's price to changes in the underlying asset's price. A delta of 0.5 means that for every $1 increase in the underlying asset, the option's price is expected to increase by $0.50."
         "Gamma (Γ) measures the rate of change of delta with respect to changes in the underlying asset's price. It helps traders understand how delta will change as the underlying price moves."
         "Vega (ν) measures the sensitivity of the option's price to changes in the volatility of the underlying asset. A higher vega indicates that the option's price is more sensitive to changes in volatility."
         "Theta (Θ) measures the sensitivity of the option's price to the passage of time, also known as time decay. Options lose value as they approach expiration, and theta quantifies this effect."
         "Rho (ρ) measures the sensitivity of the option's price to changes in interest rates. It indicates how much the option's price will change for a 1% change in interest rates."
         "By understanding and monitoring these Greeks, traders can better manage their options portfolios and implement effective hedging strategies to mitigate risk.") 
st.write("In the Current Positions section, users can view their open options positions along with their associated Greeks."
         "This section provides a comprehensive overview of the portfolio's risk exposure and helps users make informed decisions about their trading strategies."
         "The aggregated Greeks for the entire portfolio are also displayed, giving users a quick snapshot of their overall risk profile.")


st.header("3D Volatility Surface Visualization")
