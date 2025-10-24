import streamlit as st
import pandas as pd 
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn




st.title("Interface for Financial Backtesting and Option Pricing")
st.write("This application allows users to perform backtesting of trading strategies and price options using various models based on the scripts I developped.")
st.write("The backtesting logic is implemented in my project, while the option pricing and analysis functionalities are handled in chat.py using OpenAI's GPT-4 model.")

st.header("Backtesting Module")
st.write("The backtesting module utilizes historical market data to evaluate trading strategies. It includes functions for selecting options based on criteria such as days to expiration and delta values.")




