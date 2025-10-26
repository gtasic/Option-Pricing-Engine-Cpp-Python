import streamlit as st
import pandas as pd
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn
import supabase 

import plotly.graph_objects as go
supabase_url: str = "https://wehzchguwwpopqpzyvpc.supabase.co"
supabase_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndlaHpjaGd1d3dwb3BxcHp5dnBjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc3MTE1OTQsImV4cCI6MjA3MzI4NzU5NH0.hK5fX9YowK83jx8MAzzNm5enBdvgU2XC4shZreACO2s"



supabase_client = supabase.create_client(supabase_url, supabase_key)
def get_open_positions():
    response = supabase_client.table("portfolio_options").select("*").eq("status", "open").execute()
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