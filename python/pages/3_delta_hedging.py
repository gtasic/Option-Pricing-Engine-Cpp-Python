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

st.title("Delta Hedging Strategy Analysis")
st.write("This page explores the implementation and performance of a delta hedging strategy for options trading."
         "Delta hedging involves adjusting the hedge position in the underlying asset to maintain a delta-neutral portfolio, thereby minimizing risk from price movements of the underlying asset.")

portfolio_df = supabase_client.table("daily_portfolio_pnl").select("*").execute().data
portfolio_df = pd.DataFrame(portfolio_df)


col1, col2, col3, col4 = st.columns(4)
current_nav = 1_000_000  # Example current NAV
daily_pnl = 2_500  # Example daily PnL
sharpe = 1.8  # Example Sharpe ratio
max_dd = -0.03  # Example max drawdown
total_delta = 0.005  # Example total delta


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
                         mode='lines+markers',
                         name='Portfolio Value'))
fig.update_layout(title='Portfolio Value Over Time',
                   xaxis_title='Date',
                   yaxis_title='Portfolio Value (€)')
st.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns(2)
with col1:
    st.subheader("Daily PnL Distribution")
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Histogram(x=portfolio_df['daily_pnl'], nbinsx=50))
    fig_pnl.update_layout(title='Daily PnL Distribution',
                          xaxis_title='Daily PnL (€)',
                          yaxis_title='Frequency')
    st.plotly_chart(fig_pnl, use_container_width=True)

with col2:
    st.subheader("Delta Over Time")
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=portfolio_df['asof'], y=portfolio_df['total_delta'],
                                   mode='lines+markers',
                                   name='Delta'))
    fig_delta.update_layout(title='Delta Over Time',
                            xaxis_title='Date',
                            yaxis_title='Delta')
    st.plotly_chart(fig_delta, use_container_width=True)    
