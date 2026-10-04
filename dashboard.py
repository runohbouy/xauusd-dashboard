"""
dashboard.py — Streamlit dashboard for XAU/USD live signals.

Run locally:
    streamlit run dashboard.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import REFRESH_INTERVAL_SECONDS
from data_fetcher import (
    fetch_gold,
    fetch_dxy,
    fetch_macro,
    fetch_calendar,
    fetch_news_sentiment,
)
from signals import (
    add_technical_features,
    compute_composite_signal,
    compute_market_regime,
)


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="XAU/USD Live Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Data assembly
# ============================================================
@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS, show_spinner=False)
def load_all_data():
    gold = fetch_gold()
    dxy = fetch_dxy()
    macro = fetch_macro()
    calendar = fetch_calendar()
    news = fetch_news_sentiment()

    if gold.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = gold.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if not dxy.empty:
        dxy_idx = pd.to_datetime(dxy.index).tz_localize(None)
        dxy = dxy.copy()
        dxy.index = dxy_idx
        df = df.join(dxy, how="left")

    if not macro.empty:
        m = macro.copy()
        m.index = pd.to_datetime(m.index).tz_localize(None)
        m = m.resample("D").ffill()
        df = df.join(m, how="left")

    if not news.empty:
        news_daily = (
            news.groupby("date")["sentiment_score"]
            .mean()
            .rename("news_sent")
        )
        df = df.join(news_daily, how="left")
        df["news_sent"] = df["news_sent"].fillna(0.0)

    df = add_technical_features(df)
    df = compute_composite_signal(df)
    return df, calendar, news


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("⚙️ Controls")
time_range = st.sidebar.selectbox(
    "Time range", ["1M", "3M", "6M", "1Y", "3Y", "10Y"], index=2
)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
refresh_secs = st.sidebar.slider("Refresh every (s)", 30, 600, 60, 30)

if st.sidebar.button("🔄 Force refresh"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# Load
# ============================================================
with st.spinner("Loading live data..."):
    df, calendar, news = load_all_data()

if df.empty:
    st.error("No data available. Check API keys and network.")
    st.stop()

range_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 365 * 3, "10Y": 3650}
window_days = range_map[time_range]
cutoff = df.index.max() - pd.Timedelta(days=window_days)
view = df[df.index >= cutoff].copy()

if view.empty:
    st.warning("Not enough data for the selected range.")
    st.stop()


# ============================================================
# Header
# ============================================================
st.title("🥇 XAU/USD Live Signal Dashboard")
st.caption(
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    f"Range shown: {view.index.min().date()} → {view.index.max().date()}"
)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Gold",
    f"${latest['gold_close']:.2f}",
    f"{latest['gold_close'] - prev['gold_close']:+.2f}",
)
if "dxy_close" in df.columns:
    c2.metric(
        "DXY",
        f"{latest['dxy_close']:.2f}",
        f"{latest['dxy_close'] - prev['dxy_close']:+.2f}",
    )
else:
    c2.metric("DXY", "n/a")
c3.metric("RSI (14)", f"{latest['rsi_14']:.1f}" if pd.notna(latest["rsi_14"]) else "n/a")
c4.metric("ATR (14)", f"{latest['atr_14']:.2f}" if pd.notna(latest["atr_14"]) else "n/a")
regime = compute_market_regime(df)
c5.metric("Regime", regime.split(" ")[0].title())


# ============================================================
# Signal panel
# ============================================================
st.subheader("📡 Current Composite Signal")
sig_cols = st.columns(4)
sig_cols[0].metric("Trend", int(latest["signal_trend"]))
sig_cols[1].metric("RSI", int(latest["signal_rsi"]))
sig_cols[2].metric("Macro", int(latest["signal_macro"]))
sig_cols[3].metric("Sentiment", int(latest["signal_sentiment"]))

composite = float(latest["signal_composite"])
signal_label = "BUY" if composite > 0.2 else ("SELL" if composite < -0.2 else "NEUTRAL")
signal_icon = "🟢" if composite > 0.2 else ("🔴" if composite < -0.2 else "🟡")
st.markdown(
    f"### {signal_icon} Composite Score: **{composite:+.2f}** → **{signal_label}**"
)


# ============================================================
# Price chart with indicators
# ============================================================
st.subheader("📈 Price & Indicators")
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.6, 0.2, 0.2],
    vertical_spacing=0.03,
    subplot_titles=("Price + Bollinger Bands", "RSI", "MACD"),
)

fig.add_trace(
    go.Candlestick(
        x=view.index,
        open=view["gold_open"],
        high=view["gold_high"],
        low=view["gold_low"],
        close=view["gold_close"],
        name="Gold",
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=view.index, y=view["sma_50"],
               line=dict(color="orange", width=1), name="SMA 50"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=view.index, y=view["sma_200"],
               line=dict(color="blue", width=1), name="SMA 200"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=view.index, y=view["bb_upper"],
               line=dict(color="gray", dash="dot", width=1), name="BB Upper"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=view.index, y=view["bb_lower"],
               line=dict(color="gray", dash="dot", width=1), name="BB Lower"),
    row=1, col=1,
)

fig.add_trace(
    go.Scatter(x=view.index, y=view["rsi_14"],
               line=dict(color="purple"), name="RSI"),
    row=2, col=1,
)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

fig.add_trace(
    go.Bar(x=view.index, y=view["macd_hist"],
           name="MACD Hist", marker_color="steelblue"),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(x=view.index, y=view["macd"],
               line=dict(color="black"), name="MACD"),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(x=view.index, y=view["macd_signal"],
               line=dict(color="red"), name="Signal"),
    row=3, col=1,
)

fig.update_layout(
    height=800,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Macro panel
# ============================================================
st.subheader("🏦 Macro Drivers")
macro_cols = [
    c for c in ["real_10y_yield", "us10y_yield", "fed_funds_rate",
                "vix", "cpi", "unemployment"]
    if c in view.columns
]
if macro_cols:
    macro_view = view[macro_cols].tail(365).copy()
    macro_fig = make_subplots(rows=2, cols=3, subplot_titles=macro_cols)
    for i, col in enumerate(macro_cols):
        r, c = divmod(i, 3)
        macro_fig.add_trace(
            go.Scatter(x=macro_view.index, y=macro_view[col],
                       name=col, line=dict(width=1.5)),
            row=r + 1, col=c + 1,
        )
    macro_fig.update_layout(
        height=450, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(macro_fig, use_container_width=True)
else:
    st.info("No macro data. Check FRED_API_KEY.")


# ============================================================
# Sentiment
# ============================================================
st.subheader("📰 News Sentiment (GDELT)")
sent_tabs = st.tabs(["Daily Sentiment", "Recent Headlines"])

with sent_tabs[0]:
    if "news_sent" in view.columns and view["news_sent"].abs().sum() > 0:
        sent_fig = go.Figure()
        sent_fig.add_trace(go.Scatter(
            x=view.index, y=view["news_sent"],
            name="News sentiment", mode="lines+markers",
            line=dict(color="teal"),
        ))
        sent_fig.add_hline(y=0, line_dash="dot", line_color="gray")
        sent_fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(sent_fig, use_container_width=True)
    else:
        st.info(
            "No sentiment data yet. GDELT may take a minute on first load. "
            "Try clicking 'Force refresh' in the sidebar."
        )

with sent_tabs[1]:
    if not news.empty:
        recent = news.sort_values("date", ascending=False).head(30).copy()
        cols = [c for c in ["date", "source", "sentiment_label",
                            "sentiment_score", "title"] if c in recent.columns]
        st.dataframe(recent[cols], use_container_width=True, height=500)
    else:
        st.info("No headlines available.")


# ============================================================
# Economic calendar
# ============================================================
st.subheader("📅 High-Impact USD Events")
if not calendar.empty and "date" in calendar.columns:
    upcoming = calendar[
        calendar["date"] >= pd.Timestamp.now() - pd.Timedelta(days=7)
    ].sort_values("date", ascending=False).head(20)
    cols = [c for c in ["date", "time", "event", "impact",
                        "forecast", "previous"] if c in upcoming.columns]
    st.dataframe(upcoming[cols], use_container_width=True, height=300)
else:
    st.info("No calendar data. Configure APIFY_API_TOKEN.")


# ============================================================
# Export
# ============================================================
with st.expander("📥 Export / Inspect Data"):
    st.download_button(
        "Download merged dataset (CSV)",
        data=df.to_csv().encode("utf-8"),
        file_name=f"xauusd_merged_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )
    st.dataframe(df.tail(50), use_container_width=True)


# ============================================================
# Auto-refresh
# ============================================================
if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()