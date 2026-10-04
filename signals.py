"""
signals.py — technical indicators and composite signal generation.
"""

import numpy as np
import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, RSI, Bollinger Bands, ATR, MACD, returns, volatility."""
    df = df.copy()

    # Moving averages
    df["sma_20"] = df["gold_close"].rolling(20).mean()
    df["sma_50"] = df["gold_close"].rolling(50).mean()
    df["sma_200"] = df["gold_close"].rolling(200).mean()

    # RSI (14)
    delta = df["gold_close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2σ)
    df["bb_mid"] = df["gold_close"].rolling(20).mean()
    df["bb_std"] = df["gold_close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # ATR (14)
    hl = df["gold_high"] - df["gold_low"]
    hc = (df["gold_high"] - df["gold_close"].shift()).abs()
    lc = (df["gold_low"] - df["gold_close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # MACD (12, 26, 9)
    ema_12 = df["gold_close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["gold_close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Returns and volatility
    df["gold_return_1d"] = df["gold_close"].pct_change()
    df["gold_return_5d"] = df["gold_close"].pct_change(5)
    df["volatility_20d"] = df["gold_return_1d"].rolling(20).std() * np.sqrt(252)

    return df


def compute_composite_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Compute four component signals and the composite."""
    df = df.copy()

    # Trend: above/below SMA 50
    df["signal_trend"] = np.where(df["gold_close"] > df["sma_50"], 1, -1)

    # RSI extremes
    df["signal_rsi"] = 0
    df.loc[df["rsi_14"] < 30, "signal_rsi"] = 1
    df.loc[df["rsi_14"] > 70, "signal_rsi"] = -1

    # Macro: falling real yields → bullish gold
    if "real_10y_yield" in df.columns:
        df["signal_macro"] = np.where(df["real_10y_yield"].diff() < 0, 1, -1)
    else:
        df["signal_macro"] = 0

    # Sentiment: GDELT news tone
    if "news_sent" in df.columns:
        df["signal_sentiment"] = df["news_sent"].apply(
            lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0)
        )
    else:
        df["signal_sentiment"] = 0

    cols = ["signal_trend", "signal_rsi", "signal_macro", "signal_sentiment"]
    df["signal_composite"] = df[cols].mean(axis=1)
    return df


def compute_market_regime(df: pd.DataFrame) -> str:
    """Classify current market regime based on VIX, DXY, and real yields."""
    if df.empty:
        return "unknown"

    try:
        vix = float(df["vix"].dropna().iloc[-1]) if "vix" in df.columns else None
        dxy_chg = (
            float(df["dxy_close"].pct_change(5).dropna().iloc[-1])
            if "dxy_close" in df.columns else 0.0
        )
        ry_chg = (
            float(df["real_10y_yield"].diff().dropna().iloc[-5:].mean())
            if "real_10y_yield" in df.columns else 0.0
        )
    except Exception:
        return "unknown"

    if vix is not None and vix > 25 and dxy_chg > 0.005:
        return "risk-off (bullish gold)"
    if vix is not None and vix < 15 and ry_chg > 0:
        return "risk-on (bearish gold)"
    return "neutral"