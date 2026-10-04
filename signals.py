"""signals.py — computes technical + composite signals from raw data."""

import numpy as np
import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_20"] = df["gold_close"].rolling(20).mean()
    df["sma_50"] = df["gold_close"].rolling(50).mean()
    df["sma_200"] = df["gold_close"].rolling(200).mean()

    delta = df["gold_close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["bb_mid"] = df["gold_close"].rolling(20).mean()
    df["bb_std"] = df["gold_close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    hl = df["gold_high"] - df["gold_low"]
    hc = (df["gold_high"] - df["gold_close"].shift()).abs()
    lc = (df["gold_low"] - df["gold_close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    ema_12 = df["gold_close"].ewm(span=12).mean()
    ema_26 = df["gold_close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["gold_return_1d"] = df["gold_close"].pct_change()
    df["volatility_20d"] = df["gold_return_1d"].rolling(20).std() * np.sqrt(252)
    return df


def compute_composite_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signal_trend"] = np.where(df["gold_close"] > df["sma_50"], 1, -1)

    df["signal_rsi"] = 0
    df.loc[df["rsi_14"] < 30, "signal_rsi"] = 1
    df.loc[df["rsi_14"] > 70, "signal_rsi"] = -1

    if "real_10y_yield" in df.columns:
        df["signal_macro"] = np.where(df["real_10y_yield"].diff() < 0, 1, -1)
    else:
        df["signal_macro"] = 0

    sent_cols = [c for c in df.columns if c.startswith("tw_sent_") or c == "ts_sent"]
    if sent_cols:
        df["signal_sentiment"] = df[sent_cols].mean(axis=1).apply(
            lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0)
        )
    else:
        df["signal_sentiment"] = 0

    cols = ["signal_trend", "signal_rsi", "signal_macro", "signal_sentiment"]
    df["signal_composite"] = df[cols].mean(axis=1)
    return df


def compute_market_regime(df: pd.DataFrame) -> str:
    """Classify current regime: risk-on / risk-off / neutral."""
    if df.empty or "vix" not in df.columns:
        return "unknown"
    vix = df["vix"].iloc[-1]
    real_yield_change = df["real_10y_yield"].diff().iloc[-5:].mean() if "real_10y_yield" in df.columns else 0
    dxy_change = df["dxy_close"].pct_change(5).iloc[-1] if "dxy_close" in df.columns else 0

    if vix > 25 and dxy_change > 0.005:
        return "risk-off (bullish gold)"
    if vix < 15 and real_yield_change > 0:
        return "risk-on (bearish gold)"
    return "neutral"