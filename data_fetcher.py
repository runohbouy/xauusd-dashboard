"""
data_fetcher.py — modular, cacheable data fetchers.
Every fetch writes to cache/ as a Parquet file with a timestamp.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from config import (
    CACHE_DIR, FRED_API_KEY, APIFY_API_TOKEN, TWITTER_BEARER_TOKEN,
    HISTORY_YEARS, INFLUENTIAL_ACCOUNTS,
)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _is_fresh(path: str, max_age_minutes: int) -> bool:
    if not os.path.exists(path):
        return False
    age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path)))
    return age < timedelta(minutes=max_age_minutes)


def _save(df: pd.DataFrame, name: str):
    df.to_parquet(_cache_path(name))


def _load(name: str) -> pd.DataFrame:
    path = _cache_path(name)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


# ------------------------------------------------------------
# 1. Gold price
# ------------------------------------------------------------
def fetch_gold(max_age_minutes: int = 15) -> pd.DataFrame:
    if _is_fresh(_cache_path("gold"), max_age_minutes):
        return _load("gold")

    end = datetime.today()
    start = end - timedelta(days=HISTORY_YEARS * 365)
    df = yf.download("GC=F", start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return _load("gold")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["gold_open", "gold_high", "gold_low", "gold_close", "gold_volume"]
    df.sort_index(inplace=True)

    _save(df, "gold")
    return df


# ------------------------------------------------------------
# 2. DXY
# ------------------------------------------------------------
def fetch_dxy(max_age_minutes: int = 15) -> pd.DataFrame:
    if _is_fresh(_cache_path("dxy"), max_age_minutes):
        return _load("dxy")

    end = datetime.today()
    start = end - timedelta(days=HISTORY_YEARS * 365)
    df = yf.download("DX-Y.NYB", start=start, end=end, progress=False)
    if df.empty:
        return _load("dxy")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Close"]].copy()
    df.columns = ["dxy_close"]
    _save(df, "dxy")
    return df


# ------------------------------------------------------------
# 3. Macro indicators (FRED)
# ------------------------------------------------------------
FRED_SERIES = {
    "fed_funds_rate": "DFF",
    "us10y_yield": "DGS10",
    "real_10y_yield": "DFII10",
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "vix": "VIXCLS",
    "unemployment": "UNRATE",
    "m2": "M2SL",
    "dxy_broad": "DTWEXBGS",
}


def fetch_macro(max_age_minutes: int = 360) -> pd.DataFrame:
    if _is_fresh(_cache_path("macro"), max_age_minutes):
        return _load("macro")
    if not FRED_API_KEY:
        return _load("macro")

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
    except Exception:
        return _load("macro")

    end = datetime.today()
    start = end - timedelta(days=HISTORY_YEARS * 365)
    frames = {}
    for name, sid in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            s.index = pd.to_datetime(s.index)
            frames[name] = s
        except Exception:
            continue
        time.sleep(0.2)

    if not frames:
        return _load("macro")

    df = pd.concat(frames, axis=1).sort_index()
    _save(df, "macro")
    return df


# ------------------------------------------------------------
# 4. Economic calendar (Apify)
# ------------------------------------------------------------
def fetch_calendar(max_age_minutes: int = 720) -> pd.DataFrame:
    if _is_fresh(_cache_path("calendar"), max_age_minutes):
        return _load("calendar")
    if not APIFY_API_TOKEN:
        return _load("calendar")

    try:
        from apify_client import ApifyClient
    except Exception:
        return _load("calendar")

    client = ApifyClient(APIFY_API_TOKEN)
    end = datetime.today()
    start = end - timedelta(days=90)

    try:
        run = client.actor("axery/forexfactory-economic-calendar-scraper").call(
            run_input={
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate": end.strftime("%Y-%m-%d"),
                "impact": ["High"],
                "currencies": ["USD"],
            }
        )
        items = list(client.dataset(run.default_dataset_id).iterate_items())
        df = pd.DataFrame(items)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        _save(df, "calendar")
        return df
    except Exception:
        return _load("calendar")


# ------------------------------------------------------------
# 5. Twitter sentiment (FinBERT)
# ------------------------------------------------------------
# data_fetcher.py — replace the _score_sentiment function

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()
def _score_sentiment(texts):
    """
    Lightweight sentiment scoring using VADER.
    Returns list of (label, score) tuples.
    Compatible with Streamlit Community Cloud (no torch, no transformers).
    """
    results = []
    for text in texts:
        scores = _analyzer.polarity_scores(str(text))
        compound = scores["compound"]  # range: -1.0 to 1.0
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        results.append((label, compound))
    return results


def fetch_twitter(max_age_minutes: int = 30) -> pd.DataFrame:
    if _is_fresh(_cache_path("twitter"), max_age_minutes):
        return _load("twitter")
    if not TWITTER_BEARER_TOKEN:
        return _load("twitter")

    try:
        import tweepy
    except Exception:
        return _load("twitter")

    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    rows = []
    for acct in INFLUENTIAL_ACCOUNTS:
        try:
            u = client.get_user(username=acct)
            if not u.data:
                continue
            tweets = client.get_users_tweets(
                u.data.id, max_results=50,
                tweet_fields=["created_at", "text", "public_metrics"]
            )
            if tweets.data:
                for t in tweets.data:
                    rows.append({
                        "account": acct,
                        "date": pd.to_datetime(t.created_at).tz_localize(None),
                        "text": t.text,
                        "likes": t.public_metrics.get("like_count", 0),
                    })
        except Exception:
            continue
        time.sleep(1)

    if not rows:
        return _load("twitter")

    df = pd.DataFrame(rows)
    labels_scores = _score_sentiment(df["text"].tolist())
    df["sentiment_label"] = [x[0] for x in labels_scores]
    df["sentiment_score"] = [x[1] for x in labels_scores]
    df["date"] = df["date"].dt.normalize()
    _save(df, "twitter")
    return df


# ------------------------------------------------------------
# 6. Truth Social (GitHub archive fallback)
# ------------------------------------------------------------
def fetch_truth_social(max_age_minutes: int = 60) -> pd.DataFrame:
    if _is_fresh(_cache_path("truth_social"), max_age_minutes):
        return _load("truth_social")

    url = ("https://raw.githubusercontent.com/"
           "stiles/trump-truth-social-archive/main/posts.csv")
    try:
        df = pd.read_csv(url)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
        if "content" in df.columns:
            labels_scores = _score_sentiment(df["content"].astype(str).tolist())
            df["sentiment_label"] = [x[0] for x in labels_scores]
            df["sentiment_score"] = [x[1] for x in labels_scores]
        _save(df, "truth_social")
        return df
    except Exception:
        return _load("truth_social")