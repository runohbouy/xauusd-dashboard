"""
data_fetcher.py — modular, cacheable data fetchers.

Sources:
  - Gold price: Yahoo Finance (GC=F)
  - DXY: Yahoo Finance (DX-Y.NYB)
  - Macro: FRED API
  - Calendar: Apify (ForexFactory scraper)
  - News sentiment: GDELT (free, no key)

All fetchers cache to cache/*.parquet with a freshness check.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    CACHE_DIR,
    FRED_API_KEY,
    APIFY_API_TOKEN,
    HISTORY_YEARS,
    NEWS_QUERIES,
)


# ============================================================
# Cache helpers
# ============================================================
def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _is_fresh(path: str, max_age_minutes: int) -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(minutes=max_age_minutes)


def _save(df: pd.DataFrame, name: str) -> None:
    try:
        df.to_parquet(_cache_path(name))
    except Exception as e:
        print(f"Cache save failed for {name}: {e}")


def _load(name: str) -> pd.DataFrame:
    path = _cache_path(name)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# ============================================================
# 1. Gold price (10 years, daily)
# ============================================================
def fetch_gold(max_age_minutes: int = 15) -> pd.DataFrame:
    if _is_fresh(_cache_path("gold"), max_age_minutes):
        return _load("gold")

    end = datetime.today()
    start = end - timedelta(days=HISTORY_YEARS * 365)

    try:
        df = yf.download(
            "GC=F",
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:
        print(f"Gold fetch failed: {e}")
        return _load("gold")

    if df is None or df.empty:
        return _load("gold")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["gold_open", "gold_high", "gold_low", "gold_close", "gold_volume"]
    df.sort_index(inplace=True)

    _save(df, "gold")
    return df


# ============================================================
# 2. DXY
# ============================================================
def fetch_dxy(max_age_minutes: int = 15) -> pd.DataFrame:
    if _is_fresh(_cache_path("dxy"), max_age_minutes):
        return _load("dxy")

    end = datetime.today()
    start = end - timedelta(days=HISTORY_YEARS * 365)

    try:
        df = yf.download(
            "DX-Y.NYB",
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception:
        return _load("dxy")

    if df is None or df.empty:
        return _load("dxy")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Close"]].copy()
    df.columns = ["dxy_close"]
    df.sort_index(inplace=True)

    _save(df, "dxy")
    return df


# ============================================================
# 3. Macro indicators (FRED)
# ============================================================
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
    except Exception as e:
        print(f"FRED init failed: {e}")
        return _load("macro")

    end = datetime.today()
    start = end - timedelta(days=HISTORY_YEARS * 365)

    frames = {}
    for name, sid in FRED_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            frames[name] = s
        except Exception:
            continue
        time.sleep(0.2)

    if not frames:
        return _load("macro")

    df = pd.concat(frames, axis=1).sort_index()
    _save(df, "macro")
    return df


# ============================================================
# 4. Economic calendar (Apify)
# ============================================================
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
    except Exception as e:
        print(f"Calendar fetch failed: {e}")
        return _load("calendar")


# ============================================================
# 5. News sentiment (GDELT — free, no key)
# ============================================================
def fetch_news_sentiment(max_age_minutes: int = 30) -> pd.DataFrame:
    """
    Fetch gold-related news headlines from GDELT.
    Tone is pre-computed by GDELT and scaled to -1..1 via tanh.
    """
    if _is_fresh(_cache_path("news_sentiment"), max_age_minutes):
        return _load("news_sentiment")

    rows = []
    for q in NEWS_QUERIES:
        try:
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": q,
                    "mode": "ArtList",
                    "maxrecords": 250,
                    "format": "json",
                    "timespan": "14d",
                    "sort": "datedesc",
                },
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code != 200:
                continue
            payload = r.json()
            for a in payload.get("articles", []):
                date_val = a.get("seendate", "")
                try:
                    dt = pd.to_datetime(date_val, format="%Y%m%dT%H%M%SZ")
                except Exception:
                    continue
                rows.append({
                    "date": dt,
                    "source": a.get("domain", ""),
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "tone": float(a.get("tone", 0) or 0),
                })
        except Exception as e:
            print(f"GDELT query '{q}' failed: {e}")
        time.sleep(0.4)

    if not rows:
        return _load("news_sentiment")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"])
    df["sentiment_score"] = np.tanh(df["tone"] / 3.0)
    df["sentiment_label"] = np.where(
        df["sentiment_score"] > 0.1,
        "positive",
        np.where(df["sentiment_score"] < -0.1, "negative", "neutral"),
    )
    df.sort_values("date", inplace=True)
    _save(df, "news_sentiment")
    return df