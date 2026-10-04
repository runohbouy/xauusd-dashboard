"""
config.py — centralized configuration and secret loading.

Works locally via .streamlit/secrets.toml or environment variables,
and on Streamlit Cloud via st.secrets.
"""

import os
import streamlit as st
from datetime import datetime, timedelta


def _get_secret(key: str, default: str = "") -> str:
    """
    Priority order:
      1. st.secrets (Streamlit Cloud + local .streamlit/secrets.toml)
      2. os.environ (local .env or shell environment)
      3. provided default
    """
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, Exception):
        return os.environ.get(key, default)


# API credentials
FRED_API_KEY = _get_secret("FRED_API_KEY")
APIFY_API_TOKEN = _get_secret("APIFY_API_TOKEN")

# Cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Data settings
HISTORY_YEARS = 10
REFRESH_INTERVAL_SECONDS = 60
DATA_REFRESH_MINUTES = 15

# News sentiment keywords (used by GDELT fetcher)
NEWS_QUERIES = [
    "gold price",
    "XAUUSD",
    "gold market",
    "gold forecast",
    "fed rate gold",
]