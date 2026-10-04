# config.py — updated for Streamlit Cloud + local development

import os
import streamlit as st
from datetime import datetime, timedelta

def _get_secret(key, default=""):
    """
    Priority: st.secrets → os.environ → default
    Works on Streamlit Cloud (st.secrets), locally (.env via os.environ),
    and in CI/testing (environment variables).
    """
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

FRED_API_KEY = _get_secret("FRED_API_KEY")
TWITTER_BEARER_TOKEN = _get_secret("TWITTER_BEARER_TOKEN")
APIFY_API_TOKEN = _get_secret("APIFY_API_TOKEN")

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

HISTORY_YEARS = 10
REFRESH_INTERVAL_SECONDS = 60
DATA_REFRESH_MINUTES = 15

INFLUENTIAL_ACCOUNTS = [
    "realDonaldTrump", "federalreserve", "POTUS", "SecYellen",
    "ecb", "IMFNews", "WorldBank",
]

GOLD_KEYWORDS = [
    "gold", "tariff", "sanction", "inflation", "rate", "fed",
    "war", "strike", "iran", "china", "dollar", "recession",
]