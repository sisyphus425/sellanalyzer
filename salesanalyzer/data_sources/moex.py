"""Мосбиржа (MOEX ISS API): название бумаги и история цен."""

from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

from salesanalyzer.config import CACHE_TTL, ISS


def iss_get(path: str, **params) -> dict:
    r = requests.get(f"{ISS}/{path}.json", params={"iss.meta": "off", **params}, timeout=15)
    r.raise_for_status()
    return {k: pd.DataFrame(v["data"], columns=v["columns"])
            for k, v in r.json().items() if isinstance(v, dict) and "columns" in v}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_moex_name(ticker: str):
    """Полное название бумаги на Мосбирже или None, если тикер неизвестен."""
    desc = iss_get(f"securities/{ticker}", **{"iss.only": "description"}).get("description")
    if desc is None or desc.empty:
        return None
    return dict(zip(desc["name"], desc["value"])).get("NAME", ticker)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_moex_candles(ticker: str, years: int = 3) -> pd.DataFrame:
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    frames, cursor = [], 0
    while True:  # ISS отдает максимум 500 свечей за запрос
        df = iss_get(f"engines/stock/markets/shares/securities/{ticker}/candles",
                     interval=24, **{"from": start, "start": cursor}).get("candles")
        if df is None or df.empty:
            break
        frames.append(df)
        cursor += len(df)
        if len(df) < 500:
            break
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["begin"] = pd.to_datetime(out["begin"])
    return out
