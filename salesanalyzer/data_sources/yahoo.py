"""Yahoo Finance (yfinance): отчетность, профиль, дивиденды и цены зарубежных компаний."""

import pandas as pd
import streamlit as st
import yfinance as yf

from salesanalyzer.config import CACHE_TTL


def us_wide(fin: pd.DataFrame) -> pd.DataFrame:
    """Годовая отчетность yfinance -> таблица по годам: revenue, net_income, ebitda, roe (NaN, если нет)."""
    rows = {"revenue": "Total Revenue", "net_income": "Net Income", "ebitda": "EBITDA"}
    data = {}
    for key, label in rows.items():
        if label in fin.index:
            ser = fin.loc[label].dropna()
            if not ser.empty:
                ser = pd.Series(ser.values, index=ser.index.year)
                data[key] = ser[~ser.index.duplicated(keep="last")]
    if not data:
        return pd.DataFrame()
    w = pd.DataFrame(data)
    w.index.name = "year"
    w = w.reset_index().sort_values("year").reset_index(drop=True)
    for c in ("revenue", "net_income", "ebitda", "roe"):
        if c not in w.columns:
            w[c] = float("nan")
    w["has_estimate"] = False
    return w


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_us_data(ticker: str):
    company = yf.Ticker(ticker)
    info = company.info or {}
    name = info.get("longName")
    if not name:
        return None
    df, wide = None, pd.DataFrame()
    fin = company.financials
    if fin is not None and not fin.empty:
        wide = us_wide(fin)
        if "Total Revenue" in fin.index:
            rev = fin.loc["Total Revenue"].dropna()
            if not rev.empty:
                df = pd.DataFrame({"year": rev.index.year, "revenue": rev.values}).sort_values("year")
    div_df, div_ttm = pd.DataFrame(), None
    try:
        div = company.dividends
        if div is not None and not div.empty:
            div_df = div.groupby(div.index.year).sum().rename("dps").rename_axis("year").reset_index()
            div_ttm = float(div[div.index > pd.Timestamp.now(tz=div.index.tz) - pd.Timedelta(days=365)].sum())
    except Exception:
        pass
    return {"name": name, "sector": info.get("sector") or "Не указан",
            "website": info.get("website"), "df": df, "wide": wide,
            "currency": info.get("currency") or "USD", "fin_currency": info.get("financialCurrency"),
            "market_cap": info.get("marketCap"), "shares": info.get("sharesOutstanding"),
            "dividends": div_df, "div_ttm": div_ttm}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_us_prices(ticker: str, years: int = 3) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=f"{years}y", interval="1d")
    if hist is None or hist.empty:
        return pd.DataFrame()
    out = hist[["Close"]].reset_index()
    out.columns = ["begin", "close"]
    out["begin"] = pd.to_datetime(out["begin"]).dt.tz_localize(None)
    return out
