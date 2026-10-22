"""Yahoo Finance (yfinance): отчетность, профиль, дивиденды и цены зарубежных компаний."""

import pandas as pd
import streamlit as st
import yfinance as yf

from salesanalyzer.config import CACHE_TTL, PRICE_HISTORY_YEARS
from salesanalyzer.data_sources.errors import DataSourceError
from salesanalyzer.data_sources.retry import call_with_retries

_NOT_FOUND_MARKERS = ("404", "not found", "no data found", "delisted")


def _looks_like_not_found(error) -> bool:
    """Тикер не существует: повторять запрос бессмысленно."""
    return any(marker in str(error).lower() for marker in _NOT_FOUND_MARKERS)


def _is_transient(error) -> bool:
    return not _looks_like_not_found(error)


def _describe(error, what: str) -> str:
    """Понятное пользователю описание ошибки обращения к Yahoo Finance."""
    text = (str(error) + " " + type(error).__name__).lower()
    if "too many requests" in text or "429" in text or "ratelimit" in text:
        return "Yahoo Finance временно ограничил число запросов. Подождите минуту и повторите."
    if any(m in text for m in ("connection", "resolve", "timed out", "timeout", "max retries")):
        return "Нет соединения с Yahoo Finance. Проверьте интернет-подключение."
    return f"Yahoo Finance не вернул данные ({what}). Повторите попытку позже."


def _fetch(what: str, func):
    """Вызов yfinance с повторными попытками; при постоянном сбое DataSourceError с понятным текстом."""
    try:
        return call_with_retries(func, is_transient=_is_transient)
    except Exception as error:
        raise DataSourceError(_describe(error, what), "Yahoo") from error


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
    try:
        info = _fetch(f"профиль {ticker}", lambda: company.info) or {}
    except DataSourceError as error:
        if _looks_like_not_found(error.__cause__):
            return None  # такого тикера нет
        raise
    name = info.get("longName")
    if not name:
        return None
    df, wide = None, pd.DataFrame()
    fin = _fetch(f"отчетность {ticker}", lambda: company.financials)
    if fin is not None and not fin.empty:
        wide = us_wide(fin)
        if "Total Revenue" in fin.index:
            rev = fin.loc["Total Revenue"].dropna()
            if not rev.empty:
                df = pd.DataFrame({"year": rev.index.year, "revenue": rev.values}).sort_values("year")
    div_df, div_ttm = pd.DataFrame(), None
    try:
        div = _fetch(f"дивиденды {ticker}", lambda: company.dividends)  # дивиденды необязательны
        if div is not None and not div.empty:
            div_df = div.groupby(div.index.year).sum().rename("dps").rename_axis("year").reset_index()
            div_ttm = float(div[div.index > pd.Timestamp.now(tz=div.index.tz) - pd.Timedelta(days=365)].sum())
    except Exception:
        div_df, div_ttm = pd.DataFrame(), None
    return {"name": name, "sector": info.get("sector") or "Не указан",
            "website": info.get("website"), "df": df, "wide": wide,
            "currency": info.get("currency") or "USD", "fin_currency": info.get("financialCurrency"),
            "market_cap": info.get("marketCap"), "shares": info.get("sharesOutstanding"),
            "dividends": div_df, "div_ttm": div_ttm}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_us_prices(ticker: str, years: int = PRICE_HISTORY_YEARS) -> pd.DataFrame:
    hist = _fetch(f"цены {ticker}", lambda: yf.Ticker(ticker).history(period=f"{years}y", interval="1d"))
    if hist is None or hist.empty:
        return pd.DataFrame()
    out = hist[["Close"]].reset_index()
    out.columns = ["begin", "close"]
    out["begin"] = pd.to_datetime(out["begin"]).dt.tz_localize(None)
    return out
