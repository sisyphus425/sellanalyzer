"""Чтение файлов данных: financials_ru.csv (показатели компаний РФ) и macro_ru.csv (курс ЦБ, инфляция)."""

import pandas as pd
import streamlit as st

from salesanalyzer import config
from salesanalyzer.config import MACRO_COLS, REQUIRED_COLS


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    """mtime в аргументах: при изменении файла кэш сбрасывается сам."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": str})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("в файле нет колонок: " + ", ".join(missing))
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["indicator"] = df["indicator"].astype(str).str.strip().str.lower()
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["is_estimate"] = df["is_estimate"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "да"])
    for col in ("unit", "metric", "source", "note"):
        df[col] = df[col].fillna("").astype(str)
    return df.dropna(subset=["value"])


def load_ru_all() -> pd.DataFrame:
    """Все показатели РФ в длинном формате (ticker, year, indicator, value, ...)."""
    if not config.CSV_PATH.exists():
        return pd.DataFrame()
    try:
        return _read_csv(str(config.CSV_PATH), config.CSV_PATH.stat().st_mtime)
    except ValueError as e:
        st.error(f"Ошибка в файле {config.CSV_PATH.name}: {e}")
        return pd.DataFrame()


def load_revenue_csv() -> pd.DataFrame:
    """Только выручка; значение лежит в колонке revenue."""
    df = load_ru_all()
    if df.empty:
        return df
    rev = df[df["indicator"] == "revenue"].rename(columns={"value": "revenue"})
    return rev.drop(columns=["indicator"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _read_macro(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in MACRO_COLS if c not in df.columns]
    if missing:
        raise ValueError("в файле нет колонок: " + ", ".join(missing))
    df["year"] = df["year"].astype(int)
    df["indicator"] = df["indicator"].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def load_macro() -> dict:
    """{'usd_rub': {год: курс}, 'cpi': {год: инфляция дек/дек, %}}. Пустые словари, если файла нет."""
    out = {"usd_rub": {}, "cpi": {}}
    if not config.MACRO_PATH.exists():
        return out
    try:
        df = _read_macro(str(config.MACRO_PATH), config.MACRO_PATH.stat().st_mtime)
    except ValueError as e:
        st.error(f"Ошибка в файле {config.MACRO_PATH.name}: {e}")
        return out
    for key in out:
        sub = df[df["indicator"] == key]
        out[key] = dict(zip(sub["year"].astype(int), sub["value"].astype(float)))
    return out


def read_raw_csv() -> pd.DataFrame:
    """Сырой файл без преобразований: все значения как текст."""
    return pd.read_csv(config.CSV_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
