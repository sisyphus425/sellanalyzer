"""Сбор данных для сравнения компаний: объединяет CSV, Мосбиржу и Yahoo Finance."""

import pandas as pd
import requests

from salesanalyzer import config
from salesanalyzer.core.growth import add_growth
from salesanalyzer.core.profitability import add_margins, make_wide
from salesanalyzer.data_sources.moex import get_moex_candles
from salesanalyzer.data_sources.yahoo import get_us_data, get_us_prices


def collect_revenue(ru_tickers, us_tickers, revenue_all, problems) -> pd.DataFrame:
    """Единая таблица выручки: label, year, revenue, unit, is_estimate, growth."""
    cols = ["label", "year", "revenue", "unit", "is_estimate"]
    frames = []
    for t in ru_tickers:
        rows = revenue_all[revenue_all["ticker"] == t]
        if rows.empty:
            problems.append(f"{t}: нет данных в {config.CSV_PATH.name}")
            continue
        frames.append(rows.assign(label=t)[cols])
    for t in us_tickers:
        try:
            data = get_us_data(t)
        except Exception as e:
            problems.append(f"{t}: ошибка загрузки ({e})")
            continue
        if data is None:
            problems.append(f"{t}: тикер не найден")
        elif data["df"] is None:
            problems.append(f"{t}: нет открытых данных по выручке")
        else:
            frames.append(data["df"].assign(label=t, unit="$", is_estimate=False)[cols])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return pd.concat([add_growth(g) for _, g in out.groupby("label", sort=False)], ignore_index=True)


def collect_profit(ru_tickers, us_tickers, ru_all) -> pd.DataFrame:
    """Единая таблица прибыльности: label, year, net_income, ebitda, roe, net_margin, ebitda_margin."""
    parts = []
    for t in ru_tickers:
        w, unit, mixed = make_wide(ru_all[ru_all["ticker"] == t] if not ru_all.empty else ru_all)
        if not w.empty:
            parts.append(add_margins(w, enabled=not mixed).assign(label=t, unit=unit))
    for t in us_tickers:
        try:
            data = get_us_data(t)
        except Exception:
            continue  # ошибка уже показана на вкладках выручки
        if data is not None and not data["wide"].empty:
            parts.append(add_margins(data["wide"]).assign(label=t, unit="$"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_prices(ru_tickers, us_tickers, years, problems) -> pd.DataFrame:
    """Цены закрытия, приведенные к 100 на общую дату старта. Колонки: begin, label, close, index."""
    series = {}
    for t in ru_tickers:
        try:
            df = get_moex_candles(t, years)
        except requests.RequestException as e:
            problems.append(f"{t}: ошибка MOEX ({e})")
            continue
        if df.empty:
            problems.append(f"{t}: нет истории цен")
        else:
            series[t] = df[["begin", "close"]]
    for t in us_tickers:
        try:
            df = get_us_prices(t, years)
        except Exception as e:
            problems.append(f"{t}: ошибка загрузки цен ({e})")
            continue
        if df.empty:
            problems.append(f"{t}: нет истории цен")
        else:
            series[t] = df[["begin", "close"]]
    if len(series) == 0:
        return pd.DataFrame()
    start = max(df["begin"].min() for df in series.values())
    parts = []
    for label, df in series.items():
        df = df[df["begin"] >= start].sort_values("begin")
        if df.empty or df["close"].iloc[0] <= 0:
            continue
        parts.append(df.assign(label=label, index=df["close"] / df["close"].iloc[0] * 100))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
