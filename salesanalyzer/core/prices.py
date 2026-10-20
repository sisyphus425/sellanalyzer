"""Метрики цены: доходность, волатильность, просадка, скользящие средние."""

import pandas as pd

from salesanalyzer.config import TRADING_DAYS


def last_close(prices: pd.DataFrame):
    """(последняя цена закрытия, дата) или (None, None)."""
    if prices is None or prices.empty:
        return None, None
    r = prices.sort_values("begin").iloc[-1]
    return float(r["close"]), r["begin"]


def add_price_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Сортирует цены и добавляет скользящие средние MA50 и MA200 (по торговым дням)."""
    p = prices[["begin", "close"]].sort_values("begin").reset_index(drop=True).copy()
    p["ma50"] = p["close"].rolling(50).mean()
    p["ma200"] = p["close"].rolling(200).mean()
    return p


def price_metrics(p: pd.DataFrame):
    """Метрики по выбранному периоду (таблица после add_price_indicators). None, если данных мало."""
    if len(p) < 2 or p["close"].iloc[0] <= 0:
        return None
    close = p["close"]
    start, end = close.iloc[0], close.iloc[-1]
    days = (p["begin"].iloc[-1] - p["begin"].iloc[0]).days
    rets = close.pct_change().dropna()
    dd = (close / close.cummax() - 1) * 100
    i_min = int(dd.values.argmin())
    i_peak = int(close.iloc[: i_min + 1].values.argmax())
    ma50, ma200 = p["ma50"].iloc[-1], p["ma200"].iloc[-1]
    return {
        "last": end,
        "total": (end / start - 1) * 100,
        "annual": ((end / start) ** (365 / days) - 1) * 100 if days >= 90 else None,
        "vol": rets.std() * (TRADING_DAYS ** 0.5) * 100 if len(rets) >= 20 else None,
        "max_dd": dd.min(), "dd_peak": p["begin"].iloc[i_peak], "dd_trough": p["begin"].iloc[i_min],
        "cur_dd": dd.iloc[-1],
        "ma50": None if pd.isna(ma50) else ma50,
        "ma200": None if pd.isna(ma200) else ma200,
    }


def trend_label(m: dict) -> str:
    if m["ma200"] is None:
        return "н/д"
    return "🟢 выше MA200" if m["last"] > m["ma200"] else "🔴 ниже MA200"


def price_metrics_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица метрик по нескольким бумагам (колонки prices: begin, label, close)."""
    rows = []
    for label, g in prices.groupby("label", sort=False):
        m = price_metrics(add_price_indicators(g))
        if m is None:
            continue
        rows.append({
            "Компания": label,
            "Доходность за период, %": round(m["total"], 1),
            "Годовая доходность, %": None if m["annual"] is None else round(m["annual"], 1),
            "Волатильность (год), %": None if m["vol"] is None else round(m["vol"], 1),
            "Макс. просадка, %": round(m["max_dd"], 1),
            "Текущая просадка, %": round(m["cur_dd"], 1),
            "Тренд": trend_label(m),
        })
    return pd.DataFrame(rows)
