"""Таблица показателей по годам, маржа и вспомогательные функции прибыльности."""

import pandas as pd

from salesanalyzer.config import MONEY_INDICATORS


def make_wide(rows: pd.DataFrame):
    """Длинная таблица (year, indicator, value, unit, is_estimate) -> по годам. Возвращает (wide, unit, mixed_units)."""
    if rows.empty:
        return pd.DataFrame(), "", False
    p = rows.pivot_table(index="year", columns="indicator", values="value", aggfunc="last")
    for c in ("revenue", "net_income", "ebitda", "roe", "dps", "shares"):
        if c not in p.columns:
            p[c] = float("nan")
    p["has_estimate"] = rows.groupby("year")["is_estimate"].any()
    units = rows.loc[rows["indicator"].isin(MONEY_INDICATORS), "unit"].unique()
    unit = units[-1] if len(units) else ""
    return p.reset_index(), unit, len(units) > 1


def add_margins(w: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    """Чистая маржа и EBITDA-маржа в % от выручки."""
    w = w.sort_values("year").reset_index(drop=True).copy()
    if enabled:
        base = w["revenue"].where(w["revenue"] > 0)
        w["net_margin"] = w["net_income"] / base * 100
        w["ebitda_margin"] = w["ebitda"] / base * 100
    else:
        w["net_margin"] = float("nan")
        w["ebitda_margin"] = float("nan")
    return w


def latest_value(w: pd.DataFrame, col: str):
    s_ = w.dropna(subset=[col])
    return None if s_.empty else (int(s_["year"].iloc[-1]), float(s_[col].iloc[-1]))
