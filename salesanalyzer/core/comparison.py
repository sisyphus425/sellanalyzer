"""Расчеты для сравнения компаний: индекс выручки, сводки роста и CAGR, разбор тикеров."""

import re

import pandas as pd

from salesanalyzer.core.growth import calc_cagr, signal_icon


def parse_tickers(text: str) -> list:
    return list(dict.fromkeys(t.strip().upper() for t in re.split(r"[,\s;]+", text) if t.strip()))


def index_revenue(rev: pd.DataFrame, base_year=None) -> pd.DataFrame:
    """Выручка в индексах (базовый год = 100). base_year=None: у каждой компании свой первый год."""
    parts = []
    for _, g in rev.groupby("label", sort=False):
        g = g.sort_values("year")
        if base_year is not None:
            g = g[g["year"] >= base_year]
        if g.empty or g["revenue"].iloc[0] <= 0:
            continue
        if base_year is not None and g["year"].iloc[0] != base_year:
            continue
        parts.append(g.assign(index=g["revenue"] / g["revenue"].iloc[0] * 100))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summary_table(rev: pd.DataFrame, common_years: list) -> pd.DataFrame:
    c_from, c_to = (min(common_years), max(common_years)) if len(common_years) >= 2 else (None, None)
    rows = []
    for label, g in rev.groupby("label", sort=False):
        g = g.sort_values("year")
        last = g.iloc[-1]
        cagr = calc_cagr(g)
        cagr_common = None
        if c_from is not None:
            cagr_common = calc_cagr(g[(g["year"] >= c_from) & (g["year"] <= c_to)])
        rows.append({
            "Компания": label,
            "Период": f"{int(g['year'].iloc[0])}–{int(last['year'])}",
            "Рост к пред. году": "н/д" if pd.isna(last["growth"]) else f"{signal_icon(last['growth'])} {last['growth']:+.1f}%",
            "CAGR (весь ряд)": "н/д" if cagr is None else f"{signal_icon(cagr)} {cagr:.1f}%",
            f"CAGR ({c_from}–{c_to})" if c_from is not None else "CAGR (общий период)":
                "н/д" if cagr_common is None else f"{signal_icon(cagr_common)} {cagr_common:.1f}%",
            "Есть оценки": "да" if g["is_estimate"].any() else "нет",
        })
    return pd.DataFrame(rows).set_index("Компания")


def summary_numeric(rev: pd.DataFrame, common_years: list) -> pd.DataFrame:
    """Числовая версия сводки роста и CAGR для выгрузки."""
    c_from, c_to = (min(common_years), max(common_years)) if len(common_years) >= 2 else (None, None)
    rows = []
    for label, g in rev.groupby("label", sort=False):
        g = g.sort_values("year")
        last = g.iloc[-1]
        cagr = calc_cagr(g)
        cagr_common = calc_cagr(g[(g["year"] >= c_from) & (g["year"] <= c_to)]) if c_from is not None else None
        rows.append({
            "Компания": label,
            "Период данных": f"{int(g['year'].iloc[0])}–{int(last['year'])}",
            "Рост к пред. году, %": None if pd.isna(last["growth"]) else round(last["growth"], 2),
            "CAGR весь ряд, %": None if cagr is None else round(cagr, 2),
            "CAGR общий период, %": None if cagr_common is None else round(cagr_common, 2),
            "Общий период": f"{c_from}–{c_to}" if c_from is not None else "",
            "Единицы": last["unit"],
            "Есть оценки": "да" if g["is_estimate"].any() else "нет",
        })
    return pd.DataFrame(rows)
