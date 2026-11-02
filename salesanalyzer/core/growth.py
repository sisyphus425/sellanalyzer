"""Рост к прошлому году, CAGR и «светофор» роста."""

import pandas as pd

from salesanalyzer.config import GROWTH_DOWN, GROWTH_UP


def signal_name(g) -> str:
    if g is None or pd.isna(g):
        return "н/д"
    if g >= GROWTH_UP:
        return "Рост"
    if g <= GROWTH_DOWN:
        return "Падение"
    return "Стагнация"


def signal_icon(g) -> str:
    return {"Рост": "🟢", "Стагнация": "🟡", "Падение": "🔴"}.get(signal_name(g), "⚪")


def add_growth(df: pd.DataFrame, col: str = "revenue") -> pd.DataFrame:
    """Колонка growth (% к предыдущему году). Только для соседних лет и положительных значений в обоих годах."""
    df = df.sort_values("year").reset_index(drop=True).copy()
    prev = df[col].shift(1)
    consecutive = df["year"].diff() == 1
    df["growth"] = ((df[col] - prev) / prev * 100).where(consecutive & (prev > 0) & (df[col] > 0))
    return df


def calc_cagr(df: pd.DataFrame, col: str = "revenue"):
    """Среднегодовой темп роста между первым и последним годом, % (или None)."""
    if len(df) < 2:
        return None
    first, last = df.iloc[0], df.iloc[-1]
    years = int(last["year"] - first["year"])
    if years < 1 or first[col] <= 0 or last[col] <= 0:
        return None
    return ((last[col] / first[col]) ** (1 / years) - 1) * 100
