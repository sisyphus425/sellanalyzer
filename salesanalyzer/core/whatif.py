"""Сценарный калькулятор «что если».

Пользователь задает допущения по годам (рост выручки, чистая маржа, изменение числа акций, payout), а калькулятор
считает чистую прибыль, прибыль на акцию (EPS), P/E, дивиденд и цену при неизменном P/E.
Это арифметика допущений, а не прогноз: калькулятор ничего не предсказывает.
"""

import numpy as np
import pandas as pd

from salesanalyzer.config import FORECAST_HORIZONS, UNIT_TO_BN
from salesanalyzer.core.forecast import forecast_series
from salesanalyzer.core.profitability import latest_value

PRESETS = ("Как в базовом прогнозе", "Рост как в прошлом году", "Стагнация", "Улучшение маржи (+2 п.п.)",
           "Ухудшение (рост −5 п.п., маржа −2 п.п.)")
SCENARIO_COLS = ["Год", "Выручка", "Чистая прибыль", "Чистая маржа, %", "Число акций, млн", "EPS", "Дивиденд на акцию",
                 "Дивиденды всего", "Payout, %", "P/E", "Дивидендная доходность, %", "Цена при неизменном P/E",
                 "Рост выручки, %", "Рост прибыли, %", "Рост EPS, %"]
SENSITIVITY_METRICS = {"P/E": "P/E", "EPS": "EPS", "Чистая прибыль": "Чистая прибыль", "Цена при неизменном P/E": "Цена при неизменном P/E"}
# допустимые границы допущений, %; ими же ограничены поля ввода на экране
LIMITS = {"growth": (-100.0, 500.0), "margin": (-100.0, 100.0), "shares_change": (-50.0, 100.0), "payout": (0.0, 300.0)}
GROWTH_DELTAS = (-10.0, -5.0, 0.0, 5.0, 10.0)
MARGIN_DELTAS = (-3.0, -1.5, 0.0, 1.5, 3.0)


def _growth(prev, cur):
    """Рост в % между положительными значениями, иначе None."""
    if prev is None or cur is None or prev <= 0 or cur <= 0:
        return None
    return (cur / prev - 1) * 100


def build_base(w: pd.DataFrame, unit: str, price=None, currency="₽", shares_mn=None, dps=None, dps_by_year=None):
    """Базовый год и исходные значения из таблицы по годам (колонки year, revenue, net_income и необязательно shares, dps).
    Денежные показатели переводятся в млрд валюты. dps_by_year: {год: дивиденд на акцию}. Возвращает (база или None, список замечаний)."""
    notes = []
    k = UNIT_TO_BN.get(unit)
    if k is None:
        return None, [f"Единицы «{unit}» не поддерживаются калькулятором."]
    ok = w.dropna(subset=["revenue", "net_income"])
    ok = ok[ok["revenue"] > 0]
    if ok.empty:
        return None, ["Для калькулятора нужны выручка и чистая прибыль хотя бы за один год."]
    row = ok.iloc[-1]
    year = int(row["year"])
    revenue, net_income = float(row["revenue"]) * k, float(row["net_income"]) * k
    prev = w[(w["year"] == year - 1) & (w["revenue"] > 0)]
    last_growth = _growth(float(prev["revenue"].iloc[0]) * k, revenue) if not prev.empty else None

    if shares_mn is None and "shares" in w.columns:
        lv = latest_value(w, "shares")
        shares_mn = lv[1] if lv else None
    if dps is None and dps_by_year:
        dps = dps_by_year.get(year)  # например, сумма выплат за календарный год (зарубежные компании)
    if dps is None and "dps" in w.columns:
        dps_row = w.dropna(subset=["dps"])
        dps_row = dps_row[dps_row["year"] == year]
        dps = float(dps_row["dps"].iloc[-1]) if not dps_row.empty else None
    if not shares_mn:
        notes.append("Число акций неизвестно: прибыль на акцию, P/E и дивиденд на акцию не считаются.")
        shares_mn = None
    eps = net_income * 1000 / shares_mn if shares_mn else None
    payout = dps * shares_mn / 1000 / net_income * 100 if (dps is not None and shares_mn and net_income > 0) else None
    pe = price / eps if (price and eps and eps > 0) else None
    base = {"year": year, "revenue": revenue, "net_income": net_income, "margin": net_income / revenue * 100,
            "shares_mn": shares_mn, "eps": eps, "dps": dps, "payout": payout, "price": price or None, "pe": pe,
            "currency": currency, "last_growth": last_growth}
    if not price:
        notes.append("Цена акции неизвестна: P/E, дивидендная доходность и цена при неизменном P/E не считаются (её можно ввести вручную).")
    return base, notes


def with_price(base, price):
    """Копия базы с другой ценой акции (например, введенной вручную); P/E пересчитывается."""
    price = price or None
    pe = price / base["eps"] if (price and base["eps"] and base["eps"] > 0) else None
    return {**base, "price": price, "pe": pe}


def default_growths(years, values, horizon):
    """Темпы роста выручки по годам из базового прогноза (медиана методов); дальше последний темп повторяется. None, если прогноза нет."""
    res = forecast_series(years, values, horizon=min(horizon, max(FORECAST_HORIZONS)), nonnegative=True)
    sc = res["scenarios"]
    if sc.empty:
        return None
    prev, out = float(res["values"][-1]), []
    for b in sc["База"]:
        out.append((float(b) / prev - 1) * 100)
        prev = float(b)
    while len(out) < horizon:
        out.append(out[-1])
    return out[:horizon]


def growth_source_reliability(years, values):
    """Надежность базового прогноза выручки, из которого берутся начальные темпы роста: 'низкая', 'средняя' или 'недостаточно данных'."""
    return forecast_series(years, values, horizon=1, nonnegative=True)["reliability"]["label"]


def clamp_params(params):
    """Приводит допущения к допустимым границам LIMITS (например, payout 484% становится 300%)."""
    return [{k: min(max(float(v), LIMITS[k][0]), LIMITS[k][1]) for k, v in p.items()} for p in params]


def make_params(base, horizon, preset=PRESETS[0], growths=None):
    """Готовые допущения по годам для выбранного набора. Каждый год: growth, margin, shares_change, payout (все в %)."""
    g_forecast = list(growths) if growths else [base["last_growth"] if base["last_growth"] is not None else 0.0] * horizon
    g_forecast = (g_forecast + [g_forecast[-1]] * horizon)[:horizon]
    margin0, payout0 = base["margin"], (base["payout"] if base["payout"] is not None else 0.0)
    last_g = base["last_growth"] if base["last_growth"] is not None else 0.0
    out = []
    for i in range(horizon):
        g, m = g_forecast[i], margin0
        if preset == "Рост как в прошлом году":
            g = last_g
        elif preset == "Стагнация":
            g = 0.0
        elif preset == "Улучшение маржи (+2 п.п.)":
            m = margin0 + 2.0
        elif preset == "Ухудшение (рост −5 п.п., маржа −2 п.п.)":
            g, m = g - 5.0, margin0 - 2.0
        out.append({"growth": float(g), "margin": float(m), "shares_change": 0.0, "payout": float(payout0)})
    return clamp_params(out)


def scenario(base, params):
    """Расчет сценария: строка «Факт» базового года и по строке на каждый год допущений."""
    if not params:
        raise ValueError("Нужно хотя бы одно допущение (один год).")
    price, shares0 = base["price"], base["shares_mn"]
    rows = [{"Год": base["year"], "Выручка": base["revenue"], "Чистая прибыль": base["net_income"], "Чистая маржа, %": base["margin"],
             "Число акций, млн": shares0, "EPS": base["eps"], "Дивиденд на акцию": base["dps"],
             "Дивиденды всего": (base["dps"] * shares0 / 1000) if (base["dps"] is not None and shares0) else None,
             "Payout, %": base["payout"], "P/E": base["pe"],
             "Дивидендная доходность, %": (base["dps"] / price * 100) if (base["dps"] is not None and price) else None,
             "Цена при неизменном P/E": price, "Рост выручки, %": base["last_growth"], "Рост прибыли, %": None, "Рост EPS, %": None}]
    rev, shares, prev = base["revenue"], shares0, rows[0]
    for i, p in enumerate(params, start=1):
        rev = rev * (1 + p["growth"] / 100)
        ni = rev * p["margin"] / 100
        shares = shares * (1 + p["shares_change"] / 100) if shares else None
        eps = ni * 1000 / shares if shares else None
        dividends = max(ni, 0.0) * p["payout"] / 100  # дивиденды платятся только из положительной прибыли
        dps = dividends * 1000 / shares if shares else None
        pe = price / eps if (price and eps and eps > 0) else None
        implied = base["pe"] * eps if (base["pe"] and eps and eps > 0) else None
        row = {"Год": base["year"] + i, "Выручка": rev, "Чистая прибыль": ni, "Чистая маржа, %": p["margin"], "Число акций, млн": shares,
               "EPS": eps, "Дивиденд на акцию": dps, "Дивиденды всего": dividends, "Payout, %": p["payout"], "P/E": pe,
               "Дивидендная доходность, %": (dps / price * 100) if (dps is not None and price) else None,
               "Цена при неизменном P/E": implied, "Рост выручки, %": p["growth"],
               "Рост прибыли, %": _growth(prev["Чистая прибыль"], ni), "Рост EPS, %": _growth(prev["EPS"], eps)}
        rows.append(row)
        prev = row
    return pd.DataFrame(rows, columns=SCENARIO_COLS)


def _label(delta):
    return "0 п.п." if delta == 0 else f"{delta:+g} п.п.".replace("-", "−")


def sensitivity(base, params, metric="P/E", growth_deltas=GROWTH_DELTAS, margin_deltas=MARGIN_DELTAS):
    """Чувствительность показателя на конец горизонта к росту выручки (во все годы) и марже (во все годы), п.п. к сценарию.
    Строки: изменение маржи, столбцы: изменение роста."""
    col = SENSITIVITY_METRICS[metric]
    grid = []
    for dm in margin_deltas:
        row = []
        for dg in growth_deltas:
            shifted = [{**p, "growth": p["growth"] + dg, "margin": p["margin"] + dm} for p in params]
            row.append(scenario(base, shifted)[col].iloc[-1])
        grid.append(row)
    out = pd.DataFrame(grid, index=[f"Маржа {_label(d)}" for d in margin_deltas], columns=[f"Рост {_label(d)}" for d in growth_deltas],
                       dtype=float)
    return out
