"""Полная доходность (цена + реинвестированные дивиденды) и сравнение бумаги с индексом рынка.

Полная доходность: в день отсечки (первый день торгов без дивиденда) дивиденд считается полученным и сразу
вложенным в ту же бумагу по цене закрытия. Индекс полной доходности: TR_t = TR_(t−1) × (P_t + D_t) / P_(t−1).
Налог на дивиденды не учитывается.

Сравнение с индексом: доходности считаются только по общим торговым дням. Бета = cov(r_бумаги, r_индекса) / var(r_индекса)
по дневным доходностям, корреляция по ним же. Разница доходностей = доходность бумаги − доходность индекса за общий период, п.п.
"""

import numpy as np
import pandas as pd

from salesanalyzer.config import BENCHMARK_MIN_DAYS, TRADING_DAYS

EVENT_COLS = ["ex_date", "dps"]


def _sorted_prices(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices[["begin", "close"]].dropna().sort_values("begin").reset_index(drop=True).copy()
    p["begin"] = pd.to_datetime(p["begin"])
    return p


def book_dividends(prices: pd.DataFrame, dividends) -> pd.DataFrame:
    """Дивиденды, разнесенные по торговым дням: каждая выплата попадает на первый торговый день не раньше даты отсечки.
    Учитываются только выплаты внутри периода (после первого дня): в первый день реинвестировать еще нечего.
    Возвращает таблицу ex_date, dps, booked (дата в ценах); пустую, если выплат нет."""
    out = pd.DataFrame(columns=EVENT_COLS + ["booked"])
    if dividends is None or len(dividends) == 0 or prices is None or prices.empty:
        return out
    p = _sorted_prices(prices)
    ev = dividends[EVENT_COLS].dropna().copy()
    ev["ex_date"] = pd.to_datetime(ev["ex_date"]).dt.normalize()
    ev = ev[ev["dps"] > 0].sort_values("ex_date")
    days = p["begin"].dt.normalize()
    pos = np.searchsorted(days.values, ev["ex_date"].values, side="left")
    keep = (pos > 0) & (pos < len(p))
    ev = ev[keep].copy()
    ev["booked"] = p["begin"].values[pos[keep]]
    return ev.reset_index(drop=True)


def add_total_return(prices: pd.DataFrame, dividends=None) -> pd.DataFrame:
    """Колонки begin, close, div (дивиденд, учтенный в этот день), tr (индекс полной доходности, на старте равен цене)."""
    p = _sorted_prices(prices)
    p["div"] = 0.0
    booked = book_dividends(p, dividends)
    if not booked.empty:
        per_day = booked.groupby("booked")["dps"].sum()
        p["div"] = p["begin"].map(per_day).fillna(0.0).astype(float)
    if p.empty:
        p["tr"] = pd.Series(dtype=float)
        return p
    ratio = (p["close"] + p["div"]) / p["close"].shift(1)
    p["tr"] = p["close"].iloc[0] * ratio.fillna(1.0).cumprod()
    return p


def _annualize(total_pct, days):
    return ((1 + total_pct / 100) ** (365 / days) - 1) * 100 if days >= 90 and total_pct > -100 else None


def total_return_metrics(p: pd.DataFrame):
    """Метрики по таблице add_total_return: доходность по цене, полная доходность, их разница (вклад дивидендов),
    годовая полная доходность, число и сумма учтенных выплат. None, если данных мало."""
    if len(p) < 2 or p["close"].iloc[0] <= 0:
        return None
    days = (p["begin"].iloc[-1] - p["begin"].iloc[0]).days
    price_total = (p["close"].iloc[-1] / p["close"].iloc[0] - 1) * 100
    tr_total = (p["tr"].iloc[-1] / p["tr"].iloc[0] - 1) * 100
    n = int((p["div"] > 0).sum())
    return {"price_total": price_total, "tr_total": tr_total, "div_effect": tr_total - price_total,
            "tr_annual": _annualize(tr_total, days), "price_annual": _annualize(price_total, days),
            "n_dividends": n, "dividends_sum": float(p["div"].sum())}


def compare_with_benchmark(stock: pd.DataFrame, bench: pd.DataFrame, min_days=None):
    """Сравнение бумаги с индексом по общим торговым дням. stock и bench: колонки begin и close (для полной доходности
    в close можно передать tr). Возвращает (метрики, таблица begin, stock, bench, relative) или (None, пустая таблица),
    если общих дней меньше min_days. relative = индекс бумаги / индекс рынка × 100 (растет, когда бумага обгоняет рынок)."""
    min_days = BENCHMARK_MIN_DAYS if min_days is None else min_days
    empty = pd.DataFrame(columns=["begin", "stock", "bench", "relative"])
    if stock is None or bench is None or stock.empty or bench.empty:
        return None, empty
    s, b = _sorted_prices(stock), _sorted_prices(bench)
    s["day"], b["day"] = s["begin"].dt.normalize(), b["begin"].dt.normalize()
    m = (s.drop_duplicates("day", keep="last")[["day", "close"]]
         .merge(b.drop_duplicates("day", keep="last")[["day", "close"]], on="day", suffixes=("_s", "_b")))
    m = m[(m["close_s"] > 0) & (m["close_b"] > 0)].reset_index(drop=True)
    if len(m) < max(min_days, 3):
        return None, empty
    rs, rb = m["close_s"].pct_change().dropna(), m["close_b"].pct_change().dropna()
    var_b = rb.var()
    beta = float(rs.cov(rb) / var_b) if var_b > 0 else None
    corr = float(rs.corr(rb)) if rs.std() > 0 and rb.std() > 0 else None
    stock_total = (m["close_s"].iloc[-1] / m["close_s"].iloc[0] - 1) * 100
    bench_total = (m["close_b"].iloc[-1] / m["close_b"].iloc[0] - 1) * 100
    days = (m["day"].iloc[-1] - m["day"].iloc[0]).days
    track = float((rs - rb).std() * TRADING_DAYS ** 0.5 * 100) if len(rs) >= 20 else None
    metrics = {"stock_total": stock_total, "bench_total": bench_total, "excess": stock_total - bench_total,
               "stock_annual": _annualize(stock_total, days), "bench_annual": _annualize(bench_total, days),
               "beta": beta, "corr": corr, "tracking_error": track, "days": len(m),
               "start": m["day"].iloc[0], "end": m["day"].iloc[-1]}
    frame = pd.DataFrame({"begin": m["day"], "stock": m["close_s"] / m["close_s"].iloc[0] * 100,
                          "bench": m["close_b"] / m["close_b"].iloc[0] * 100})
    frame["relative"] = frame["stock"] / frame["bench"] * 100
    return metrics, frame


def beta_label(beta) -> str:
    """Короткое пояснение к бете."""
    if beta is None or pd.isna(beta):
        return "н/д"
    if beta > 1.2:
        return "сильнее рынка"
    if beta < 0.8:
        return "слабее рынка"
    return "примерно как рынок"
