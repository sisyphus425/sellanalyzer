"""Прогноз дивидендов: прогнозная чистая прибыль × payout ÷ число акций.

Payout берется из политики компании (целевое значение хранится в файле данных), из среднего фактического значения за последние годы
или задается вручную. Политика это ориентир: фактические выплаты зависят от решений совета директоров, долговой нагрузки и инвестиций.
"""

import pandas as pd

from salesanalyzer.config import DIVIDEND_AVG_YEARS

FORECAST_COLS = ["Год", "Дивиденд на акцию (низкий)", "Дивиденд на акцию (база)", "Дивиденд на акцию (высокий)",
                 "Дивиденды всего (база)", "Дивидендная доходность (база), %", "Рост к последнему факту, %"]
HISTORY_COLS = ["Год", "Дивиденд на акцию", "Чистая прибыль", "Фактический payout, %"]


def historical_payouts(ni_by_year, dps_by_year, shares_mn):
    """Фактический payout по годам: дивиденд на акцию × число акций / чистая прибыль.
    ni_by_year: {год: чистая прибыль в млрд}, dps_by_year: {год: дивиденд на акцию}, shares_mn: число акций, млн (текущее, поэтому для прошлых лет приближение)."""
    rows = []
    for year in sorted(set(ni_by_year) & set(dps_by_year)):
        ni, dps = ni_by_year[year], dps_by_year[year]
        payout = dps * shares_mn / 1000 / ni * 100 if (shares_mn and ni and ni > 0) else None
        rows.append({"Год": int(year), "Дивиденд на акцию": float(dps), "Чистая прибыль": float(ni), "Фактический payout, %": payout})
    return pd.DataFrame(rows, columns=HISTORY_COLS)


def average_payout(history, last_n=None):
    """Средний фактический payout за последние last_n лет с известным payout (без лет с убытком) или None."""
    last_n = DIVIDEND_AVG_YEARS if last_n is None else last_n
    known = history.dropna(subset=["Фактический payout, %"]).sort_values("Год").tail(last_n)
    return None if known.empty else float(known["Фактический payout, %"].mean())


def dividend_forecast(ni_scenarios, shares_mn, payout_pct, price=None, last_dps=None):
    """Дивиденд на акцию по сценариям прибыли. ni_scenarios: таблица «Год, Низкий, База, Высокий» (чистая прибыль, млрд).
    При убытке дивиденды не платятся. Возвращает таблицу FORECAST_COLS (пустую, если числа акций нет)."""
    if not shares_mn or ni_scenarios.empty:
        return pd.DataFrame(columns=FORECAST_COLS)
    rows = []
    for _, r in ni_scenarios.iterrows():
        dps = {name: max(float(r[name]), 0.0) * payout_pct / 100 * 1000 / shares_mn for name in ("Низкий", "База", "Высокий")}
        rows.append({"Год": int(r["Год"]), "Дивиденд на акцию (низкий)": dps["Низкий"], "Дивиденд на акцию (база)": dps["База"],
                     "Дивиденд на акцию (высокий)": dps["Высокий"],
                     "Дивиденды всего (база)": dps["База"] * shares_mn / 1000,
                     "Дивидендная доходность (база), %": dps["База"] / price * 100 if price else None,
                     "Рост к последнему факту, %": (dps["База"] / last_dps - 1) * 100 if (last_dps and last_dps > 0 and dps["База"] > 0) else None})
    return pd.DataFrame(rows, columns=FORECAST_COLS)


def policy_gap(history, policy_pct, last_n=None):
    """Насколько фактические выплаты отстают от политики: (средний фактический payout, политика, предупреждение или None)."""
    avg = average_payout(history, last_n)
    if policy_pct is None or avg is None:
        return avg, policy_pct, None
    if avg < 0.5 * policy_pct:
        return avg, policy_pct, (f"в последние годы фактический payout (в среднем около {avg:.0f}%) заметно ниже политики ({policy_pct:.0f}%): "
                                 "компания может платить меньше заявленного")
    return avg, policy_pct, None


def policy_from_rows(rows_long):
    """Целевой payout из файла данных: {pct, year, source, note, metric} для последнего года или None, если политики в файле нет."""
    if rows_long is None or rows_long.empty:
        return None
    sub = rows_long[rows_long["indicator"] == "payout_target"].dropna(subset=["value"]).sort_values("year")
    if sub.empty:
        return None
    r = sub.iloc[-1]
    note = r.get("note", "")
    return {"pct": float(r["value"]), "year": int(r["year"]), "source": str(r.get("source", "") or ""),
            "note": "" if pd.isna(note) else str(note), "metric": str(r.get("metric", "") or "")}


def payout_choices(policy_pct, avg_pct):
    """Варианты payout для выбора пользователем: {название: значение в %}. Порядок: политика, средний фактический."""
    out = {}
    if policy_pct is not None:
        out["По политике компании"] = float(policy_pct)
    if avg_pct is not None:
        out["Средний фактический"] = float(avg_pct)
    return out
