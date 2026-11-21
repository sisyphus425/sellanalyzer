"""Исторические мультипликаторы по годам и сравнение доходности акции с доходностью ОФЗ.

Мультипликаторы прошлых лет считаются по цене на конец календарного года и текущему числу акций. Цены с биржи уже
приведены к дроблениям (история пересчитана на новое число акций), а число акций в файле данных указано после дробления,
поэтому капитализация прошлых лет не искажается дроблениями (T 10:1, GMKN 1:100). Выкупы и допэмиссии не учитываются:
для прошлых лет это приближение.

Сравнение с ОФЗ: дивидендная доходность и доходность прибыли (E/P = 1 / P/E) сопоставляются с доходностью ОФЗ
по кривой бескупонной доходности Мосбиржи (G-кривая) на тот же срок. Спред = доходность акции − доходность ОФЗ, п.п.
"""

import numpy as np
import pandas as pd

from salesanalyzer.config import UNIT_TO_BN
from salesanalyzer.core.adjustments import adjust_long
from salesanalyzer.core.profitability import latest_value, make_wide

HISTORY_COLS = ["Год", "Дата цены", "Цена", "Капитализация, млрд", "Выручка, млрд", "Чистая прибыль, млрд",
                "P/S", "P/E", "Доходность прибыли (E/P), %", "Дивиденд на акцию", "Дивдоходность, %"]


def year_end_prices(prices: pd.DataFrame, min_month: int = 12) -> dict:
    """{год: (дата, цена закрытия)} по последнему торговому дню каждого календарного года.
    Год берется, только если последняя цена в нем не раньше min_month (по умолчанию декабрь): незаконченный год пропускается."""
    if prices is None or prices.empty:
        return {}
    p = prices[["begin", "close"]].dropna().sort_values("begin")
    p = p.assign(begin=pd.to_datetime(p["begin"]))
    last = p.groupby(p["begin"].dt.year).tail(1)
    return {int(r["begin"].year): (r["begin"], float(r["close"])) for _, r in last.iterrows()
            if r["begin"].month >= min_month and r["close"] > 0}


def historical_multiples(wide: pd.DataFrame, shares_mn, prices_by_year: dict, to_bn: float) -> pd.DataFrame:
    """Мультипликаторы по годам. wide: таблица по годам (year, revenue, net_income, dps); денежные показатели × to_bn = млрд
    в валюте цены. shares_mn: число акций, млн (текущее). prices_by_year: результат year_end_prices.
    Возвращает таблицу HISTORY_COLS по годам, где есть и цена на конец года, и выручка или прибыль."""
    if wide is None or wide.empty or not shares_mn or not prices_by_year:
        return pd.DataFrame(columns=HISTORY_COLS)
    rows = []
    for _, r in wide.sort_values("year").iterrows():
        year = int(r["year"])
        if year not in prices_by_year:
            continue
        rev, ni = r.get("revenue"), r.get("net_income")
        rev = None if rev is None or pd.isna(rev) else float(rev) * to_bn
        ni = None if ni is None or pd.isna(ni) else float(ni) * to_bn
        if rev is None and ni is None:
            continue
        date, price = prices_by_year[year]
        mcap = price * shares_mn / 1000
        dps = r.get("dps")
        dps = None if dps is None or pd.isna(dps) else float(dps)
        rows.append({"Год": year, "Дата цены": date, "Цена": price, "Капитализация, млрд": mcap,
                     "Выручка, млрд": rev, "Чистая прибыль, млрд": ni,
                     "P/S": mcap / rev if rev and rev > 0 else None,
                     "P/E": mcap / ni if ni and ni > 0 else None,
                     "Доходность прибыли (E/P), %": ni / mcap * 100 if ni is not None else None,
                     "Дивиденд на акцию": dps, "Дивдоходность, %": dps / price * 100 if dps is not None else None})
    return pd.DataFrame(rows, columns=HISTORY_COLS)


def compare_to_history(history: pd.DataFrame, current: dict) -> list:
    """Текущие мультипликаторы против медианы прошлых лет.
    current: {"P/S": значение, "P/E": значение}. Возвращает список словарей: показатель, текущее, медиана, лет,
    отклонение от медианы в %, вывод. Показатель пропускается, если нет текущего значения или меньше 2 лет истории."""
    out = []
    for col in ("P/S", "P/E"):
        cur = current.get(col)
        hist = history[col].dropna() if col in history.columns else pd.Series(dtype=float)
        if cur is None or len(hist) < 2:
            continue
        med = float(hist.median())
        diff = (cur / med - 1) * 100 if med > 0 else None
        if diff is None:
            verdict = "н/д"
        elif diff <= -15:
            verdict = "ниже своей истории"
        elif diff >= 15:
            verdict = "выше своей истории"
        else:
            verdict = "около своей истории"
        out.append({"Показатель": col, "Сейчас": cur, "Медиана прошлых лет": med, "Лет": int(len(hist)),
                    "Отклонение от медианы, %": diff, "Вывод": verdict})
    return out


def curve_yield(curve: pd.DataFrame, years: float):
    """Доходность ОФЗ на срок `years` лет по G-кривой (колонки period, value), линейная интерполяция. None, если кривой нет."""
    if curve is None or curve.empty:
        return None
    c = curve.dropna(subset=["period", "value"]).sort_values("period")
    if c.empty:
        return None
    return float(np.interp(years, c["period"].astype(float), c["value"].astype(float)))


def trailing_dividends(events: pd.DataFrame, as_of, days: int = 365) -> float:
    """Сумма дивидендов на акцию с датой отсечки за `days` дней до даты as_of включительно."""
    if events is None or len(events) == 0:
        return 0.0
    as_of = pd.Timestamp(as_of).normalize()
    ex = pd.to_datetime(events["ex_date"]).dt.normalize()
    mask = (ex > as_of - pd.Timedelta(days=days)) & (ex <= as_of)
    return float(events.loc[mask, "dps"].sum())


def spread_rows(items: list, ofz_yield) -> pd.DataFrame:
    """Таблица «доходность акции против ОФЗ». items: [(название, доходность %, пояснение)]."""
    rows = []
    for name, value, note in items:
        if value is None:
            continue
        rows.append({"Показатель": name, "Доходность, %": value,
                     "ОФЗ, %": ofz_yield, "Спред к ОФЗ, п.п.": value - ofz_yield if ofz_yield is not None else None,
                     "Как считается": note})
    return pd.DataFrame(rows, columns=["Показатель", "Доходность, %", "ОФЗ, %", "Спред к ОФЗ, п.п.", "Как считается"])


def history_vs_ofz(history: pd.DataFrame, ofz_by_year: dict) -> pd.DataFrame:
    """По годам: дивдоходность и E/P на конец года против доходности ОФЗ на ту же дату. ofz_by_year: {год: доходность %}."""
    cols = ["Год", "Дивдоходность, %", "Доходность прибыли (E/P), %", "ОФЗ, %", "Спред дивдоходности, п.п.", "Спред E/P, п.п."]
    if history is None or history.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, r in history.iterrows():
        ofz = ofz_by_year.get(int(r["Год"]))
        dy, ep = r["Дивдоходность, %"], r["Доходность прибыли (E/P), %"]
        dy = None if dy is None or pd.isna(dy) else float(dy)
        ep = None if ep is None or pd.isna(ep) else float(ep)
        rows.append({"Год": int(r["Год"]), "Дивдоходность, %": dy, "Доходность прибыли (E/P), %": ep, "ОФЗ, %": ofz,
                     "Спред дивдоходности, п.п.": dy - ofz if dy is not None and ofz is not None else None,
                     "Спред E/P, п.п.": ep - ofz if ep is not None and ofz is not None else None})
    return pd.DataFrame(rows, columns=cols)


def history_ru(rows_long: pd.DataFrame, macro: dict, prices: pd.DataFrame):
    """Исторические мультипликаторы российской компании по данным файла и ценам Мосбиржи.
    Денежные показатели в $ переводятся в рубли по среднему курсу года. Возвращает (таблица, примечания)."""
    notes = []
    rows_rub, adj_notes = adjust_long(rows_long, True, False, None, macro)
    notes += adj_notes
    w, unit, mixed = make_wide(rows_rub)
    k = UNIT_TO_BN.get(unit)
    shares = latest_value(w, "shares") if not w.empty else None
    if w.empty or k is None or mixed:
        notes.append("Единицы показателей не позволяют посчитать мультипликаторы по годам.")
        return pd.DataFrame(columns=HISTORY_COLS), notes
    if shares is None:
        notes.append("Не указано число акций: мультипликаторы по годам не считаются.")
        return pd.DataFrame(columns=HISTORY_COLS), notes
    return historical_multiples(w, shares[1], year_end_prices(prices), k), notes


def history_us(data: dict, prices: pd.DataFrame):
    """Исторические мультипликаторы зарубежной компании по данным Yahoo Finance. Возвращает (таблица, примечания)."""
    notes = []
    w = data.get("wide")
    if w is None or w.empty:
        return pd.DataFrame(columns=HISTORY_COLS), ["Нет годовой отчетности."]
    if data.get("fin_currency") and data["fin_currency"] != data.get("currency"):
        return pd.DataFrame(columns=HISTORY_COLS), [f"Валюта отчетности ({data['fin_currency']}) отличается от валюты торгов: "
                                                    "мультипликаторы по годам не считаются."]
    shares = data.get("shares")
    if not shares:
        return pd.DataFrame(columns=HISTORY_COLS), ["Yahoo не вернул число акций: мультипликаторы по годам не считаются."]
    w = w.copy()
    div = data.get("dividends")
    if div is not None and not div.empty:
        w["dps"] = w["year"].map(dict(zip(div["year"].astype(int), div["dps"].astype(float))))
        notes.append("Дивиденд года — сумма выплат за календарный год (Yahoo Finance).")
    else:
        w["dps"] = float("nan")
    notes.append("Отчетный год компании может не совпадать с календарным: цена берется на конец календарного года.")
    return historical_multiples(w, shares / 1e6, year_end_prices(prices), 1e-9), notes
