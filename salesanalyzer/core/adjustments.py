"""Пересчет денежных показателей: доллары в рубли по курсу ЦБ и поправка на инфляцию."""

import pandas as pd

from salesanalyzer.config import MONEY_INDICATORS, USD_UNITS


def price_levels(cpi: dict) -> dict:
    """Средний уровень цен в году (середина года) относительно конца года перед первым годом ряда.
    Берется среднее геометрическое уровней на конец предыдущего и текущего года; ряд должен быть непрерывным."""
    levels, prev, last_year = {}, 1.0, None
    for y in sorted(cpi):
        if last_year is not None and y != last_year + 1:
            break  # разрыв в годах: дальше цепочку считать нельзя
        end = prev * (1 + cpi[y] / 100)
        levels[y] = (prev * end) ** 0.5
        prev, last_year = end, y
    return levels


def adjust_long(rows: pd.DataFrame, to_rub: bool, real: bool, base_year, macro: dict):
    """Пересчет денежных показателей: $ -> ₽ по среднему курсу года и/или приведение к ценам base_year.
    ROE не меняется. Возвращает (таблица, список замечаний)."""
    notes = []
    if rows.empty or not (to_rub or real):
        return rows, notes
    levels = price_levels(macro["cpi"]) if real else {}
    if real and base_year not in levels:
        notes.append("Нет данных об инфляции для выбранного базового года: поправка не применена.")
        real = False
    df = rows.copy()
    missing_fx, missing_cpi, skipped_real = set(), set(), set()
    values, units, extra = [], [], []
    nan = float("nan")
    for _, r in df.iterrows():
        v, u, how = r["value"], r["unit"], []
        if r["indicator"] in MONEY_INDICATORS:
            if to_rub and u in USD_UNITS:
                rate = macro["usd_rub"].get(int(r["year"]))
                if rate is None:
                    missing_fx.add(int(r["year"]))
                    v = nan
                else:
                    new_u, k = USD_UNITS[u]
                    v, u = v * rate * k, new_u
                    how.append(f"{rate:.2f} ₽ за $")
            if real:
                if "$" in u:
                    skipped_real.add(r["ticker"])
                elif int(r["year"]) in levels:
                    k = levels[base_year] / levels[int(r["year"])]
                    v *= k
                    how.append(f"в ценах {base_year} г. (×{k:.3f})")
                    u = f"{u} (цены {base_year} г.)"
                else:
                    missing_cpi.add(int(r["year"]))
                    v = nan
        values.append(v)
        units.append(u)
        extra.append("; пересчет: " + ", ".join(how) if how else "")
    df["value"], df["unit"] = values, units
    df["note"] = [(n + e).lstrip("; ") for n, e in zip(df["note"], extra)]
    df = df.dropna(subset=["value"])
    if missing_fx:
        notes.append("Нет курса ЦБ за годы: " + ", ".join(map(str, sorted(missing_fx))) + " — эти значения исключены.")
    if missing_cpi:
        notes.append("Нет данных об инфляции за годы: " + ", ".join(map(str, sorted(missing_cpi))) + " — эти значения исключены.")
    if skipped_real:
        notes.append("Для значений в долларах (" + ", ".join(sorted(skipped_real)) + ") поправка на российскую инфляцию "
                     "не применяется. Включите пересчет в рубли.")
    return df.reset_index(drop=True), notes
