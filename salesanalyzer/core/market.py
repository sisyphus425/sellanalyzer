"""Мультипликаторы (P/S, P/E), капитализация, дивиденды и payout."""

import pandas as pd

from salesanalyzer.config import PAYOUT_COLS, UNIT_TO_BN
from salesanalyzer.core.adjustments import adjust_long
from salesanalyzer.core.profitability import latest_value, make_wide


def _metric_of(rows: pd.DataFrame, indicator: str) -> str:
    sub = rows[rows["indicator"] == indicator].sort_values("year")
    return "" if sub.empty else sub["metric"].iloc[-1]


def _payout_rows(dps_rows, shares_mn, ni_bn_by_year, est_years, divisor):
    """Таблица дивидендов по годам: DPS, сумма выплат (≈ по текущему числу акций), payout к чистой прибыли."""
    out = []
    for _, r in dps_rows.sort_values("year").iterrows():
        y, dps = int(r["year"]), float(r["dps"])
        total = dps * shares_mn / divisor if shares_mn else None
        ni_bn = ni_bn_by_year.get(y)
        payout = total / ni_bn * 100 if total is not None and ni_bn and ni_bn > 0 else None
        out.append({"Год": y, "Дивиденд на акцию": dps,
                    "Дивиденды всего, млрд ≈": None if total is None else round(total, 1),
                    "Чистая прибыль, млрд": None if ni_bn is None else round(ni_bn, 1),
                    "Payout, %": None if payout is None else round(payout, 1),
                    "Оценка": "да" if y in est_years else ""})
    return pd.DataFrame(out, columns=PAYOUT_COLS)


def market_stats_ru(rows_long: pd.DataFrame, macro: dict, price, price_date):
    """Капитализация, P/S, P/E, дивиденды для российской компании. Денежные показатели в $ переводятся в ₽ по среднему курсу."""
    rows_rub, notes = adjust_long(rows_long, True, False, None, macro)
    w, unit, _ = make_wide(rows_rub)
    if w.empty:
        return None, pd.DataFrame(columns=PAYOUT_COLS), notes + ["Нет данных по компании."]
    k = UNIT_TO_BN.get(unit)
    if k is None:
        notes.append(f"Единицы «{unit}» не поддерживаются: мультипликаторы не считаются.")
    shares, rev, ni = latest_value(w, "shares"), latest_value(w, "revenue"), latest_value(w, "net_income")
    if shares is None:
        notes.append("Не указано число акций (indicator = shares): капитализация и мультипликаторы не считаются.")
    if price is None:
        notes.append("Нет цены акции: капитализация и мультипликаторы не считаются.")
    mcap = price * shares[1] / 1000 if price and shares else None  # млн шт. × ₽ = млн ₽ -> млрд ₽
    ps = mcap / (rev[1] * k) if mcap and rev and k and rev[1] > 0 else None
    pe = mcap / (ni[1] * k) if mcap and ni and k and ni[1] > 0 else None

    dps_rows = w.dropna(subset=["dps"])
    est_years = set(rows_rub[(rows_rub["indicator"] == "dps") & rows_rub["is_estimate"]]["year"].astype(int))
    ni_bn = {int(r["year"]): r["net_income"] * k for _, r in w.dropna(subset=["net_income"]).iterrows()} if k else {}
    payout = _payout_rows(dps_rows, shares[1] if shares else None, ni_bn, est_years, 1000)
    dps_last = None if dps_rows.empty else (int(dps_rows["year"].iloc[-1]), float(dps_rows["dps"].iloc[-1]))
    return {
        "price": price, "price_date": price_date, "currency": "₽", "mcap": mcap, "shares_mn": shares[1] if shares else None,
        "ps": ps, "pe": pe, "y_rev": rev[0] if rev else None, "y_ni": ni[0] if ni else None,
        "rev_metric": _metric_of(rows_rub, "revenue"), "ni_metric": _metric_of(rows_rub, "net_income"),
        "div_yield": dps_last[1] / price * 100 if dps_last and price else None,
        "dps_last": dps_last, "dps_est": bool(dps_last and dps_last[0] in est_years), "div_basis": "последний год из файла",
    }, payout, notes


def market_stats_us(data: dict, price, price_date):
    """Те же показатели для зарубежной компании по данным yfinance."""
    notes, w, cur = [], data["wide"], data["currency"]
    mcap_raw = data.get("market_cap")
    if data.get("fin_currency") and data["fin_currency"] != cur:
        notes.append(f"Валюта отчетности ({data['fin_currency']}) отличается от валюты торгов ({cur}): мультипликаторы не считаются.")
        mcap_raw = None
    if not mcap_raw:
        notes.append("Yahoo не вернул рыночную капитализацию: мультипликаторы не считаются.")
    rev = latest_value(w, "revenue") if not w.empty else None
    ni = latest_value(w, "net_income") if not w.empty else None
    ps = mcap_raw / rev[1] if mcap_raw and rev and rev[1] > 0 else None
    pe = mcap_raw / ni[1] if mcap_raw and ni and ni[1] > 0 else None
    shares = data.get("shares")
    shares_mn = shares / 1e6 if shares else None
    ni_bn = {int(r["year"]): r["net_income"] / 1e9 for _, r in w.dropna(subset=["net_income"]).iterrows()} if not w.empty else {}
    div_df = data.get("dividends")
    if div_df is None or div_df.empty:
        payout = pd.DataFrame(columns=PAYOUT_COLS)
    else:
        payout = _payout_rows(div_df.rename(columns={"dps": "dps"}), shares_mn, ni_bn, set(), 1000)
    ttm = data.get("div_ttm")
    return {
        "price": price, "price_date": price_date, "currency": cur, "mcap": mcap_raw / 1e9 if mcap_raw else None,
        "shares_mn": shares_mn, "ps": ps, "pe": pe, "y_rev": rev[0] if rev else None, "y_ni": ni[0] if ni else None,
        "rev_metric": "Total Revenue (Yahoo Finance)", "ni_metric": "Net Income (Yahoo Finance)",
        "div_yield": ttm / price * 100 if ttm and price else None,
        "dps_last": (int(div_df["year"].iloc[-1]), float(div_df["dps"].iloc[-1])) if div_df is not None and not div_df.empty else None,
        "dps_est": False, "div_basis": "выплаты за последние 12 месяцев",
    }, payout, notes


def market_tables(stats: dict, payout: pd.DataFrame) -> dict:
    cur = stats["currency"]
    kv = [(f"Цена, {cur}", stats["price"]), ("Дата цены", None if stats["price_date"] is None else f"{stats['price_date']:%Y-%m-%d}"),
          ("Число акций, млн", stats["shares_mn"]), (f"Капитализация, млрд {cur}", stats["mcap"]),
          (f"P/S (выручка {stats['y_rev']})", stats["ps"]), (f"P/E (прибыль {stats['y_ni']})", stats["pe"]),
          ("Дивдоходность, %", stats["div_yield"])]
    out = {"Мультипликаторы": pd.DataFrame(kv, columns=["Показатель", "Значение"])}
    if not payout.empty:
        out["Дивиденды"] = payout
    return out


def market_summary_row(label: str, stats: dict, payout: pd.DataFrame) -> dict:
    pr = payout["Payout, %"].dropna() if not payout.empty else pd.Series(dtype=float)
    r = lambda v, n=2: None if v is None else round(v, n)
    return {"Компания": label, "Валюта": stats["currency"], "Цена": r(stats["price"]), "Капитализация, млрд": r(stats["mcap"], 1),
            "P/S": r(stats["ps"]), "P/E": r(stats["pe"], 1),
            "Дивиденд на акцию": None if stats["dps_last"] is None else stats["dps_last"][1],
            "Дивдоходность, %": r(stats["div_yield"]), "Payout, %": None if pr.empty else float(pr.iloc[-1]),
            "Год выручки": stats["y_rev"], "Год прибыли": stats["y_ni"]}
