"""Исторические мультипликаторы и сравнение доходности акции с ОФЗ (core/valuation.py, get_ofz_curve)."""
import importlib

import numpy as np
import pandas as pd
import pytest


def daily(start, end, price_fn=lambda d: 100.0):
    days = pd.date_range(start, end, freq="B")
    return pd.DataFrame({"begin": days, "close": [float(price_fn(d)) for d in days]})


def wide(*rows):
    """rows: (year, revenue, net_income, dps)."""
    return pd.DataFrame(rows, columns=["year", "revenue", "net_income", "dps"])


# ---------- цены на конец года ----------

def test_year_end_price_is_the_last_trading_day_and_unfinished_year_is_skipped(app):
    p = daily("2023-06-01", "2025-06-30", lambda d: d.year * 10 + d.month)
    out = app.year_end_prices(p)
    assert sorted(out) == [2023, 2024]  # 2025 не закончился
    assert out[2023] == (pd.Timestamp("2023-12-29"), 2023 * 10 + 12.0)
    assert out[2024][0] == pd.Timestamp("2024-12-31")
    assert app.year_end_prices(pd.DataFrame(columns=["begin", "close"])) == {}


# ---------- мультипликаторы по годам ----------

def test_historical_multiples_values(app):
    prices = {2024: (pd.Timestamp("2024-12-30"), 200.0)}
    h = app.historical_multiples(wide((2023, 500, 50, 5), (2024, 1000, 100, 10)), 1000, prices, 1.0)
    assert h["Год"].tolist() == [2024]  # за 2023 нет цены
    r = h.iloc[0]
    assert r["Капитализация, млрд"] == pytest.approx(200.0)  # 200 ₽ × 1000 млн шт. = 200 млрд
    assert r["P/S"] == pytest.approx(0.2) and r["P/E"] == pytest.approx(2.0)
    assert r["Доходность прибыли (E/P), %"] == pytest.approx(50.0) and r["Дивдоходность, %"] == pytest.approx(5.0)


def test_loss_gives_no_pe_but_negative_earnings_yield(app):
    prices = {2024: (pd.Timestamp("2024-12-30"), 100.0)}
    r = app.historical_multiples(wide((2024, 1000, -50, None)), 1000, prices, 1.0).iloc[0]
    assert r["P/E"] is None and r["Доходность прибыли (E/P), %"] == pytest.approx(-50.0) and r["Дивдоходность, %"] is None


def test_units_are_converted_to_billions(app):
    """Выручка в млн: множитель 1/1000 переводит в млрд перед делением."""
    prices = {2024: (pd.Timestamp("2024-12-30"), 100.0)}
    r = app.historical_multiples(wide((2024, 50_000, 10_000, None)), 1000, prices, 1 / 1000).iloc[0]
    assert r["Выручка, млрд"] == pytest.approx(50.0) and r["P/S"] == pytest.approx(2.0)


def test_split_does_not_change_past_market_cap(app):
    """Дробление 1:100: биржа делит прошлые цены на 100, число акций в файле ×100 — капитализация та же, что до дробления."""
    before = app.historical_multiples(wide((2023, 1000, 100, 150)), 152.86, {2023: (pd.Timestamp("2023-12-29"), 16172.0)}, 1.0)
    after = app.historical_multiples(wide((2023, 1000, 100, 1.5)), 15286.0, {2023: (pd.Timestamp("2023-12-29"), 161.72)}, 1.0)
    for col in ("Капитализация, млрд", "P/E", "P/S", "Дивдоходность, %"):
        assert after[col].iloc[0] == pytest.approx(before[col].iloc[0])


def test_no_shares_or_prices_gives_empty_table(app):
    w = wide((2024, 1000, 100, None))
    assert app.historical_multiples(w, None, {2024: (pd.Timestamp("2024-12-30"), 1.0)}, 1.0).empty
    assert app.historical_multiples(w, 100, {}, 1.0).empty


def test_compare_to_history_verdicts(app):
    h = pd.DataFrame({"P/E": [10.0, 12.0, 14.0], "P/S": [2.0, 2.0, None]})
    res = {r["Показатель"]: r for r in app.compare_to_history(h, {"P/E": 8.0, "P/S": 2.1})}
    assert res["P/E"]["Медиана прошлых лет"] == 12.0 and res["P/E"]["Вывод"] == "ниже своей истории"
    assert res["P/E"]["Отклонение от медианы, %"] == pytest.approx(-100 / 3)
    assert res["P/S"]["Вывод"] == "около своей истории" and res["P/S"]["Лет"] == 2
    assert app.compare_to_history(h, {"P/E": 20.0})[0]["Вывод"] == "выше своей истории"
    assert app.compare_to_history(h.head(1), {"P/E": 8.0}) == []  # меньше двух лет истории
    assert app.compare_to_history(h, {"P/E": None, "P/S": None}) == []


# ---------- ОФЗ ----------

def curve(points):
    return pd.DataFrame({"tradedate": pd.Timestamp("2026-11-20"), "period": [p for p, _ in points], "value": [v for _, v in points]})


def test_curve_yield_interpolates_and_clamps(app):
    c = curve([(0.5, 10.0), (1.0, 12.0), (3.0, 16.0)])
    assert app.curve_yield(c, 1.0) == pytest.approx(12.0)
    assert app.curve_yield(c, 2.0) == pytest.approx(14.0)
    assert app.curve_yield(c, 10.0) == pytest.approx(16.0)  # за пределами кривой — крайнее значение
    assert app.curve_yield(pd.DataFrame(columns=["period", "value"]), 1.0) is None


def test_trailing_dividends_window(app):
    ev = pd.DataFrame({"ex_date": pd.to_datetime(["2025-11-01", "2026-03-01", "2026-11-20", "2026-12-01"]), "dps": [1.0, 2.0, 3.0, 4.0]})
    assert app.trailing_dividends(ev, "2026-11-20") == pytest.approx(5.0)  # 01.03 и 20.11; 01.11.2025 старше года, 01.12 в будущем
    assert app.trailing_dividends(None, "2026-11-20") == 0.0


def test_spread_rows_and_history_vs_ofz(app):
    t = app.spread_rows([("Дивдоходность", 12.0, "x"), ("Пусто", None, "y"), ("E/P", 20.0, "z")], 14.0)
    assert t["Показатель"].tolist() == ["Дивдоходность", "E/P"] and t["Спред к ОФЗ, п.п."].tolist() == pytest.approx([-2.0, 6.0])
    h = pd.DataFrame({"Год": [2023, 2024], "Дивдоходность, %": [10.0, None], "Доходность прибыли (E/P), %": [20.0, 25.0]})
    out = app.history_vs_ofz(h, {2023: 12.0})
    assert out.iloc[0]["Спред дивдоходности, п.п."] == pytest.approx(-2.0) and out.iloc[0]["Спред E/P, п.п."] == pytest.approx(8.0)
    assert pd.isna(out.iloc[1]["ОФЗ, %"]) and pd.isna(out.iloc[1]["Спред E/P, п.п."])


# ---------- сборка истории РФ и США ----------

def test_history_ru_converts_dollars_to_rubles(app, mk_rows):
    rows = mk_rows(("GMKN", 2024, "revenue", 10_000.0, "млн $"), ("GMKN", 2024, "net_income", 2_000.0, "млн $"),
                   ("GMKN", 2024, "shares", 1_000.0, "млн шт."), ("GMKN", 2024, "dps", 5.0, "₽ на акцию"))
    prices = daily("2024-12-01", "2024-12-31", lambda d: 100.0)
    h, _ = app.history_ru(rows, {"usd_rub": {2024: 90.0}, "cpi": {}}, prices)
    r = h.iloc[0]
    assert r["Выручка, млрд"] == pytest.approx(900.0)  # 10 000 млн $ × 90 / 1000
    assert r["Капитализация, млрд"] == pytest.approx(100.0) and r["P/E"] == pytest.approx(100 / 180)
    assert r["Дивдоходность, %"] == pytest.approx(5.0)


def test_history_ru_without_shares(app, mk_rows):
    rows = mk_rows(("X", 2024, "revenue", 100.0, "млрд ₽"))
    h, notes = app.history_ru(rows, {"usd_rub": {}, "cpi": {}}, daily("2024-12-01", "2024-12-31"))
    assert h.empty and any("число акций" in n.lower() for n in notes)


def us_data(**over):
    w = pd.DataFrame({"year": [2024], "revenue": [400e9], "net_income": [100e9], "ebitda": [np.nan], "roe": [np.nan]})
    d = {"wide": w, "currency": "USD", "fin_currency": "USD", "shares": 10e9,
         "dividends": pd.DataFrame({"year": [2024], "dps": [1.0]})}
    d.update(over)
    return d


def test_history_us(app):
    h, notes = app.history_us(us_data(), daily("2024-12-01", "2024-12-31", lambda d: 200.0))
    r = h.iloc[0]
    assert r["Капитализация, млрд"] == pytest.approx(2000.0) and r["P/S"] == pytest.approx(5.0) and r["P/E"] == pytest.approx(20.0)
    assert r["Дивдоходность, %"] == pytest.approx(0.5) and notes


def test_history_us_currency_mismatch_and_missing_shares(app):
    prices = daily("2024-12-01", "2024-12-31")
    assert app.history_us(us_data(fin_currency="TWD"), prices)[0].empty
    assert app.history_us(us_data(shares=None), prices)[0].empty
    assert app.history_us(us_data(dividends=pd.DataFrame()), prices)[0]["Дивдоходность, %"].isna().all()


# ---------- загрузка G-кривой ----------

def test_ofz_curve_walks_back_to_the_last_trading_day(app, cfg, monkeypatch):
    moex = importlib.import_module("salesanalyzer.data_sources.moex")
    asked = []

    def fake_iss_get(path, **params):
        asked.append(params.get("date"))
        if params.get("date") != "2025-12-30":
            return {"yearyields": pd.DataFrame(columns=["tradedate", "tradetime", "period", "value"])}
        return {"yearyields": pd.DataFrame({"tradedate": ["2025-12-30"] * 2, "tradetime": ["18:39:59"] * 2,
                                            "period": [0.5, 1.0], "value": [13.0, 13.14]})}

    monkeypatch.setattr(moex, "iss_get", fake_iss_get)
    out = moex.get_ofz_curve("2025-12-31")
    assert asked == ["2025-12-31", "2025-12-30"]
    assert out["period"].tolist() == [0.5, 1.0] and out["tradedate"].iloc[0] == pd.Timestamp("2025-12-30")
    assert app.curve_yield(out, 1) == pytest.approx(13.14)

    asked.clear()
    assert moex.get_ofz_curve("2025-12-20", lookback_days=2).empty and len(asked) == 3
    asked.clear()
    moex.get_ofz_curve()
    assert asked == [None]  # без даты — последняя доступная кривая
