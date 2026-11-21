"""Портфель: состав, пересчет в рубли, общая история, стоимость, вклады, корреляции, дивиденды, сбор данных (core/portfolio.py, services)."""
import importlib

import numpy as np
import pandas as pd
import pytest


def series(values, start="2025-01-01", freq="D"):
    return pd.DataFrame({"begin": pd.date_range(start, periods=len(values), freq=freq), "close": [float(v) for v in values]})


def events(*items):
    return pd.DataFrame({"ex_date": pd.to_datetime([d for d, _ in items]), "dps": [float(v) for _, v in items]})


# ---------- состав ----------

def test_normalize_positions_cleans_and_merges(app):
    raw = pd.DataFrame({"ticker": [" sber", "SBER", "aapl", "", "LKOH", "GMKN"], "market": ["РФ", "ru", "США", "ru", "ru", "ru"],
                        "qty": [10, 30, "5", 3, 0, None], "buy_price": [100, 200, None, 1, 1, 1]})
    out = app.normalize_positions(raw).set_index("ticker")
    assert list(out.index) == ["SBER", "AAPL"]  # пустой тикер, нулевое и пустое количество убраны
    assert out.loc["SBER", "qty"] == 40 and out.loc["SBER", "buy_price"] == pytest.approx(175.0)  # (10×100 + 30×200) / 40
    assert out.loc["AAPL", "market"] == "us" and pd.isna(out.loc["AAPL", "buy_price"])
    assert app.normalize_positions(None).empty


def test_buy_price_unknown_if_any_lot_has_no_price(app):
    out = app.normalize_positions(pd.DataFrame({"ticker": ["SBER", "SBER"], "market": ["ru", "ru"], "qty": [1, 1], "buy_price": [100, None]}))
    assert pd.isna(out["buy_price"].iloc[0])


def test_weights_to_qty_normalizes_weights(app):
    qty = app.weights_to_qty(pd.Series({"A": 30, "B": 30}), pd.Series({"A": 100.0, "B": 50.0}), 1000)
    assert qty.to_dict() == pytest.approx({"A": 5.0, "B": 10.0})  # 50% и 50% от 1000 ₽
    assert app.weights_to_qty(pd.Series({"A": 0}), pd.Series({"A": 1.0}), 1000).sum() == 0


def test_read_portfolio_csv_formats(app):
    out = app.read_portfolio_csv("ticker;market;qty;weight;buy_price\nsber;ru;100;;250,5\nAAPL;us;5;;\n;ru;1;;\n".encode("utf-8-sig"))
    assert out["ticker"].tolist() == ["SBER", "AAPL"] and out["market"].tolist() == ["ru", "us"]
    assert out["buy_price"].iloc[0] == 250.5 and pd.isna(out["weight"]).all()
    only = app.read_portfolio_csv(b"ticker,qty\nGAZP,10\n")
    assert only["market"].tolist() == ["ru"] and only["qty"].tolist() == [10.0]
    with pytest.raises(ValueError):
        app.read_portfolio_csv(b"name,qty\nX,1\n")


# ---------- валюта ----------

def test_rate_on_takes_last_known_rate(app):
    fx = series([90, 91, 92], start="2025-01-02")  # 2, 3, 4 января
    rates = app.rate_on(pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-10"]), fx)
    assert rates.tolist() == [90.0, 91.0, 92.0]  # до начала ряда — первый курс; после конца — последний
    assert np.isnan(app.rate_on(["2025-01-01"], pd.DataFrame())).all()


def test_to_rub_converts_prices_and_dividends_at_their_dates(app):
    fx = series([100, 50], start="2025-01-01")
    p, ev = app.to_rub(series([10, 10], start="2025-01-01"), events(("2025-01-02", 1)), "USD", fx)
    assert p["close"].tolist() == [1000.0, 500.0] and ev["dps"].tolist() == [50.0]
    p, ev = app.to_rub(series([10]), None, "RUB", None)
    assert p["close"].tolist() == [10.0] and ev.empty
    assert app.to_rub(series([10]), None, "USD", pd.DataFrame()) == (None, None)


# ---------- общая история и стоимость ----------

def test_build_panel_aligns_calendars_and_fills_gaps(app):
    a = series([100, 101, 102, 103, 104], start="2025-01-01")  # 1–5 января
    b = series([10, 11, 12], start="2025-01-02", freq="2D")  # 2, 4, 6 января
    prices, tr = app.build_panel({"A": {"prices": a, "dividends": None}, "B": {"prices": b, "dividends": None}})
    assert prices.index[0] == pd.Timestamp("2025-01-02")  # с даты, когда есть цены всех бумаг
    assert prices.index[-1] == pd.Timestamp("2025-01-06")
    assert prices.loc["2025-01-03", "B"] == 10.0 and prices.loc["2025-01-06", "A"] == 104.0  # пропуски — последней ценой
    assert np.allclose(tr.to_numpy(), prices.to_numpy())  # без дивидендов полная доходность = цена


def test_build_panel_total_return_and_years_window(app):
    a = series([100] * 400 + [90] * 400, start="2023-01-01")
    prices, tr = app.build_panel({"A": {"prices": a, "dividends": events(("2024-02-05", 10))}}, years=1)
    assert prices.index[0] >= prices.index[-1] - pd.DateOffset(years=1)
    full_p, full_tr = app.build_panel({"A": {"prices": a, "dividends": events(("2024-02-05", 10))}})
    assert full_tr["A"].iloc[-1] == pytest.approx(100.0) and full_p["A"].iloc[-1] == 90.0  # гэп компенсирован


def test_value_series_and_holdings(app):
    prices = pd.DataFrame({"A": [10.0, 12.0], "B": [100.0, 90.0]}, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))
    v = app.value_series(prices, pd.Series({"A": 10, "B": 1}))
    assert v["close"].tolist() == [200.0, 210.0]
    pos = pd.DataFrame({"ticker": ["A", "B"], "market": ["ru", "us"], "qty": [10.0, 1.0], "buy_price": [8.0, np.nan]})
    h = app.holdings_table(pos, {"A": 12.0, "B": 2.0}, {"A": 1.0, "B": 45.0}, {"A": "RUB", "B": "USD"}).set_index("Тикер")
    assert h.loc["A", "Стоимость, ₽"] == 120.0 and h.loc["B", "Стоимость, ₽"] == 90.0
    assert h["Доля, %"].sum() == pytest.approx(100.0) and h.loc["A", "Прибыль/убыток, ₽"] == pytest.approx(40.0)
    assert h.loc["A", "Прибыль/убыток, %"] == pytest.approx(50.0) and pd.isna(h.loc["B", "Прибыль/убыток, ₽"])


def test_contributions_add_up_and_risk_shares_sum_to_100(app):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=300)
    prices = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0, s, 300))) for t, s in (("A", 0.01), ("B", 0.03))}, index=idx)
    qty = pd.Series({"A": 3.0, "B": 1.0})
    c = app.contributions(prices, prices, qty).set_index("Тикер")
    v = app.value_series(prices, qty)["close"]
    assert c["Вклад в доходность по цене, п.п."].sum() == pytest.approx((v.iloc[-1] / v.iloc[0] - 1) * 100)
    assert c["Вклад в риск, %"].sum() == pytest.approx(100.0) and c["Доля сейчас, %"].sum() == pytest.approx(100.0)
    share_b, risk_b = c.loc["B", "Доля сейчас, %"], c.loc["B", "Вклад в риск, %"]
    assert risk_b > share_b  # более волатильная бумага дает риска больше своей доли


def test_correlation_matrix(app):
    idx = pd.date_range("2025-01-01", periods=50)
    base = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 50))
    tr = pd.DataFrame({"A": base, "B": base * 2}, index=idx)
    assert app.correlation_matrix(tr).loc["A", "B"] == pytest.approx(1.0)
    assert app.correlation_matrix(tr[["A"]]).empty


# ---------- дивиденды ----------

def test_portfolio_dividend_events_scale_by_quantity_and_sum_by_date(app):
    out = app.portfolio_dividend_events({"A": events(("2025-03-01", 2)), "B": events(("2025-03-01", 1), ("2025-04-01", 5)), "C": None},
                                        pd.Series({"A": 10.0, "B": 3.0}))
    assert out.set_index("ex_date")["dps"].to_dict() == {pd.Timestamp("2025-03-01"): 23.0, pd.Timestamp("2025-04-01"): 15.0}


def test_dividend_calendar_and_monthly(app):
    divs = {"A": events(("2026-01-15", 2), ("2026-07-20", 3)), "B": events(("2026-03-10", 0.5))}
    cal = app.dividend_calendar(divs, pd.Series({"A": 10.0, "B": 4.0}), {"A": 1.0, "B": 80.0}, "2026-11-20")
    assert cal["Тикер"].tolist() == ["A", "B", "A"]
    assert cal["Сумма, ₽"].tolist() == pytest.approx([20.0, 160.0, 30.0])  # B: 0,5 $ × 4 × 80 ₽
    assert (cal["Откуда"] == "как год назад").all()
    monthly = app.monthly_dividends(cal)
    assert monthly["Сумма, ₽"].tolist() == pytest.approx([20.0, 160.0, 30.0]) and len(monthly) == 3
    assert app.dividend_calendar({}, pd.Series(dtype=float), {}, "2026-11-20").empty


# ---------- сбор данных ----------

@pytest.fixture
def services(app, monkeypatch):
    mod = importlib.import_module("salesanalyzer.services")
    data = {"SBER": series([100, 110], start="2026-11-19"), "AAPL": series([10, 11], start="2026-11-19")}
    monkeypatch.setattr(mod, "get_moex_candles", lambda t, years: data.get(t, pd.DataFrame()))
    monkeypatch.setattr(mod, "get_us_prices", lambda t, years: data.get(t, pd.DataFrame()))
    monkeypatch.setattr(mod, "get_us_dividend_events", lambda t, years: events(("2026-11-20", 0.1)))
    monkeypatch.setattr(mod, "get_us_data", lambda t: {"currency": "USD"})
    monkeypatch.setattr(mod, "load_dividends_ru", lambda t: events())
    monkeypatch.setattr(mod, "get_fx_rub", lambda cur, years: series([80, 90], start="2026-11-19"))
    return mod


def test_collect_portfolio(services):
    pos = pd.DataFrame({"ticker": ["SBER", "AAPL", "MYCO"], "market": ["ru", "us", "ru"], "qty": [1.0, 2.0, 1.0], "buy_price": [np.nan] * 3})
    problems = []
    out = services.collect_portfolio(pos, 1, problems)
    assert list(out["positions"]["ticker"]) == ["SBER", "AAPL"]
    assert out["currency"] == {"SBER": "RUB", "AAPL": "USD"} and out["last_rate"]["AAPL"] == 90.0
    assert out["rub"]["AAPL"]["prices"]["close"].tolist() == [800.0, 990.0]
    assert out["rub"]["AAPL"]["dividends"]["dps"].tolist() == pytest.approx([9.0])  # 0,1 $ × 90 ₽
    assert any("MYCO" in p and "частную" in p for p in problems)


def test_collect_portfolio_without_fx(services, monkeypatch):
    monkeypatch.setattr(services, "get_fx_rub", lambda cur, years: pd.DataFrame(columns=["begin", "close"]))
    problems = []
    out = services.collect_portfolio(pd.DataFrame({"ticker": ["AAPL"], "market": ["us"], "qty": [1.0], "buy_price": [np.nan]}), 1, problems)
    assert out["positions"].empty and any("нет курса USD/RUB" in p for p in problems)


def test_fx_loader_pages_and_prefers_main_clearing(app, monkeypatch):
    moex = importlib.import_module("salesanalyzer.data_sources.moex")
    rows = [["2026-11-19", "13:45:00", "USD/RUB", 81.0, "tc"], ["2026-11-19", "18:49:00", "USD/RUB", 82.0, "mc"],
            ["2026-11-20", "13:45:00", "USD/RUB", 83.0, "tc"]]
    cols = ["tradedate", "tradetime", "secid", "rate", "clearing"]
    calls = []

    def fake(path, **params):
        calls.append(params["start"])
        chunk = rows[params["start"]:params["start"] + 2]
        return {"securities": pd.DataFrame(chunk, columns=cols),
                "securities.cursor": pd.DataFrame([[params["start"], len(rows), 2]], columns=["INDEX", "TOTAL", "PAGESIZE"])}

    monkeypatch.setattr(moex, "iss_get", fake)
    out = moex.get_fx_rub("usd", 1)
    assert calls == [0, 2] and out["close"].tolist() == [82.0, 83.0]  # за 19.11 — основной клиринг, за 20.11 — промежуточный
    assert moex.get_fx_rub("RUB", 1).empty
