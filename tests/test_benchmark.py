"""Полная доходность с дивидендами и сравнение с индексом рынка (core/benchmark.py, services.collect_prices)."""
import numpy as np
import pandas as pd
import pytest


def prices(closes, start="2025-01-01", freq="D"):
    return pd.DataFrame({"begin": pd.date_range(start, periods=len(closes), freq=freq), "close": [float(c) for c in closes]})


def events(*items):
    return pd.DataFrame({"ex_date": pd.to_datetime([d for d, _ in items]), "dps": [float(v) for _, v in items]})


# ---------- полная доходность ----------

def test_total_return_without_dividends_equals_price(app):
    p = app.add_total_return(prices([100, 110, 99, 120]))
    assert np.allclose(p["tr"], p["close"]) and (p["div"] == 0).all()
    m = app.total_return_metrics(p)
    assert m["tr_total"] == pytest.approx(20.0) and m["price_total"] == pytest.approx(20.0) and m["div_effect"] == pytest.approx(0)


def test_dividend_gap_is_compensated_by_reinvestment(app):
    """Цена 100, в день отсечки падает ровно на дивиденд 10 и дальше не меняется: полная доходность 0%, по цене −10%."""
    p = app.add_total_return(prices([100, 100, 90, 90]), events(("2025-01-03", 10)))
    m = app.total_return_metrics(p)
    assert m["price_total"] == pytest.approx(-10.0)
    assert m["tr_total"] == pytest.approx(0.0)
    assert m["div_effect"] == pytest.approx(10.0)
    assert m["n_dividends"] == 1 and m["dividends_sum"] == pytest.approx(10.0)


def test_reinvested_dividend_grows_with_price(app):
    """Дивиденд 10 при цене 90 докупает 1/9 акции; после роста цены до 99 полная доходность (90+10)/100 × 99/90 − 1 = 10%."""
    p = app.add_total_return(prices([100, 90, 99]), events(("2025-01-02", 10)))
    assert app.total_return_metrics(p)["tr_total"] == pytest.approx(10.0)


def test_dividend_on_a_non_trading_day_goes_to_the_next_trading_day(app):
    p = prices([100, 100, 90], start="2025-01-03")  # 3, 4, 5 января
    p = p[p["begin"] != pd.Timestamp("2025-01-04")]  # 4 января торгов нет
    booked = app.book_dividends(p, events(("2025-01-04", 10)))
    assert booked["booked"].tolist() == [pd.Timestamp("2025-01-05")]


def test_dividends_outside_the_period_or_on_the_first_day_are_ignored(app):
    p = prices([100, 101, 102])  # 1–3 января
    ev = events(("2024-12-20", 5), ("2025-01-01", 5), ("2025-02-01", 5), ("2025-01-02", 0))
    assert app.book_dividends(p, ev).empty
    assert app.total_return_metrics(app.add_total_return(p, ev))["n_dividends"] == 0


def test_two_dividends_on_one_day_are_summed_and_time_of_day_is_ignored(app):
    p = prices([100, 100, 90])
    p["begin"] = p["begin"] + pd.Timedelta(hours=10)  # время в ценах не мешает сопоставлению с датой отсечки
    out = app.add_total_return(p, events(("2025-01-03", 4), ("2025-01-03", 6)))
    assert out["div"].tolist() == [0.0, 0.0, 10.0] and out["tr"].iloc[-1] == pytest.approx(100.0)


def test_total_return_metrics_annualizes_only_from_three_months(app):
    short = app.add_total_return(prices([100, 110]))
    assert app.total_return_metrics(short)["tr_annual"] is None
    year = app.add_total_return(prices([100] * 365 + [121]))
    assert app.total_return_metrics(year)["tr_annual"] == pytest.approx(21.0, abs=0.1)
    assert app.total_return_metrics(prices([100]).assign(div=0.0, tr=100.0)) is None


# ---------- сравнение с индексом ----------

def random_walk(n, seed, scale=0.01):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, scale, n)))


def test_beta_and_correlation_of_a_leveraged_copy(app):
    """Бумага, которая каждый день меняется ровно вдвое сильнее индекса: бета 2, корреляция 1."""
    idx = random_walk(300, 1)
    r = np.diff(idx) / idx[:-1]
    stock = 50 * np.concatenate([[1.0], np.cumprod(1 + 2 * r)])
    m, frame = app.compare_with_benchmark(prices(stock), prices(idx))
    assert m["beta"] == pytest.approx(2.0) and m["corr"] == pytest.approx(1.0)
    assert m["days"] == 300 and frame["stock"].iloc[0] == pytest.approx(100) and frame["bench"].iloc[0] == pytest.approx(100)
    assert m["excess"] == pytest.approx(m["stock_total"] - m["bench_total"])


def test_independent_series_have_beta_near_zero(app):
    m, _ = app.compare_with_benchmark(prices(random_walk(1000, 2)), prices(random_walk(1000, 3)))
    assert abs(m["beta"]) < 0.15 and abs(m["corr"]) < 0.15


def test_comparison_uses_only_common_days(app):
    """Индекс торгуется не во все дни бумаги (например, выходные): доходности считаются по общим дням."""
    stock = prices([100, 102, 104, 106, 108, 110])
    bench = stock.iloc[[0, 2, 4, 5]].assign(close=[200, 208, 216, 220])
    m, frame = app.compare_with_benchmark(stock, bench, min_days=3)
    assert m["days"] == 4 and frame["begin"].tolist() == bench["begin"].tolist()
    assert m["stock_total"] == pytest.approx(10.0) and m["bench_total"] == pytest.approx(10.0) and m["excess"] == pytest.approx(0)
    assert frame["relative"].iloc[-1] == pytest.approx(100.0)


def test_relative_strength_rises_when_stock_outperforms(app):
    m, frame = app.compare_with_benchmark(prices([100, 110, 121]), prices([100, 100, 100]), min_days=3)
    assert frame["relative"].tolist() == pytest.approx([100, 110, 121]) and m["excess"] == pytest.approx(21.0)
    assert m["beta"] is None  # индекс не меняется: бета не определена


def test_too_few_common_days_gives_no_comparison(app, cfg):
    m, frame = app.compare_with_benchmark(prices([100, 101, 102]), prices([100, 101, 102]))
    assert m is None and frame.empty and cfg.BENCHMARK_MIN_DAYS > 3
    assert app.compare_with_benchmark(prices([100, 101]), pd.DataFrame(columns=["begin", "close"]))[0] is None


def test_beta_label(app):
    assert app.beta_label(1.5) == "сильнее рынка" and app.beta_label(0.5) == "слабее рынка"
    assert app.beta_label(1.0) == "примерно как рынок" and app.beta_label(None) == "н/д"


# ---------- сбор цен для режима сравнения ----------

@pytest.fixture
def services(app, monkeypatch):
    import importlib
    mod = importlib.import_module("salesanalyzer.services")
    idx = random_walk(120, 5)
    data = {"SBER": prices([100] * 60 + [90] * 60), "AAPL": prices(random_walk(120, 6)),
            "IMOEX": prices(idx), "MCFTR": prices(idx * 1.1), "^GSPC": prices(random_walk(120, 7)), "^SP500TR": prices(random_walk(120, 8))}
    monkeypatch.setattr(mod, "get_moex_candles", lambda t, years: data[t])
    monkeypatch.setattr(mod, "get_moex_index_candles", lambda t, years: data[t])
    monkeypatch.setattr(mod, "get_us_prices", lambda t, years: data[t])
    monkeypatch.setattr(mod, "get_us_dividend_events", lambda t, years: events())
    monkeypatch.setattr(mod, "load_dividends_ru", lambda t: events(("2025-03-02", 10)) if t == "SBER" else events())
    return mod


def test_collect_prices_adds_market_indexes(services):
    problems = []
    out = services.collect_prices(["SBER"], ["AAPL"], 1, problems, with_index=True)
    assert set(out["label"]) == {"SBER", "AAPL", "Индекс МосБиржи", "S&P 500"}
    assert set(out.loc[out["is_benchmark"], "label"]) == {"Индекс МосБиржи", "S&P 500"}
    assert (out.groupby("label")["index"].first() == 100).all() and problems == []
    table = services.benchmark_table(out, ["SBER"])
    assert table.set_index("Компания")["Индекс"].to_dict() == {"SBER": "Индекс МосБиржи", "AAPL": "S&P 500"}


def test_collect_prices_total_return_uses_dividends_and_total_return_indexes(services):
    problems = []
    out = services.collect_prices(["SBER"], ["AAPL"], 1, problems, with_index=True, total=True)
    sber = out[out["label"] == "SBER"]
    assert sber["index"].iloc[-1] == pytest.approx(100.0)  # гэп 100 → 90 компенсирован дивидендом 10
    assert {"Индекс МосБиржи полной доходности", "S&P 500 полной доходности"} <= set(out["label"])
    assert any("AAPL: нет дат выплаты дивидендов" in p for p in problems)


def test_collect_prices_without_index_and_only_russian(services):
    out = services.collect_prices(["SBER"], [], 1, [], with_index=True)
    assert set(out["label"]) == {"SBER", "Индекс МосБиржи"}
    out = services.collect_prices(["SBER"], [], 1, [])
    assert set(out["label"]) == {"SBER"} and not out["is_benchmark"].any()
    assert services.benchmark_table(out, ["SBER"]).empty
