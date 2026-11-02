"""Метрики цены: доходность, волатильность, просадка, скользящие средние."""
import numpy as np
import pandas as pd
import pytest


def prices(values, start="2025-01-01", freq="D"):
    return pd.DataFrame({"begin": pd.date_range(start, periods=len(values), freq=freq), "close": values})


def test_max_drawdown_dates_and_current_drawdown(app):
    m = app.price_metrics(app.add_price_indicators(prices([100, 120, 90, 110.0])))
    assert m["max_dd"] == pytest.approx(-25.0)
    assert m["dd_peak"] == pd.Timestamp("2025-01-02") and m["dd_trough"] == pd.Timestamp("2025-01-03")
    assert m["cur_dd"] == pytest.approx((110 / 120 - 1) * 100)
    assert m["total"] == pytest.approx(10.0)


def test_steady_growth_has_no_volatility_and_no_drawdown(app):
    p = prices(list(100 * 1.001 ** np.arange(500)), start="2023-01-02", freq="B")
    m = app.price_metrics(app.add_price_indicators(p))
    assert m["vol"] == pytest.approx(0.0, abs=1e-9) and m["max_dd"] == 0
    assert m["total"] == pytest.approx((1.001 ** 499 - 1) * 100)
    assert app.trend_label(m) == "🟢 выше MA200"


def test_volatility_matches_numpy_and_moving_averages_match_means(app):
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.02, 600))
    p = app.add_price_indicators(prices(list(close), start="2023-01-02", freq="B"))
    m = app.price_metrics(p)
    expected_vol = pd.Series(close).pct_change().dropna().std() * np.sqrt(252) * 100
    assert m["vol"] == pytest.approx(expected_vol)
    assert m["ma50"] == pytest.approx(close[-50:].mean()) and m["ma200"] == pytest.approx(close[-200:].mean())
    assert m["max_dd"] == pytest.approx(((close / np.maximum.accumulate(close) - 1) * 100).min())


def test_moving_averages_are_empty_until_window_is_full(app):
    p = app.add_price_indicators(prices(list(range(1, 101))))
    assert p["ma50"].iloc[:49].isna().all() and not np.isnan(p["ma50"].iloc[49])
    assert p["ma200"].isna().all()
    assert app.trend_label(app.price_metrics(p)) == "н/д"


def test_not_enough_data_gives_none_and_short_periods_have_no_annualized_values(app):
    assert app.price_metrics(app.add_price_indicators(prices([100.0]))) is None
    m = app.price_metrics(app.add_price_indicators(prices([100 + i for i in range(10)])))
    assert m["vol"] is None and m["annual"] is None


def test_price_below_ma200_is_a_downtrend(app):
    close = [200.0] * 250 + [100.0] * 5
    assert app.trend_label(app.price_metrics(app.add_price_indicators(prices(close)))) == "🔴 ниже MA200"


def test_metrics_table_for_several_companies(app):
    up = prices(list(100 * 1.001 ** np.arange(300))).assign(label="UP")
    down = prices(list(100 * 0.999 ** np.arange(300))).assign(label="DOWN")
    table = app.price_metrics_table(pd.concat([up, down], ignore_index=True)).set_index("Компания")
    assert table.loc["UP", "Доходность за период, %"] > 0 > table.loc["DOWN", "Доходность за период, %"]
    assert table.loc["UP", "Макс. просадка, %"] == 0 and table.loc["DOWN", "Макс. просадка, %"] < 0


def test_last_close_returns_latest_row(app):
    price, date = app.last_close(prices([10.0, 11.0, 12.5]))
    assert price == 12.5 and date == pd.Timestamp("2025-01-03")
    assert app.last_close(pd.DataFrame()) == (None, None) and app.last_close(None) == (None, None)
