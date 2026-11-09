"""Диапазон будущей цены (Монте-Карло) и проверка калибровки на прошлом."""
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest


def make_prices(n=756, seed=0, sigma=0.015, drift=0.0, start="2023-10-17", log_drift=None):
    """Синтетическая цена: логарифмические доходности ~ N(m, sigma). По умолчанию ожидаемая цена не меняется."""
    m = (drift - 0.5 * sigma ** 2) if log_drift is None else log_drift
    r = np.random.default_rng(seed).normal(m, sigma, n)
    return pd.DataFrame({"begin": pd.bdate_range(start, periods=n), "close": 100 * np.exp(np.cumsum(r))})


# ---------- подготовка данных и оценка параметров ----------

def test_clean_prices_drops_bad_rows_and_sorts(app):
    df = pd.DataFrame({"begin": pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02", "2025-01-06"]),
                       "close": [12.0, 10.0, -1.0, float("nan")]})
    out = app.clean_prices(df)
    assert out["close"].tolist() == [10.0, 12.0] and out["begin"].is_monotonic_increasing


def test_log_returns(app):
    assert app.log_returns([100.0, 110.0, 99.0]) == pytest.approx([np.log(1.1), np.log(0.9)])


def test_ewma_variance_of_constant_returns_and_recency_weighting(app):
    assert app.ewma_variance([0.01] * 100) == pytest.approx(1e-4)
    calm_then_wild = np.r_[np.full(300, 0.005), np.full(30, 0.03)] * np.tile([1, -1], 165)
    wild_then_calm = calm_then_wild[::-1]
    full = float(np.var(calm_then_wild, ddof=1))
    assert app.ewma_variance(calm_then_wild) > full > app.ewma_variance(wild_then_calm)


def test_estimate_params_for_both_drift_modes(app):
    r = np.random.default_rng(1).normal(0.001, 0.02, 500)
    zero = app.estimate_params(r, app.VOL_FULL, app.DRIFT_ZERO)
    hist = app.estimate_params(r, app.VOL_FULL, app.DRIFT_HIST)
    assert zero["sigma"] == pytest.approx(np.std(r, ddof=1)) and zero["mu"] == pytest.approx(-0.5 * zero["sigma"] ** 2)
    assert hist["mu"] == pytest.approx(np.mean(r)) and zero["sigma_annual"] == pytest.approx(zero["sigma"] * np.sqrt(252))
    assert app.estimate_params(r, app.VOL_EWMA)["sigma"] == pytest.approx(np.sqrt(app.ewma_variance(r)))


# ---------- моделирование ----------

def test_not_enough_data_gives_none(app):
    assert app.price_range(make_prices(50), 63) is None
    assert app.price_range(make_prices(1), 63) is None
    assert app.price_range(pd.DataFrame({"begin": [], "close": []}), 63) is None


def test_without_volatility_the_range_collapses_to_one_value(app):
    prices = pd.DataFrame({"begin": pd.bdate_range("2024-01-01", periods=300), "close": 100 * 1.001 ** np.arange(300)})
    res = app.price_range(prices, 21, drift_mode=app.DRIFT_HIST, vol_mode=app.VOL_FULL)
    expected = prices["close"].iloc[-1] * 1.001 ** 21
    assert res["terminal"] == pytest.approx(np.full(5000, expected), rel=1e-6)
    assert res["metrics"]["prob_below_current"] == 0.0 and res["metrics"]["prob_touch_drop_10"] == 0.0
    flat = app.price_range(prices, 21, drift_mode=app.DRIFT_ZERO, vol_mode=app.VOL_FULL)
    assert flat["terminal"] == pytest.approx(np.full(5000, prices["close"].iloc[-1]), rel=1e-6)


def test_gbm_terminal_quantiles_match_the_analytic_formula(app):
    prices = make_prices(seed=3)
    res = app.price_range(prices, 63, vol_mode=app.VOL_FULL, drift_mode=app.DRIFT_ZERO)
    p, nd = res["params"], NormalDist()
    for q in (0.10, 0.50, 0.90):
        expected = res["p0"] * np.exp(p["mu"] * 63 + p["sigma"] * np.sqrt(63) * nd.inv_cdf(q))
        assert np.quantile(res["terminal"], q) == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("model", ["gbm", "bootstrap"])
def test_zero_drift_keeps_the_expected_price_equal_to_current(app, model):
    # высокая волатильность и длинный горизонт: без поправки на дрейф ожидаемая цена сместилась бы почти на 10%
    res = app.price_range(make_prices(seed=5, sigma=0.03), 252, model=model, vol_mode=app.VOL_FULL, drift_mode=app.DRIFT_ZERO, n_sim=20000)
    assert res["terminal"].mean() == pytest.approx(res["p0"], rel=0.04)


def test_probability_below_current_matches_theory_for_zero_drift(app):
    res = app.price_range(make_prices(seed=5), 63, vol_mode=app.VOL_FULL, drift_mode=app.DRIFT_ZERO, n_sim=20000)
    s_h = res["params"]["sigma"] * np.sqrt(63)
    assert res["metrics"]["prob_below_current"] == pytest.approx(NormalDist().cdf(s_h / 2), abs=0.02)


def test_bootstrap_spread_matches_history(app):
    prices = make_prices(seed=2)
    res = app.price_range(prices, 63, model=app.MODEL_BOOTSTRAP, n_sim=20000)
    sigma = np.std(app.log_returns(prices["close"]), ddof=1)
    assert np.std(np.log(res["terminal"] / res["p0"])) == pytest.approx(sigma * np.sqrt(63), rel=0.05)


def test_historical_drift_shifts_the_median_up_for_a_rising_series(app):
    prices = make_prices(seed=4, log_drift=0.002)
    up = app.price_range(prices, 63, drift_mode=app.DRIFT_HIST)
    flat = app.price_range(prices, 63, drift_mode=app.DRIFT_ZERO)
    assert up["metrics"]["median"] > flat["metrics"]["median"] and up["metrics"]["median"] > up["p0"]


def test_results_are_reproducible_with_a_seed(app):
    prices = make_prices(seed=6)
    a, b = app.price_range(prices, 63), app.price_range(prices, 63)
    c = app.price_range(prices, 63, seed=1)
    assert np.array_equal(a["terminal"], b["terminal"]) and not np.array_equal(a["terminal"], c["terminal"])
    assert len(app.price_range(prices, 63, n_sim=100)["terminal"]) == 100


def test_quantile_table_dates_and_widening(app):
    prices = make_prices(seed=7)
    res = app.price_range(prices, 126)
    q = res["quantiles"]
    assert list(q.columns) == ["День", "p5", "p10", "p25", "p50", "p75", "p90", "p95"] and len(q) == 126
    assert (q[["p5", "p10", "p25", "p50", "p75", "p90", "p95"]].diff(axis=1).iloc[:, 1:] >= 0).all().all()  # p5 <= p10 <= ... <= p95
    width = q["p90"] - q["p10"]
    assert width.iloc[-1] > width.iloc[20] > width.iloc[0]
    dates = res["dates"]
    assert len(dates) == 126 and dates[0] > prices["begin"].iloc[-1] and (dates.day_of_week < 5).all()


def test_risk_metrics_on_a_known_sample(app):
    m = app.risk_metrics(100.0, [80, 90, 100, 110, 120], [70, 85, 95, 100, 100])
    assert m["prob_below_current"] == pytest.approx(0.4) and m["prob_end_drop_10"] == pytest.approx(0.4)
    assert m["prob_end_drop_20"] == pytest.approx(0.2) and m["prob_touch_drop_10"] == pytest.approx(0.4)
    assert m["prob_touch_drop_20"] == pytest.approx(0.2) and m["median"] == 100.0
    assert m["var95_pct"] == pytest.approx(18.0) and m["cvar95_pct"] == pytest.approx(20.0)  # 5%-квантиль = 82


def test_touching_a_level_is_at_least_as_likely_as_ending_below_it(app):
    m = app.price_range(make_prices(seed=8), 126)["metrics"]
    assert m["prob_touch_drop_10"] >= m["prob_end_drop_10"] and m["prob_touch_drop_20"] >= m["prob_end_drop_20"]
    assert m["cvar95_pct"] >= m["var95_pct"] > 0


# ---------- проверка калибровки на прошлом ----------

@pytest.mark.parametrize("model", ["gbm", "bootstrap"])
def test_calibration_is_close_to_nominal_when_the_model_is_correct(app, model):
    """Если цены действительно подчиняются модели, факт попадает в 80%-коридор примерно в 80% случаев."""
    table, summary = app.calibrate(make_prices(n=6000, seed=0), 63, model=model)
    assert summary["n"] == len(table) > 200
    assert summary["coverage80"] == pytest.approx(0.80, abs=0.07) and summary["coverage50"] == pytest.approx(0.50, abs=0.07)
    assert summary["below_median"] == pytest.approx(0.50, abs=0.07) and summary["width80_pct"] > 0
    assert app.calibration_verdict(summary) == "в пределах разброса"
    assert summary["n_eff"] == pytest.approx(summary["n"] * 21 / 63) and summary["margin80"] == pytest.approx(2 * np.sqrt(0.16 / summary["n_eff"]))


def test_calibration_detects_a_model_that_is_too_confident(app):
    """Волатильность выросла в 4 раза после обучения: коридоры по старой волатильности слишком узкие, покрытие заметно падает."""
    r = np.r_[np.random.default_rng(1).normal(0, 0.005, 3000), np.random.default_rng(2).normal(0, 0.02, 3000)]
    prices = pd.DataFrame({"begin": pd.bdate_range("2000-01-03", periods=6000), "close": 100 * np.exp(np.cumsum(r))})
    _, summary = app.calibrate(prices, 63, vol_mode=app.VOL_FULL)
    assert summary["coverage80"] < 0.75 and summary["coverage50"] < 0.45
    assert app.calibration_verdict(summary) == "отклоняется"


def test_a_short_history_cannot_judge_the_model(app):
    """На трехлетней истории при горизонте 12 месяцев независимая проверка по сути одна: вывод невозможен."""
    _, summary = app.calibrate(make_prices(n=756, seed=1), 252)
    assert summary["n_eff"] == pytest.approx(1.0) and app.calibration_verdict(summary) == "мало проверок"


def test_calibration_uses_only_past_data(app):
    prices = make_prices(n=700, seed=9)
    table, _ = app.calibrate(prices, 63, vol_mode=app.VOL_FULL)
    changed = prices.copy()
    changed.loc[400:, "close"] = changed.loc[400:, "close"] * 5  # меняем только будущее относительно ранних дат
    table2, _ = app.calibrate(changed, 63, vol_mode=app.VOL_FULL)
    early = table["Дата прогноза"] < prices["begin"].iloc[400]
    cols = ["p10", "p25", "p50", "p75", "p90"]
    assert early.sum() > 5 and np.allclose(table.loc[early, cols], table2.loc[table2["Дата прогноза"] < prices["begin"].iloc[400], cols])


def test_calibration_structure_and_edge_cases(app):
    table, summary = app.calibrate(make_prices(n=800, seed=1), 63)
    assert list(table.columns) == app.CALIBRATION_COLS and table["В 80%-коридоре"].dtype == bool
    assert (table["p10"] <= table["p50"]).all() and (table["p50"] <= table["p90"]).all()
    short_table, short = app.calibrate(make_prices(n=200, seed=1), 63)
    assert short_table.empty and short["n"] == 0 and app.calibration_verdict(short) == "мало проверок"


@pytest.mark.parametrize("n, n_eff, c80, c50, verdict", [
    (0, 0.0, None, None, "мало проверок"), (10, 3.0, 0.8, 0.5, "мало проверок"),
    (100, 33.0, 0.78, 0.52, "в пределах разброса"), (100, 33.0, 0.55, 0.50, "отклоняется"), (100, 33.0, 0.80, 0.80, "отклоняется"),
    (100, 33.0, 0.80 - 0.10, 0.50, "в пределах разброса"),  # допустимое отклонение при 33 независимых проверках около ±14 п.п.
])
def test_calibration_verdict(app, n, n_eff, c80, c50, verdict):
    assert app.calibration_verdict({"n": n, "n_eff": n_eff, "coverage80": c80, "coverage50": c50}) == verdict


def test_settings_in_config_are_sane(cfg):
    assert cfg.MC_SIMULATIONS >= 1000 and cfg.MC_MIN_RETURNS >= 20 and 0 < cfg.MC_EWMA_LAMBDA < 1
    assert cfg.MC_CALIBRATION_STEP >= 1 and cfg.MC_CALIBRATION_MIN_TRAIN >= cfg.MC_MIN_RETURNS
    assert all(m * cfg.MC_DAYS_PER_MONTH > 0 for m in cfg.MC_HORIZON_MONTHS)
