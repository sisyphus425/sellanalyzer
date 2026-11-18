"""Лаборатория моделей цены: честное сравнение с наивным прогнозом на синтетических рядах с известными свойствами."""
import numpy as np
import pandas as pd
import pytest


def prices_from_returns(r, p0=100.0):
    close = p0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return pd.DataFrame({"begin": pd.date_range("2018-01-01", periods=len(close), freq="B"), "close": close})


def random_walk(n=1500, seed=1, sigma=0.01):
    return prices_from_returns(np.random.default_rng(seed).normal(0, sigma, n))


def ar1_series(n=1500, seed=2, phi=0.5, sigma=0.01):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, sigma)
    return prices_from_returns(r)


def row(summary, model):
    return summary[summary["Модель"] == model].iloc[0]


def test_naive_forecasts_zero_return(app):
    lp = np.log(random_walk(300)["close"].to_numpy())
    assert app.FORECASTERS[app.NAIVE](lp, 5) == 0.0


def test_drift_is_mean_return_times_horizon(app):
    lp = np.log(100 * 1.001 ** np.arange(300))
    assert app.forecast_drift(lp, 10) == pytest.approx(10 * np.log(1.001))


def test_ar_recovers_momentum_on_ar1(app):
    lp = np.log(ar1_series(3000, phi=0.6)["close"].to_numpy())
    r = np.diff(lp)
    assert app.forecast_ar(lp, 1) == pytest.approx(0.6 * r[-1], abs=0.004)


def test_ar_multi_step_decays(app):
    lp = np.log(ar1_series(3000, phi=0.6)["close"].to_numpy())
    r = np.diff(lp)[-1]
    h5 = app.forecast_ar(lp, 5)
    assert abs(h5) < abs(r) * 0.6 / (1 - 0.6) * 1.2 + 0.003 and np.sign(h5) == np.sign(r)


def test_arima_tracks_ar_signal(app):
    lp = np.log(ar1_series(3000, phi=0.6)["close"].to_numpy())
    r = np.diff(lp)[-1]
    f = app.forecast_arima(lp, 1)
    assert np.sign(f) == np.sign(r) and abs(f) > 0.2 * abs(r)


def test_short_history_returns_zero(app):
    lp = np.log(random_walk(20)["close"].to_numpy())
    for fn in (app.forecast_ar, app.forecast_arima, app.forecast_ridge):
        assert fn(lp, 1) == 0.0


def test_walk_forward_uses_only_the_past(app):
    """Меняем будущее после даты проверки: прогноз на эту дату не должен измениться."""
    base = random_walk(500)
    a = app.walk_forward(base, 1, [app.AR, app.RIDGE], min_train=300, max_origins=5)
    changed = base.copy()
    changed.loc[changed.index[-3:], "close"] *= 3          # правим только последние точки
    b = app.walk_forward(changed, 1, [app.AR, app.RIDGE], min_train=300, max_origins=5)
    first = a["Дата"].iloc[0]
    pa = a[a["Дата"] == first]["Прогноз доходности, %"].tolist()
    pb = b[b["Дата"] == first]["Прогноз доходности, %"].tolist()
    assert pa == pytest.approx(pb)


def test_walk_forward_origins_do_not_overlap(app):
    wf = app.walk_forward(random_walk(600), 21, [app.NAIVE], min_train=300)
    dates = wf["Дата"].tolist()
    gaps = pd.Series(dates).diff().dropna().dt.days
    assert (gaps >= 21 * 7 / 5 - 3).all()       # шаг не меньше 21 торгового дня


def test_walk_forward_actual_is_future_return(app):
    p = random_walk(400)
    wf = app.walk_forward(p, 5, [app.NAIVE], min_train=300)
    r = wf.iloc[0]
    i = p.index[p["begin"] == r["Дата"]][0]
    assert r["Факт доходности, %"] == pytest.approx((np.log(p["close"][i + 5]) - np.log(p["close"][i])) * 100)


def test_walk_forward_empty_when_too_short(app):
    assert app.walk_forward(random_walk(100), 1).empty


def test_max_origins_limits_rows(app):
    wf = app.walk_forward(random_walk(900), 1, [app.NAIVE], min_train=300, max_origins=50)
    assert len(wf) == 50


def test_random_walk_nobody_beats_naive(app):
    wf = app.walk_forward(random_walk(1500, seed=3), 1, app.MODELS, min_train=300)
    s = app.summarize(wf, 1)
    assert (s["Вывод"] == app.VERDICT_BETTER).sum() == 0
    assert "Ни одна модель" in app.overall_conclusion(s)


def test_naive_ratio_is_one_and_marked_base(app):
    s = app.summarize(app.walk_forward(random_walk(700), 1, [app.NAIVE, app.DRIFT], min_train=300), 1)
    n = row(s, app.NAIVE)
    assert n["RMSE к наивному"] == pytest.approx(1.0) and n["Вывод"] == app.VERDICT_BASE


def test_ar_beats_naive_on_momentum_series(app):
    wf = app.walk_forward(ar1_series(1500, phi=0.6), 1, [app.NAIVE, app.AR], min_train=300)
    s = app.summarize(wf, 1)
    ar = row(s, app.AR)
    assert ar["RMSE к наивному"] < 1 and ar["Вывод"] == app.VERDICT_BETTER
    assert ar["Знак угадан, %"] > 55
    assert "Заметно точнее" in app.overall_conclusion(s)


def test_bad_model_is_flagged_worse(app):
    """Модель с заведомо вредным прогнозом (противоположный знак сильного сигнала) помечается «хуже»."""
    p = ar1_series(1500, phi=0.6)
    wf = app.walk_forward(p, 1, [app.NAIVE, app.AR], min_train=300)
    wf.loc[wf["Модель"] == app.AR, "Прогноз доходности, %"] *= -3
    assert row(app.summarize(wf, 1), app.AR)["Вывод"] == app.VERDICT_WORSE


def test_few_origins_gives_no_verdict(app):
    wf = app.walk_forward(random_walk(400), 21, [app.NAIVE, app.AR], min_train=300)
    s = app.summarize(wf, 21)
    assert row(s, app.AR)["Вывод"] == app.VERDICT_FEW
    assert "мало" in app.overall_conclusion(s).lower()


@pytest.mark.parametrize("ratio, t, n, expected", [
    (0.9, 2.5, 100, "лучше наивного"), (0.9, 1.9, 100, "не отличается от наивного"),
    (1.1, -2.5, 100, "хуже наивного"), (1.1, -1.0, 100, "не отличается от наивного"),
    (0.9, 5.0, 10, "мало проверок"), (1.1, 2.5, 100, "не отличается от наивного"),
])
def test_verdict_rules(app, ratio, t, n, expected):
    assert app.verdict(ratio, t, n) == expected


def test_latest_forecasts_price_matches_return(app):
    p = random_walk(400)
    lf = app.latest_forecasts(p, 5)
    for model, (ret, price) in lf.items():
        assert price == pytest.approx(p["close"].iloc[-1] * np.exp(ret / 100))
    assert lf[app.NAIVE][0] == 0.0 and lf[app.NAIVE][1] == pytest.approx(p["close"].iloc[-1])


def test_summary_empty_without_data(app):
    assert app.summarize(pd.DataFrame(columns=app.ORIGIN_COLS), 1).empty
    assert "Недостаточно" in app.overall_conclusion(app.summarize(pd.DataFrame(columns=app.ORIGIN_COLS), 1))


def test_arima_multi_step_decays_geometrically(app):
    """Для AR-подобного ряда (phi=0.6) прогноз на 5 шагов ≈ прогноз на 1 шаг × (1 - phi^5)/(1 - phi) ≈ 2,2 (среднее по трём рядам)."""
    ratios = []
    for seed in (7, 8, 9):
        lp = np.log(ar1_series(4000, phi=0.6, seed=seed)["close"].to_numpy())
        ratios.append(app.forecast_arima(lp, 5) / app.forecast_arima(lp, 1))
    assert 1.8 < np.mean(ratios) < 2.7


def test_ridge_precomputed_features_give_same_result(app):
    lp = np.log(ar1_series(600, phi=0.3, seed=11)["close"].to_numpy())
    assert app.forecast_ridge(lp, 5) == pytest.approx(app.forecast_ridge(lp, 5, feats=app.feature_matrix(lp)))
    cut = lp[:450]
    assert app.forecast_ridge(cut, 5) == pytest.approx(app.forecast_ridge(cut, 5, feats=app.feature_matrix(lp)))


def test_features_at_date_do_not_depend_on_future(app):
    lp = np.log(random_walk(300, seed=5)["close"].to_numpy())
    a = app.feature_matrix(lp)[200]
    b = app.feature_matrix(lp[:201])[200]
    assert a == pytest.approx(b)
