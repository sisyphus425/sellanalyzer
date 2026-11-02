"""Прогноз показателей: методы, сценарии, проверка на прошлом и оценка надежности."""
import math

import numpy as np
import pandas as pd
import pytest

Y = np.array([2022, 2023, 2024])
V = np.array([100.0, 110.0, 121.0])  # ровный рост на 10% в год


# ---------- методы ----------

def test_linear_trend_extends_the_least_squares_line(app):
    y, v = np.array([2022, 2023, 2024, 2025]), np.array([10.0, 20.0, 30.0, 40.0])
    assert app.linear_trend(y, v, [2026, 2027]) == pytest.approx([50.0, 60.0])


def test_linear_trend_for_the_example_series(app):
    # прямая по трем точкам: наклон 10,5 в год, значение в 2024 по прямой 120,83
    assert app.linear_trend(Y, V, [2025]) == pytest.approx([120.8333333 + 10.5])


def test_linear_trend_needs_two_points(app):
    assert app.linear_trend(np.array([2025]), np.array([5.0]), [2026]) is None


def test_cagr_extrapolation_continues_the_average_growth(app):
    assert app.cagr_rate(Y, V) == pytest.approx(0.10)
    assert app.cagr_extrapolation(Y, V, [2025, 2026]) == pytest.approx([133.1, 146.41])


@pytest.mark.parametrize("values", [[-5.0, 10.0, 20.0], [10.0, 20.0, 0.0], [0.0, 5.0, 8.0], [50.0, -10.0, 20.0]])
def test_growth_methods_are_not_defined_for_series_with_non_positive_values(app, values):
    assert app.cagr_rate(Y, np.array(values)) is None
    assert app.cagr_extrapolation(Y, np.array(values), [2025]) is None
    assert app.damped_growth(Y, np.array(values), [2025]) is None


def test_mean_growth_uses_only_neighbouring_positive_years(app):
    assert app.mean_growth(np.array([2022, 2024, 2025]), np.array([100.0, 150.0, 165.0])) == pytest.approx(0.10)
    assert app.mean_growth(np.array([2022, 2023]), np.array([-1.0, 5.0])) is None


def test_damped_growth_shrinks_the_rate_every_year(app):
    # g = 10%, φ = 0.5: первый год +5%, второй +2,5%
    assert app.damped_growth(Y, V, [2025, 2026], damping=0.5) == pytest.approx([121 * 1.05, 121 * 1.05 * 1.025])


def test_damped_growth_without_damping_equals_cagr_for_steady_growth(app):
    assert app.damped_growth(Y, V, [2025], damping=1.0) == pytest.approx(app.cagr_extrapolation(Y, V, [2025]))


def test_damped_growth_uses_default_damping_from_config(app, cfg):
    first = app.damped_growth(Y, V, [2025])[0]
    assert first == pytest.approx(121 * (1 + 0.10 * cfg.FORECAST_DAMPING))


def test_methods_on_a_series_with_a_loss(app):
    y, v = np.array([2022, 2023, 2024]), np.array([50.0, -10.0, 20.0])
    res = app.run_methods(y, v, [2025])
    assert res["Линейный тренд"] is not None and res["CAGR"] is None and res["Затухающий рост"] is None


def test_clean_series_sorts_and_drops_gaps_and_duplicates(app):
    y, v = app.clean_series([2024, 2022, 2023, 2023, 2021], [3.0, 1.0, 2.0, 2.5, float("nan")])
    assert y.tolist() == [2022, 2023, 2024] and v.tolist() == [1.0, 2.5, 3.0]


# ---------- проверка на прошлом ----------

def test_holdout_of_a_perfect_exponential_series_has_no_cagr_error(app):
    y, v = np.array([2021, 2022, 2023, 2024]), np.array([100.0, 110.0, 121.0, 133.1])
    bt = app.holdout(y, v)
    cagr = bt[bt["Метод"] == "CAGR"]
    assert cagr["Год прогноза"].tolist() == [2023, 2024] and cagr["Ошибка, %"].abs().max() < 1e-9
    assert bt["Данные до"].tolist()[:3] == [2022, 2022, 2022]


def test_holdout_errors_have_the_right_sign_and_size(app):
    bt = app.holdout(np.array([2022, 2023, 2024]), np.array([100.0, 120.0, 100.0]))  # третий год: падение
    cagr = bt[bt["Метод"] == "CAGR"].iloc[0]
    assert cagr["Прогноз"] == pytest.approx(144.0) and cagr["Факт"] == 100.0 and cagr["Ошибка, %"] == pytest.approx(44.0)


def test_backtest_summary_per_method(app):
    bt = app.holdout(np.array([2021, 2022, 2023, 2024]), np.array([100.0, 110.0, 121.0, 150.0]))
    summary = app.backtest_summary(bt).set_index("Метод")
    assert set(summary.index) <= set(app.METHODS) and (summary["Проверок"] == 2).all()
    assert (summary["Средняя ошибка, %"] >= 0).all()
    assert app.typical_error(bt) == pytest.approx(bt["Ошибка, %"].abs().mean())
    assert app.typical_error(bt.iloc[0:0]) is None and app.backtest_summary(bt.iloc[0:0]).empty


def test_growth_spread(app):
    assert app.growth_spread(np.array([2022, 2023, 2024]), np.array([100.0, 110.0, 132.0])) == pytest.approx(
        float(np.std([10.0, 20.0], ddof=1)))
    assert app.growth_spread(np.array([2023, 2024]), np.array([100.0, 110.0])) is None


# ---------- сценарии ----------

def test_scenarios_use_median_as_base_and_methods_as_bounds(app):
    methods = {"A": [100.0], "B": [120.0], "C": [110.0]}
    sc = app.scenarios(methods, [2026], typical_err=None)
    assert sc.iloc[0]["Низкий"] == 100.0 and sc.iloc[0]["База"] == 110.0 and sc.iloc[0]["Высокий"] == 120.0


def test_band_is_widened_by_typical_error_and_grows_with_the_horizon(app):
    methods = {"A": [100.0, 100.0], "B": [100.0, 100.0]}
    sc = app.scenarios(methods, [2026, 2027], typical_err=10.0)
    assert sc.iloc[0]["Низкий"] == pytest.approx(90.0) and sc.iloc[0]["Высокий"] == pytest.approx(110.0)
    assert sc.iloc[1]["Низкий"] == pytest.approx(100 - 10 * math.sqrt(2)) and sc.iloc[1]["Высокий"] == pytest.approx(100 + 10 * math.sqrt(2))


def test_band_widening_is_capped(app):
    sc = app.scenarios({"A": [100.0]}, [2026], typical_err=500.0)
    assert sc.iloc[0]["Низкий"] == pytest.approx(40.0) and sc.iloc[0]["Высокий"] == pytest.approx(160.0)


def test_nonnegative_floor_and_unavailable_methods(app):
    methods = {"A": [-5.0], "B": [10.0], "C": [8.0], "D": None}  # линейный тренд ушел ниже нуля
    sc = app.scenarios(methods, [2026], typical_err=None, nonnegative=True)
    assert sc.iloc[0]["Низкий"] == 0.0 and sc.iloc[0]["Методов"] == 3 and sc.iloc[0]["База"] == 8.0
    assert app.scenarios(methods, [2026], typical_err=None).iloc[0]["Низкий"] == -5.0
    assert app.scenarios({"A": None}, [2026], None).empty


# ---------- полный прогноз и надежность ----------

def test_full_forecast_of_a_steady_series(app):
    res = app.forecast_series([2021, 2022, 2023, 2024], [100.0, 110.0, 121.0, 133.1], horizon=2)
    assert res["targets"] == [2025, 2026] and list(res["scenarios"]["Год"]) == [2025, 2026]
    first = res["scenarios"].iloc[0]
    assert first["Низкий"] <= first["База"] <= first["Высокий"] and first["База"] == pytest.approx(146.41, rel=0.05)
    assert res["reliability"]["label"] == "средняя"


def test_forecast_is_not_built_for_short_series(app):
    res = app.forecast_series([2024, 2025], [100.0, 120.0], horizon=1)
    assert res["scenarios"].empty and res["targets"] == [] and res["reliability"]["label"] == "недостаточно данных"
    assert app.methods_table(res).empty


def test_three_points_is_always_low_reliability(app):
    res = app.forecast_series([2023, 2024, 2025], [100.0, 110.0, 121.0])
    assert res["reliability"]["label"] == "низкая" and not res["scenarios"].empty


def test_unstable_series_gets_low_reliability_with_reasons(app):
    res = app.forecast_series([2021, 2022, 2023, 2024], [100.0, 300.0, 120.0, 400.0])
    assert res["reliability"]["label"] == "низкая"
    text = " ".join(res["reliability"]["reasons"])
    assert "ошибались" in text and "разброс" in text


def test_estimate_in_last_year_lowers_reliability(app):
    res = app.forecast_series([2021, 2022, 2023, 2024], [100.0, 110.0, 121.0, 133.1], estimate_years=[2024])
    assert res["reliability"]["label"] == "низкая" and any("расчётное" in r for r in res["reliability"]["reasons"])
    res = app.forecast_series([2021, 2022, 2023, 2024], [100.0, 110.0, 121.0, 133.1], estimate_years=[2022])
    assert res["reliability"]["label"] == "средняя"


def test_series_with_loss_still_forecasts_with_remaining_methods(app):
    res = app.forecast_series([2022, 2023, 2024, 2025], [50.0, -10.0, 20.0, 30.0])
    assert not res["scenarios"].empty and res["reliability"]["label"] == "низкая"
    assert any("отрицательные" in r for r in res["reliability"]["reasons"])


def test_forecast_never_labels_reliability_high(app):
    res = app.forecast_series(range(2015, 2026), [100.0 * 1.05 ** i for i in range(11)])
    assert res["reliability"]["label"] in ("низкая", "средняя")


def test_methods_table_has_one_column_per_forecast_year(app):
    res = app.forecast_series([2021, 2022, 2023, 2024], [100.0, 110.0, 121.0, 133.1], horizon=2)
    table = app.methods_table(res)
    assert table["Метод"].tolist() == list(app.METHODS) and {"2025", "2026"} <= set(table.columns)
    assert "Средняя ошибка на прошлом, %" in table.columns


# ---------- сводка по компаниям и реальные данные ----------

def test_forecast_overview_for_several_companies(app):
    series = {"A": ([2021, 2022, 2023, 2024], [100.0, 110.0, 121.0, 133.1], []),
              "B": ([2023, 2024], [10.0, 12.0], [])}
    out = app.forecast_overview(series, horizon=1).set_index("Компания")
    assert out.loc["A", "Год прогноза"] == 2025 and 6.0 < out.loc["A", "Рост базы к последнему году, %"] < 11.0
    assert out.loc["B", "Надежность"] == "недостаточно данных" and pd.isna(out.loc["B", "База"])


def test_real_data_sber_revenue_forecast(app, ru_all):
    rows = ru_all[(ru_all["ticker"] == "SBER") & (ru_all["indicator"] == "revenue")].sort_values("year")
    res = app.forecast_series(rows["year"], rows["value"], horizon=1, nonnegative=True)
    methods = res["methods"]
    assert methods["Линейный тренд"][0] == pytest.approx(4891, abs=1) and methods["CAGR"][0] == pytest.approx(5277, abs=1)
    # проверка на прошлом (прогноз 2025 по 2022–2024): линейный +9%, CAGR +16%
    bt = res["backtest"]
    last = bt[bt["Год прогноза"] == 2025].set_index("Метод")["Ошибка, %"]
    assert last["Линейный тренд"] == pytest.approx(9.0, abs=0.5) and last["CAGR"] == pytest.approx(16.0, abs=0.5)


def test_real_data_every_company_gets_a_sensible_forecast(app, ru_all):
    for ticker in ("SBER", "GAZP", "YDEX", "T", "GMKN"):
        rows = ru_all[(ru_all["ticker"] == ticker) & (ru_all["indicator"] == "revenue")].sort_values("year")
        res = app.forecast_series(rows["year"], rows["value"], horizon=2, nonnegative=True)
        sc = res["scenarios"]
        assert len(sc) == 2 and (sc["Низкий"] <= sc["База"]).all() and (sc["База"] <= sc["Высокий"]).all(), ticker
        assert (sc["Низкий"] >= 0).all() and np.isfinite(sc[["Низкий", "База", "Высокий"]].to_numpy()).all(), ticker
    lk = ru_all[(ru_all["ticker"] == "LKOH") & (ru_all["indicator"] == "revenue")]
    assert app.forecast_series(lk["year"], lk["value"])["reliability"]["label"] == "недостаточно данных"  # у Лукойла 2 значения
