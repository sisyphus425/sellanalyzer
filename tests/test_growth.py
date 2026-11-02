"""Рост к прошлому году, CAGR и «светофор»."""
import math

import pandas as pd
import pytest


def frame(years, values, col="revenue"):
    return pd.DataFrame({"year": years, col: values})


def test_growth_for_consecutive_years(app):
    g = app.add_growth(frame([2022, 2023, 2024], [100, 150, 135]))
    assert math.isnan(g["growth"].iloc[0])
    assert g["growth"].iloc[1] == pytest.approx(50.0)
    assert g["growth"].iloc[2] == pytest.approx(-10.0)


def test_growth_is_nan_when_year_is_missing(app):
    g = app.add_growth(frame([2023, 2025], [100, 150]))
    assert g["growth"].isna().all()


@pytest.mark.parametrize("values", [[0, 10], [-5, 10], [10, 0], [10, -3]])
def test_growth_requires_positive_values_in_both_years(app, values):
    g = app.add_growth(frame([2024, 2025], values))
    assert math.isnan(g["growth"].iloc[1])


def test_growth_sorts_years_first(app):
    g = app.add_growth(frame([2024, 2022, 2023], [135, 100, 150]))
    assert g["year"].tolist() == [2022, 2023, 2024]
    assert g["growth"].iloc[2] == pytest.approx(-10.0)


def test_growth_for_custom_column(app):
    g = app.add_growth(frame([2024, 2025], [10, 12], col="net_income"), "net_income")
    assert g["growth"].iloc[1] == pytest.approx(20.0)


def test_cagr_basic(app):
    assert app.calc_cagr(frame([2022, 2025], [100, 200])) == pytest.approx((2 ** (1 / 3) - 1) * 100)


def test_cagr_uses_first_and_last_year_when_there_is_a_gap(app):
    assert app.calc_cagr(frame([2023, 2025], [100, 150])) == pytest.approx((1.5 ** 0.5 - 1) * 100)


@pytest.mark.parametrize("years, values", [([2025], [100]), ([2024, 2025], [-10, 20]), ([2024, 2025], [10, 0]),
                                           ([2025, 2025], [10, 20])])
def test_cagr_is_none_when_not_defined(app, years, values):
    assert app.calc_cagr(frame(years, values)) is None


def test_cagr_for_custom_column(app):
    assert app.calc_cagr(frame([2024, 2025], [10, 11], col="net_income"), "net_income") == pytest.approx(10.0)


@pytest.mark.parametrize("value, name, icon", [
    (5, "Рост", "🟢"), (30, "Рост", "🟢"), (4.99, "Стагнация", "🟡"), (0, "Стагнация", "🟡"),
    (-4.99, "Стагнация", "🟡"), (-5, "Падение", "🔴"), (-40, "Падение", "🔴"),
    (float("nan"), "н/д", "⚪"), (None, "н/д", "⚪"),
])
def test_traffic_light(app, value, name, icon):
    assert app.signal_name(value) == name
    assert app.signal_icon(value) == icon


def test_known_growth_of_sber_revenue(app, ru_all):
    """Контрольные цифры по Сбербанку: рост 66,33% → 10,78% → 16,63%, CAGR 29,04%."""
    rev = ru_all[(ru_all["ticker"] == "SBER") & (ru_all["indicator"] == "revenue")].rename(columns={"value": "revenue"})
    g = app.add_growth(rev[["year", "revenue"]])
    assert g["growth"].round(2).tolist()[1:] == [66.33, 10.78, 16.63]
    assert app.calc_cagr(g) == pytest.approx(29.04, abs=0.01)
