"""Таблица по годам, маржа и вспомогательные функции прибыльности."""
import math

import pandas as pd
import pytest


def test_make_wide_pivots_indicators_by_year(app, mk_rows):
    rows = mk_rows(("X", 2024, "revenue", 100.0, "млрд ₽"), ("X", 2025, "revenue", 120.0, "млрд ₽"),
                   ("X", 2025, "net_income", 12.0, "млрд ₽"), ("X", 2025, "roe", 20.0, "%"))
    w, unit, mixed = app.make_wide(rows)
    assert unit == "млрд ₽" and mixed is False
    assert w["year"].tolist() == [2024, 2025]
    assert w.loc[w["year"] == 2025, "net_income"].iloc[0] == 12.0
    assert math.isnan(w.loc[w["year"] == 2024, "net_income"].iloc[0])
    for col in ("ebitda", "dps", "shares"):  # отсутствующие показатели присутствуют как пустые колонки
        assert col in w.columns and w[col].isna().all()


def test_make_wide_on_empty_input(app):
    w, unit, mixed = app.make_wide(pd.DataFrame())
    assert w.empty and unit == "" and mixed is False


def test_make_wide_flags_mixed_money_units(app, mk_rows):
    rows = mk_rows(("X", 2025, "revenue", 100.0, "млрд ₽"), ("X", 2025, "net_income", 5.0, "млн $"))
    assert app.make_wide(rows)[2] is True


def test_per_share_and_share_count_units_do_not_count_as_mixed(app, mk_rows):
    rows = mk_rows(("X", 2025, "revenue", 100.0, "млрд ₽"), ("X", 2025, "roe", 20.0, "%"),
                   ("X", 2025, "dps", 10.0, "₽ на акцию"), ("X", 2025, "shares", 500.0, "млн шт."))
    w, unit, mixed = app.make_wide(rows)
    assert mixed is False and unit == "млрд ₽"


def test_make_wide_marks_years_with_estimates(app, mk_rows):
    rows = mk_rows(("X", 2024, "revenue", 100.0, "млрд ₽", False), ("X", 2025, "revenue", 110.0, "млрд ₽", True))
    w, _, _ = app.make_wide(rows)
    assert w["has_estimate"].tolist() == [False, True]


def test_margins(app):
    w = pd.DataFrame({"year": [2024, 2025], "revenue": [200.0, 100.0], "net_income": [20.0, -10.0], "ebitda": [60.0, 25.0]})
    m = app.add_margins(w)
    assert m["net_margin"].tolist() == pytest.approx([10.0, -10.0])
    assert m["ebitda_margin"].tolist() == pytest.approx([30.0, 25.0])


def test_margin_is_nan_when_revenue_is_not_positive(app):
    w = pd.DataFrame({"year": [2025], "revenue": [0.0], "net_income": [5.0], "ebitda": [5.0]})
    m = app.add_margins(w)
    assert math.isnan(m["net_margin"].iloc[0]) and math.isnan(m["ebitda_margin"].iloc[0])


def test_margins_can_be_disabled(app):
    w = pd.DataFrame({"year": [2025], "revenue": [100.0], "net_income": [10.0], "ebitda": [20.0]})
    m = app.add_margins(w, enabled=False)
    assert m["net_margin"].isna().all() and m["ebitda_margin"].isna().all()


def test_latest_value(app):
    w = pd.DataFrame({"year": [2023, 2024, 2025], "roe": [10.0, 12.0, float("nan")]})
    assert app.latest_value(w, "roe") == (2024, 12.0)
    assert app.latest_value(pd.DataFrame({"year": [2025], "roe": [float("nan")]}), "roe") is None


@pytest.mark.parametrize("value, unit, expected", [(1500.0, "млрд ₽", "1,500 млрд ₽"), (12.34, "млрд ₽", "12.3 млрд ₽"),
                                                   (50.0, "", "50.0")])
def test_fmt_value(app, value, unit, expected):
    assert app.fmt_value(value, unit) == expected


def test_known_margins_of_real_companies(app, ru_all):
    """Контрольные значения: EBITDA-маржа Норникеля 2025 ≈ 41,2%, чистая маржа Газпрома 2023 отрицательная."""
    def margins(ticker):
        w, unit, mixed = app.make_wide(ru_all[ru_all["ticker"] == ticker])
        assert mixed is False
        return app.add_margins(w).set_index("year")

    assert margins("GMKN").loc[2025, "ebitda_margin"] == pytest.approx(5668 / 13763 * 100)
    assert margins("GMKN").loc[2025, "ebitda_margin"] == pytest.approx(41.2, abs=0.1)
    assert margins("GAZP").loc[2023, "net_margin"] == pytest.approx(-629.1 / 8541.8 * 100)
