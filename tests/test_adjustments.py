"""Пересчет из долларов в рубли и поправка на инфляцию."""
import math

import pytest

MACRO = {"usd_rub": {2024: 80.0, 2025: 100.0}, "cpi": {2024: 10.0, 2025: 10.0}}


def test_price_levels_use_mid_year_geometric_mean(app):
    levels = app.price_levels({2022: 10.0, 2023: 10.0})
    assert levels[2022] == pytest.approx(1.1 ** 0.5)
    assert levels[2023] == pytest.approx((1.1 * 1.21) ** 0.5)


def test_price_levels_stop_at_gap_in_years(app):
    assert sorted(app.price_levels({2022: 10.0, 2023: 5.0, 2025: 4.0})) == [2022, 2023]


def test_price_levels_for_empty_input(app):
    assert app.price_levels({}) == {}


def test_no_flags_returns_rows_unchanged(app, mk_rows):
    rows = mk_rows(("GMKN", 2025, "revenue", 1000.0, "млн $"))
    out, notes = app.adjust_long(rows, False, False, None, MACRO)
    assert out.equals(rows) and notes == []


def test_usd_to_rub_uses_average_rate_of_the_year(app, mk_rows):
    rows = mk_rows(("GMKN", 2024, "revenue", 1000.0, "млн $"), ("GMKN", 2025, "net_income", 200.0, "млн $"))
    out, notes = app.adjust_long(rows, True, False, None, MACRO)
    by_year = out.set_index("year")
    assert by_year.loc[2024, "value"] == pytest.approx(1000 * 80 / 1000)   # млн $ × курс / 1000 = млрд ₽
    assert by_year.loc[2025, "value"] == pytest.approx(200 * 100 / 1000)
    assert (out["unit"] == "млрд ₽").all() and notes == []


def test_usd_conversion_for_other_dollar_units(app, mk_rows):
    rows = mk_rows(("X", 2025, "revenue", 2.0, "млрд $"), ("X", 2025, "ebitda", 5.0, "$"))
    out, _ = app.adjust_long(rows, True, False, None, MACRO)
    by_ind = out.set_index("indicator")
    assert by_ind.loc["revenue", "value"] == pytest.approx(200.0) and by_ind.loc["revenue", "unit"] == "млрд ₽"
    assert by_ind.loc["ebitda", "value"] == pytest.approx(500.0) and by_ind.loc["ebitda", "unit"] == "₽"


def test_rows_without_fx_rate_are_dropped_with_a_note(app, mk_rows):
    rows = mk_rows(("GMKN", 2023, "revenue", 1000.0, "млн $"), ("GMKN", 2025, "revenue", 1000.0, "млн $"))
    out, notes = app.adjust_long(rows, True, False, None, MACRO)
    assert out["year"].tolist() == [2025]
    assert any("2023" in n and "курса" in n for n in notes)


def test_real_values_are_brought_to_base_year_prices(app, mk_rows):
    rows = mk_rows(("SBER", 2024, "revenue", 100.0, "млрд ₽"), ("SBER", 2025, "revenue", 100.0, "млрд ₽"))
    out, notes = app.adjust_long(rows, False, True, 2025, MACRO)
    levels = app.price_levels(MACRO["cpi"])
    by_year = out.set_index("year")
    assert by_year.loc[2025, "value"] == pytest.approx(100.0)
    assert by_year.loc[2024, "value"] == pytest.approx(100 * levels[2025] / levels[2024])
    assert by_year.loc[2024, "value"] == pytest.approx(110.0)  # при инфляции 10% в год
    assert by_year.loc[2025, "unit"] == "млрд ₽ (цены 2025 г.)" and notes == []


def test_conversion_is_recorded_in_note(app, mk_rows):
    rows = mk_rows(("GMKN", 2024, "revenue", 1000.0, "млн $"), note="исходная заметка")
    out, _ = app.adjust_long(rows, True, True, 2025, MACRO)
    note = out["note"].iloc[0]
    assert note.startswith("исходная заметка") and "пересчет" in note and "80.00 ₽ за $" in note


def test_real_adjustment_skips_dollar_values_and_says_so(app, mk_rows):
    rows = mk_rows(("GMKN", 2025, "revenue", 1000.0, "млн $"))
    out, notes = app.adjust_long(rows, False, True, 2025, MACRO)
    assert out["unit"].iloc[0] == "млн $" and out["value"].iloc[0] == 1000.0
    assert any("GMKN" in n and "доллар" in n for n in notes)


@pytest.mark.parametrize("to_rub, real", [(True, False), (False, True), (True, True)])
def test_roe_per_share_and_share_count_are_never_changed(app, mk_rows, to_rub, real):
    rows = mk_rows(("X", 2025, "roe", 22.0, "%"), ("X", 2025, "dps", 37.6, "₽ на акцию"), ("X", 2025, "shares", 500.0, "млн шт."))
    out, notes = app.adjust_long(rows, to_rub, real, 2025, MACRO)
    assert out["value"].tolist() == [22.0, 37.6, 500.0]
    assert out["unit"].tolist() == ["%", "₽ на акцию", "млн шт."]


def test_unknown_base_year_leaves_values_and_adds_a_note(app, mk_rows):
    rows = mk_rows(("SBER", 2025, "revenue", 100.0, "млрд ₽"))
    out, notes = app.adjust_long(rows, False, True, 2030, MACRO)
    assert out["value"].iloc[0] == 100.0 and notes


def test_margins_do_not_change_after_conversion(app, ru_all, macro):
    w0, _, _ = app.make_wide(ru_all[ru_all["ticker"] == "GMKN"])
    adjusted, _ = app.adjust_long(ru_all[ru_all["ticker"] == "GMKN"], True, True, 2025, macro)
    w1, unit1, mixed1 = app.make_wide(adjusted)
    assert mixed1 is False and unit1 == "млрд ₽ (цены 2025 г.)"
    assert app.add_margins(w1)["net_margin"].tolist() == pytest.approx(app.add_margins(w0)["net_margin"].tolist())


def test_nornickel_in_rubles_matches_manual_calculation(app, ru_all, macro):
    out, _ = app.adjust_long(ru_all[(ru_all["ticker"] == "GMKN") & (ru_all["indicator"] == "revenue")], True, False, None, macro)
    assert out.set_index("year").loc[2025, "value"] == pytest.approx(13763 * macro["usd_rub"][2025] / 1000)


def test_real_growth_of_sber_is_below_nominal(app, ru_all, macro):
    sber = ru_all[(ru_all["ticker"] == "SBER") & (ru_all["indicator"] == "revenue")]
    real, _ = app.adjust_long(sber, False, True, 2025, macro)
    nominal_cagr = app.calc_cagr(sber.rename(columns={"value": "revenue"}))
    real_cagr = app.calc_cagr(real.rename(columns={"value": "revenue"}))
    assert real_cagr < nominal_cagr
    assert real_cagr == pytest.approx(18.88, abs=0.05)


def test_load_macro_reads_both_series(app, macro):
    assert set(macro) == {"usd_rub", "cpi"}
    assert all(2022 <= y <= 2025 for y in macro["usd_rub"]) and macro["usd_rub"][2022] == pytest.approx(68.5494)
    assert macro["cpi"][2025] == pytest.approx(5.59) and not any(math.isnan(v) for v in macro["cpi"].values())
