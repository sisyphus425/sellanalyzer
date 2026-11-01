"""Сценарный калькулятор «что если»: расчет по допущениям, готовые наборы, чувствительность."""
import math

import numpy as np
import pandas as pd
import pytest

NAN = float("nan")


def wide(shares=10000.0, dps=5.0):
    """Годы 2024–2025: выручка 900 → 1000 млрд ₽, прибыль 80 → 100, 10 млрд акций, дивиденд за 2025 год 5 ₽."""
    return pd.DataFrame({"year": [2024, 2025], "revenue": [900.0, 1000.0], "net_income": [80.0, 100.0],
                         "shares": [NAN, shares], "dps": [NAN, dps]})


@pytest.fixture
def base(app):
    b, notes = app.build_base(wide(), "млрд ₽", price=100.0)
    assert notes == []
    return b


def params(growth=10.0, margin=12.0, shares=0.0, payout=50.0, n=1):
    return [{"growth": growth, "margin": margin, "shares_change": shares, "payout": payout}] * n


# ---------- исходные значения ----------

def test_build_base_takes_the_latest_year_with_revenue_and_profit(base):
    assert base["year"] == 2025 and base["revenue"] == 1000.0 and base["net_income"] == 100.0 and base["margin"] == pytest.approx(10.0)
    assert base["shares_mn"] == 10000.0 and base["eps"] == pytest.approx(10.0) and base["pe"] == pytest.approx(10.0)
    assert base["payout"] == pytest.approx(50.0)   # 5 ₽ × 10 млрд акций = 50 млрд ₽ из 100 млрд ₽
    assert base["last_growth"] == pytest.approx(100 / 900 * 100)


def test_build_base_converts_units_to_billions(app):
    w = pd.DataFrame({"year": [2025], "revenue": [13763.0], "net_income": [2470.0]})
    b, _ = app.build_base(w, "млн $", price=140.0, currency="$", shares_mn=15286.34)
    assert b["revenue"] == pytest.approx(13.763) and b["net_income"] == pytest.approx(2.47) and b["currency"] == "$"
    w = pd.DataFrame({"year": [2025], "revenue": [416e9], "net_income": [112e9]})
    b, _ = app.build_base(w, "$", shares_mn=14800.0, currency="USD")
    assert b["revenue"] == pytest.approx(416.0) and b["eps"] == pytest.approx(112 * 1000 / 14800)


def test_build_base_explicit_shares_and_dps_override_the_table(app):
    b, _ = app.build_base(wide(shares=NAN, dps=NAN), "млрд ₽", price=100.0, shares_mn=5000.0, dps=4.0)
    assert b["shares_mn"] == 5000.0 and b["eps"] == pytest.approx(20.0) and b["payout"] == pytest.approx(4.0 * 5000 / 1000 / 100 * 100)


def test_missing_shares_price_and_unsupported_input(app):
    b, notes = app.build_base(wide(shares=NAN), "млрд ₽", price=100.0)
    assert b["shares_mn"] is None and b["eps"] is None and b["pe"] is None and any("Число акций" in n for n in notes)
    b, notes = app.build_base(wide(), "млрд ₽", price=None)
    assert b["price"] is None and b["pe"] is None and b["eps"] == pytest.approx(10.0) and any("Цена" in n for n in notes)
    assert app.build_base(wide(), "тыс. евро")[0] is None
    assert app.build_base(pd.DataFrame({"year": [2025], "revenue": [100.0], "net_income": [NAN]}), "млрд ₽")[0] is None


def test_payout_is_unknown_when_dividend_belongs_to_another_year(app):
    w = wide()
    w.loc[w["year"] == 2025, "dps"] = NAN
    w.loc[w["year"] == 2024, "dps"] = 4.0
    b, _ = app.build_base(w, "млрд ₽", price=100.0)
    assert b["dps"] is None and b["payout"] is None


def test_dividends_by_year_are_used_for_the_base_year(app):
    w = pd.DataFrame({"year": [2024, 2025], "revenue": [900.0, 1000.0], "net_income": [80.0, 100.0]})   # как у зарубежных компаний: без колонок shares и dps
    b, _ = app.build_base(w, "млрд $", price=100.0, shares_mn=10000.0, dps_by_year={2024: 3.0, 2025: 4.0})
    assert b["dps"] == 4.0 and b["payout"] == pytest.approx(4.0 * 10000 / 1000 / 100 * 100)
    b, _ = app.build_base(w, "млрд $", price=100.0, shares_mn=10000.0, dps_by_year={2023: 3.0})
    assert b["dps"] is None and b["payout"] is None


def test_with_price_recomputes_pe_without_touching_the_original(app, base):
    other = app.with_price(base, 200.0)
    assert other["price"] == 200.0 and other["pe"] == pytest.approx(20.0) and base["price"] == 100.0 and base["pe"] == pytest.approx(10.0)
    none = app.with_price(base, 0.0)
    assert none["price"] is None and none["pe"] is None
    y1 = app.scenario(app.with_price(base, 200.0), params(growth=10.0, margin=12.0)).iloc[1]
    assert y1["P/E"] == pytest.approx(200 / 13.2) and y1["Цена при неизменном P/E"] == pytest.approx(20.0 * 13.2)


# ---------- расчет сценария ----------

def test_one_year_scenario_arithmetic(app, base):
    df = app.scenario(base, params(growth=10.0, margin=12.0, payout=50.0))
    fact, y1 = df.iloc[0], df.iloc[1]
    assert fact["Год"] == 2025 and y1["Год"] == 2026
    assert y1["Выручка"] == pytest.approx(1100.0) and y1["Чистая прибыль"] == pytest.approx(132.0)
    assert y1["EPS"] == pytest.approx(13.2) and y1["Дивиденды всего"] == pytest.approx(66.0) and y1["Дивиденд на акцию"] == pytest.approx(6.6)
    assert y1["P/E"] == pytest.approx(100 / 13.2) and y1["Дивидендная доходность, %"] == pytest.approx(6.6)
    assert y1["Рост прибыли, %"] == pytest.approx(32.0) and y1["Рост EPS, %"] == pytest.approx(32.0)
    assert y1["Цена при неизменном P/E"] == pytest.approx(132.0)   # P/E 10 × EPS 13,2


def test_the_fact_row_repeats_base_values(app, base):
    fact = app.scenario(base, params()).iloc[0]
    assert fact["Выручка"] == 1000.0 and fact["EPS"] == pytest.approx(10.0) and fact["P/E"] == pytest.approx(10.0)
    assert fact["Дивидендная доходность, %"] == pytest.approx(5.0) and fact["Цена при неизменном P/E"] == 100.0


def test_years_compound_and_share_changes_dilute_eps(app, base):
    df = app.scenario(base, params(growth=10.0, margin=12.0, shares=2.0, n=2))
    assert df["Выручка"].tolist()[1:] == pytest.approx([1100.0, 1210.0])
    assert df["Число акций, млн"].tolist()[1:] == pytest.approx([10200.0, 10404.0])
    assert df["EPS"].iloc[2] == pytest.approx(1210 * 0.12 * 1000 / 10404)
    assert df["Рост выручки, %"].tolist()[1:] == [10.0, 10.0]


def test_price_at_constant_pe_follows_eps(app, base):
    df = app.scenario(base, params(growth=20.0, margin=11.0, n=3))
    assert (df["Цена при неизменном P/E"] / df["EPS"]).iloc[1:].tolist() == pytest.approx([10.0, 10.0, 10.0])
    assert df["Цена при неизменном P/E"].iloc[3] == pytest.approx(100.0 * df["EPS"].iloc[3] / df["EPS"].iloc[0])


def test_loss_scenario_has_no_pe_no_dividends_and_no_growth(app, base):
    y1 = app.scenario(base, params(growth=0.0, margin=-5.0, payout=50.0)).iloc[1]
    assert y1["Чистая прибыль"] == pytest.approx(-50.0) and y1["EPS"] == pytest.approx(-5.0)
    assert y1["P/E"] is None or math.isnan(y1["P/E"])
    assert y1["Цена при неизменном P/E"] is None or math.isnan(y1["Цена при неизменном P/E"])
    assert y1["Дивиденды всего"] == 0.0 and y1["Дивиденд на акцию"] == 0.0
    assert y1["Рост прибыли, %"] is None or math.isnan(y1["Рост прибыли, %"])


def test_scenario_without_shares_or_price_does_not_break(app):
    b, _ = app.build_base(wide(shares=NAN), "млрд ₽", price=100.0)
    y1 = app.scenario(b, params()).iloc[1]
    assert y1["Чистая прибыль"] == pytest.approx(132.0) and y1["Дивиденды всего"] == pytest.approx(66.0)
    assert all(pd.isna(y1[c]) for c in ("EPS", "Дивиденд на акцию", "P/E", "Цена при неизменном P/E"))
    b, _ = app.build_base(wide(), "млрд ₽", price=None)
    y1 = app.scenario(b, params()).iloc[1]
    assert y1["EPS"] == pytest.approx(13.2) and all(pd.isna(y1[c]) for c in ("P/E", "Дивидендная доходность, %", "Цена при неизменном P/E"))


def test_scenario_needs_at_least_one_year(app, base):
    with pytest.raises(ValueError):
        app.scenario(base, [])


def test_columns_are_stable(app, base):
    assert list(app.scenario(base, params()).columns) == app.SCENARIO_COLS


# ---------- готовые наборы допущений ----------

def test_presets_have_the_right_shape_and_values(app, base):
    for preset in app.PRESETS:
        p = app.make_params(base, 3, preset)
        assert len(p) == 3 and all(set(x) == {"growth", "margin", "shares_change", "payout"} for x in p), preset
    last_g = base["last_growth"]
    assert [x["growth"] for x in app.make_params(base, 2, "Рост как в прошлом году")] == [pytest.approx(last_g)] * 2
    assert [x["growth"] for x in app.make_params(base, 2, "Стагнация")] == [0.0, 0.0]
    assert app.make_params(base, 1, "Стагнация")[0]["margin"] == pytest.approx(base["margin"])
    better = app.make_params(base, 1, "Улучшение маржи (+2 п.п.)", growths=[7.0])[0]
    assert better["margin"] == pytest.approx(base["margin"] + 2) and better["growth"] == 7.0
    worse = app.make_params(base, 1, "Ухудшение (рост −5 п.п., маржа −2 п.п.)", growths=[7.0])[0]
    assert worse["growth"] == pytest.approx(2.0) and worse["margin"] == pytest.approx(base["margin"] - 2)
    assert app.make_params(base, 1)[0]["payout"] == pytest.approx(50.0) and app.make_params(base, 1)[0]["shares_change"] == 0.0


def test_forecast_growths_are_used_and_extended(app, base):
    p = app.make_params(base, 3, "Как в базовом прогнозе", growths=[8.0, 6.0])
    assert [x["growth"] for x in p] == [8.0, 6.0, 6.0]


def test_default_growths_follow_the_base_forecast(app, ru_all):
    rows = ru_all[(ru_all["ticker"] == "SBER") & (ru_all["indicator"] == "revenue")].sort_values("year")
    g = app.default_growths(rows["year"], rows["value"], 3)
    assert len(g) == 3 and g[2] == g[1]   # дальше второго года прогноз не строится: последний темп повторяется
    res = app.forecast_series(rows["year"], rows["value"], horizon=2, nonnegative=True)
    assert g[0] == pytest.approx((res["scenarios"]["База"].iloc[0] / 4089.0 - 1) * 100)
    lk = ru_all[(ru_all["ticker"] == "LKOH") & (ru_all["indicator"] == "revenue")]
    assert app.default_growths(lk["year"], lk["value"], 2) is None


def test_growth_source_reliability_matches_the_forecast(app, ru_all):
    sber = ru_all[(ru_all["ticker"] == "SBER") & (ru_all["indicator"] == "revenue")].sort_values("year")
    assert app.growth_source_reliability(sber["year"], sber["value"]) == "низкая"
    lk = ru_all[(ru_all["ticker"] == "LKOH") & (ru_all["indicator"] == "revenue")]
    assert app.growth_source_reliability(lk["year"], lk["value"]) == "недостаточно данных"


def test_assumptions_are_clamped_to_the_allowed_limits(app, base):
    p = app.clamp_params([{"growth": 900.0, "margin": -250.0, "shares_change": 5.0, "payout": 484.0}])[0]
    assert p == {"growth": 500.0, "margin": -100.0, "shares_change": 5.0, "payout": 300.0}
    huge = {**base, "payout": 483.6, "last_growth": 900.0}
    for preset in app.PRESETS:
        for x in app.make_params(huge, 3, preset):
            assert all(app.LIMITS[k][0] <= v <= app.LIMITS[k][1] for k, v in x.items()), (preset, x)


def test_every_real_company_gets_assumptions_inside_the_limits(app, ru_all, macro):
    for ticker in ("SBER", "GAZP", "LKOH", "YDEX", "T", "GMKN"):
        b, _ = real_base(app, ru_all, macro, ticker, 100.0)
        for preset in app.PRESETS:
            for x in app.make_params(b, 3, preset):
                assert all(app.LIMITS[k][0] <= v <= app.LIMITS[k][1] for k, v in x.items()), (ticker, preset, x)


# ---------- чувствительность ----------

def test_sensitivity_grid_center_matches_the_scenario(app, base):
    p = params(growth=10.0, margin=12.0, n=2)
    final = app.scenario(base, p).iloc[-1]
    for metric, col in (("P/E", "P/E"), ("EPS", "EPS"), ("Чистая прибыль", "Чистая прибыль"), ("Цена при неизменном P/E", "Цена при неизменном P/E")):
        grid = app.sensitivity(base, p, metric)
        assert grid.shape == (5, 5) and grid.iloc[2, 2] == pytest.approx(final[col]), metric


def test_sensitivity_moves_in_the_expected_direction(app, base):
    p = params(growth=10.0, margin=12.0, n=2)
    eps, pe = app.sensitivity(base, p, "EPS"), app.sensitivity(base, p, "P/E")
    assert (eps.diff(axis=0).iloc[1:] > 0).all().all() and (eps.diff(axis=1).iloc[:, 1:] > 0).all().all()   # выше маржа и рост: выше EPS
    assert (pe.diff(axis=0).iloc[1:] < 0).all().all() and (pe.diff(axis=1).iloc[:, 1:] < 0).all().all()     # и ниже P/E


def test_sensitivity_cells_apply_the_right_shift_to_growth_and_to_margin(app, base):
    """Столбцы меняют рост выручки, строки меняют маржу; эффект у них разный, поэтому перепутать оси нельзя."""
    p = params(growth=10.0, margin=12.0, n=2)
    grid = app.sensitivity(base, p, "EPS")
    for row, col, dm, dg in (("Маржа +1.5 п.п.", "Рост −5 п.п.", 1.5, -5.0), ("Маржа −3 п.п.", "Рост +10 п.п.", -3.0, 10.0)):
        shifted = [{**x, "growth": x["growth"] + dg, "margin": x["margin"] + dm} for x in p]
        assert grid.loc[row, col] == pytest.approx(app.scenario(base, shifted)["EPS"].iloc[-1])
    assert grid.loc["Маржа +1.5 п.п.", "Рост −5 п.п."] != pytest.approx(grid.loc["Маржа −3 п.п.", "Рост +10 п.п."])


def test_sensitivity_labels(app, base):
    grid = app.sensitivity(base, params(), "EPS")
    assert grid.columns.tolist() == ["Рост −10 п.п.", "Рост −5 п.п.", "Рост 0 п.п.", "Рост +5 п.п.", "Рост +10 п.п."]
    assert grid.index.tolist() == ["Маржа −3 п.п.", "Маржа −1.5 п.п.", "Маржа 0 п.п.", "Маржа +1.5 п.п.", "Маржа +3 п.п."]


# ---------- реальные данные ----------

def real_base(app, ru_all, macro, ticker, price):
    from salesanalyzer.config import UNIT_TO_BN  # noqa: F401  (проверяем, что справочник доступен)
    rows, _ = app.adjust_long(ru_all[ru_all["ticker"] == ticker], True, False, None, macro)
    w, unit, _ = app.make_wide(rows)
    return app.build_base(w, unit, price=price, currency="₽")


def test_real_data_sber_base(app, ru_all, macro):
    b, notes = real_base(app, ru_all, macro, "SBER", 300.0)
    assert notes == [] and b["year"] == 2025 and b["revenue"] == pytest.approx(4089.0) and b["net_income"] == pytest.approx(1706.0)
    assert b["eps"] == pytest.approx(1706 * 1000 / 22586.95) and b["pe"] == pytest.approx(300 / b["eps"])
    assert b["payout"] == pytest.approx(37.64 * 22586.95 / 1000 / 1706 * 100)    # около 49,8%
    assert b["last_growth"] == pytest.approx((4089.0 / 3506.1 - 1) * 100)


def test_real_data_nornickel_is_converted_to_rubles(app, ru_all, macro):
    b, _ = real_base(app, ru_all, macro, "GMKN", 140.0)
    rate = macro["usd_rub"][2025]
    assert b["revenue"] == pytest.approx(13763 * rate / 1000) and b["net_income"] == pytest.approx(2470 * rate / 1000)
    assert b["payout"] == 0.0 and b["shares_mn"] == pytest.approx(15286.34)


def test_real_data_every_company_gets_a_working_scenario(app, ru_all, macro):
    for ticker, price in (("SBER", 300.0), ("GAZP", 130.0), ("LKOH", 6000.0), ("YDEX", 4000.0), ("T", 300.0), ("GMKN", 140.0)):
        b, _ = real_base(app, ru_all, macro, ticker, price)
        assert b is not None, ticker
        df = app.scenario(b, app.make_params(b, 3, "Стагнация"))
        assert len(df) == 4 and np.isfinite(df["Чистая прибыль"].to_numpy()).all(), ticker
