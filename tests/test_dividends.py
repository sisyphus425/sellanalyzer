"""Прогноз дивидендов: прибыль × payout ÷ число акций, фактический payout, политика из файла."""
import pandas as pd
import pytest


def scen(*rows):
    return pd.DataFrame(rows, columns=["Год", "Низкий", "База", "Высокий"])


def test_dividend_forecast_formula(app):
    f = app.dividend_forecast(scen((2026, 1500, 2000, 2500)), 22586.95, 50, price=300, last_dps=37.64)
    r = f.iloc[0]
    assert r["Дивиденд на акцию (база)"] == pytest.approx(2000 * 0.5 * 1000 / 22586.95)
    assert r["Дивиденд на акцию (низкий)"] == pytest.approx(1500 * 0.5 * 1000 / 22586.95)
    assert r["Дивиденд на акцию (высокий)"] == pytest.approx(2500 * 0.5 * 1000 / 22586.95)
    assert r["Дивидендная доходность (база), %"] == pytest.approx(44.27 / 300 * 100, abs=0.05)
    assert r["Рост к последнему факту, %"] == pytest.approx(17.62, abs=0.05)
    assert r["Дивиденды всего (база)"] == pytest.approx(1000.0)


def test_loss_means_no_dividend(app):
    f = app.dividend_forecast(scen((2026, -50, 100, 200)), 1000, 50)
    assert f["Дивиденд на акцию (низкий)"].iloc[0] == 0
    assert f["Дивиденд на акцию (база)"].iloc[0] == pytest.approx(50.0)


def test_zero_base_has_no_growth(app):
    f = app.dividend_forecast(scen((2026, -5, -1, 0)), 1000, 50, last_dps=10)
    assert pd.isna(f["Рост к последнему факту, %"].iloc[0])


def test_no_shares_gives_empty_table(app):
    assert app.dividend_forecast(scen((2026, 1, 2, 3)), None, 50).empty
    assert app.dividend_forecast(scen((2026, 1, 2, 3)), 0, 50).empty


def test_no_price_gives_no_yield(app):
    f = app.dividend_forecast(scen((2026, 1, 2, 3)), 1000, 50)
    assert pd.isna(f["Дивидендная доходность (база), %"].iloc[0])


def test_payout_scales_linearly(app):
    a = app.dividend_forecast(scen((2026, 1, 2, 3)), 1000, 25)["Дивиденд на акцию (база)"].iloc[0]
    b = app.dividend_forecast(scen((2026, 1, 2, 3)), 1000, 50)["Дивиденд на акцию (база)"].iloc[0]
    assert b == pytest.approx(2 * a)


def test_historical_payout(app):
    h = app.historical_payouts({2024: 1700, 2025: 1800}, {2024: 34.84, 2025: 37.64}, 22586.95)
    assert h["Год"].tolist() == [2024, 2025]
    assert h["Фактический payout, %"].iloc[1] == pytest.approx(37.64 * 22586.95 / 1000 / 1800 * 100)


def test_historical_payout_none_for_loss_or_missing_shares(app):
    h = app.historical_payouts({2024: -5, 2025: 10}, {2024: 1, 2025: 1}, 1000)
    assert pd.isna(h["Фактический payout, %"].iloc[0])
    h2 = app.historical_payouts({2025: 10}, {2025: 1}, None)
    assert pd.isna(h2["Фактический payout, %"].iloc[0])


def test_historical_payout_uses_only_common_years(app):
    h = app.historical_payouts({2023: 10, 2024: 10}, {2024: 1, 2025: 1}, 1000)
    assert h["Год"].tolist() == [2024]


def test_average_payout_takes_last_known_years(app):
    h = pd.DataFrame({"Год": [2021, 2022, 2023, 2024, 2025], "Фактический payout, %": [10, 20, 30, None, 60]})
    # последние три известных: 2022, 2023, 2025
    assert app.average_payout(h, 3) == pytest.approx((20 + 30 + 60) / 3)


def test_average_payout_none_without_data(app):
    assert app.average_payout(pd.DataFrame({"Год": [2025], "Фактический payout, %": [None]})) is None


def test_policy_gap_warns_when_actual_is_far_below_policy(app):
    h = pd.DataFrame({"Год": [2024, 2025], "Фактический payout, %": [20.0, 22.0]})
    avg, pol, warn = app.policy_gap(h, 50)
    assert avg == pytest.approx(21.0) and pol == 50 and warn
    assert app.policy_gap(h, 30)[2] is None      # 21 >= 15: в норме


def test_policy_gap_boundary(app):
    h = pd.DataFrame({"Год": [2025], "Фактический payout, %": [25.0]})
    assert app.policy_gap(h, 50)[2] is None      # ровно половина политики не предупреждает


def test_policy_gap_without_policy(app):
    h = pd.DataFrame({"Год": [2025], "Фактический payout, %": [25.0]})
    assert app.policy_gap(h, None) == (25.0, None, None)


def test_policy_from_rows_reads_latest_with_source(app, mk_rows):
    rows = mk_rows(("SBER", 2024, "payout_target", 40.0, "%"), ("SBER", 2025, "payout_target", 50.0, "%"))
    p = app.policy_from_rows(rows)
    assert p["pct"] == 50.0 and p["year"] == 2025


def test_policy_from_rows_none_when_absent(app, ru_all):
    assert app.policy_from_rows(ru_all[ru_all["ticker"] == "YDEX"]) is None
    assert app.policy_from_rows(pd.DataFrame()) is None


def test_payout_choices(app):
    assert app.payout_choices(50, 21) == {"По политике компании": 50.0, "Средний фактический": 21.0}
    assert app.payout_choices(None, 21) == {"Средний фактический": 21.0}
    assert app.payout_choices(None, None) == {}


def test_policy_rows_in_data_file_have_source_and_sane_value(ru_all):
    pol = ru_all[ru_all["indicator"] == "payout_target"]
    assert {"SBER", "GAZP", "T"} <= set(pol["ticker"])
    assert pol["value"].between(0, 100).all()
    assert (pol["source"].fillna("").str.len() > 0).all()
    assert (pol["unit"] == "%").all()


def _val(app, mk_raw, value):
    issues, _ = app.validate_financials(mk_raw(["SBER", 2025, "payout_target", value, "%", "m", "false", "s", ""]), 2026)
    return issues[issues["Проверка"] == "Целевой payout вне диапазона"]


@pytest.mark.parametrize("value", ["500", "-1", "300.5"])
def test_validation_warns_on_insane_payout_target(app, mk_raw, value):
    assert len(_val(app, mk_raw, value)) == 1


@pytest.mark.parametrize("value", ["0", "50", "300"])
def test_validation_silent_on_sane_payout_target(app, mk_raw, value):
    assert _val(app, mk_raw, value).empty
