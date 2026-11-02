"""Мультипликаторы (P/S, P/E), дивиденды и payout."""
import pandas as pd
import pytest

DATE = pd.Timestamp("2026-10-16")


def rub(app, ticker, price, ru_all, macro):
    return app.market_stats_ru(ru_all[ru_all["ticker"] == ticker], macro, price, DATE)


def test_sber_dividends_match_company_figures(app, ru_all, macro):
    """Сбербанк за 2024 год: 786,9 млрд ₽ дивидендов, 49,8% чистой прибыли по МСФО (данные компании)."""
    _, payout, notes = rub(app, "SBER", 300.0, ru_all, macro)
    row = payout[payout["Год"] == 2024].iloc[0]
    assert row["Дивиденды всего, млрд ≈"] == pytest.approx(786.9, abs=0.1)
    assert row["Payout, %"] == pytest.approx(49.8, abs=0.1) and notes == []


def test_sber_multiples(app, ru_all, macro):
    stats, _, _ = rub(app, "SBER", 300.0, ru_all, macro)
    mcap = 300.0 * 22586.95 / 1000
    assert stats["mcap"] == pytest.approx(mcap)
    assert stats["ps"] == pytest.approx(mcap / 4089.0) and stats["pe"] == pytest.approx(mcap / 1706.0)
    assert stats["div_yield"] == pytest.approx(37.64 / 300.0 * 100)
    assert stats["dps_last"] == (2025, 37.64) and stats["y_rev"] == 2025 and stats["y_ni"] == 2025


def test_nornickel_dollar_figures_are_converted_to_rubles(app, ru_all, macro):
    stats, payout, _ = rub(app, "GMKN", 140.0, ru_all, macro)
    mcap = 140.0 * 15286.34 / 1000
    assert stats["ps"] == pytest.approx(mcap / (13763 * macro["usd_rub"][2025] / 1000))
    assert stats["pe"] == pytest.approx(mcap / (2470 * macro["usd_rub"][2025] / 1000))
    assert payout.set_index("Год").loc[2023, "Дивиденд на акцию"] == pytest.approx(9.1533)


def test_zero_dividends_give_zero_yield_and_zero_payout(app, ru_all, macro):
    stats, payout, _ = rub(app, "GAZP", 130.0, ru_all, macro)
    assert stats["div_yield"] == 0.0 and stats["dps_last"] == (2025, 0.0)
    assert payout.set_index("Год").loc[2025, "Payout, %"] == 0.0


def test_payout_is_empty_for_loss_years(app, ru_all, macro):
    """У Газпрома в 2023 году убыток: payout не определен."""
    _, payout, _ = rub(app, "GAZP", 130.0, ru_all, macro)
    assert pd.isna(payout.set_index("Год").loc[2023, "Payout, %"])


def test_estimated_dividends_are_flagged(app, ru_all, macro):
    stats, payout, _ = rub(app, "YDEX", 4000.0, ru_all, macro)
    assert stats["dps_est"] is True and payout["Оценка"].tolist() == ["да", "да"]


def test_missing_shares_means_no_market_cap_but_no_crash(app, ru_all, macro):
    rows = ru_all[(ru_all["ticker"] == "SBER") & (ru_all["indicator"] != "shares")]
    stats, _, notes = app.market_stats_ru(rows, macro, 300.0, DATE)
    assert stats["mcap"] is None and stats["ps"] is None and stats["pe"] is None
    assert any("число акций" in n for n in notes)


def test_missing_price_means_no_multiples(app, ru_all, macro):
    stats, _, notes = app.market_stats_ru(ru_all[ru_all["ticker"] == "SBER"], macro, None, None)
    assert stats["mcap"] is None and stats["div_yield"] is None and any("цены" in n for n in notes)


def test_negative_earnings_give_no_pe(app, mk_rows, macro):
    rows = mk_rows(("X", 2025, "revenue", 100.0, "млрд ₽"), ("X", 2025, "net_income", -5.0, "млрд ₽"),
                   ("X", 2025, "shares", 1000.0, "млн шт."))
    stats, _, _ = app.market_stats_ru(rows, macro, 10.0, DATE)
    assert stats["pe"] is None and stats["ps"] == pytest.approx(10.0 * 1000 / 1000 / 100)


def test_unknown_units_disable_multiples(app, mk_rows, macro):
    rows = mk_rows(("X", 2025, "revenue", 100.0, "тыс. евро"), ("X", 2025, "shares", 1000.0, "млн шт."))
    stats, _, notes = app.market_stats_ru(rows, macro, 10.0, DATE)
    assert stats["ps"] is None and any("не поддерживаются" in n for n in notes)


def test_empty_input_returns_no_stats(app, macro):
    stats, payout, notes = app.market_stats_ru(pd.DataFrame(columns=["ticker"]), macro, 10.0, DATE)
    assert stats is None and payout.empty and notes


def us_data(**overrides):
    base = {
        "wide": pd.DataFrame({"year": [2024, 2025], "revenue": [391e9, 416e9], "net_income": [93.7e9, 112e9]}),
        "currency": "USD", "fin_currency": "USD", "market_cap": 3.8e12, "shares": 14.8e9,
        "dividends": pd.DataFrame({"year": [2024, 2025], "dps": [0.98, 1.02]}), "div_ttm": 1.01,
    }
    base.update(overrides)
    return base


def test_us_multiples_and_dividends(app):
    stats, payout, notes = app.market_stats_us(us_data(), 255.0, DATE)
    assert stats["mcap"] == pytest.approx(3800.0)
    assert stats["ps"] == pytest.approx(3.8e12 / 416e9) and stats["pe"] == pytest.approx(3.8e12 / 112e9)
    assert stats["div_yield"] == pytest.approx(1.01 / 255.0 * 100) and notes == []
    assert payout.set_index("Год").loc[2025, "Payout, %"] == pytest.approx(1.02 * 14.8e9 / 112e9 * 100, abs=0.1)


def test_us_multiples_are_skipped_when_currencies_differ(app):
    stats, _, notes = app.market_stats_us(us_data(fin_currency="CNY"), 255.0, DATE)
    assert stats["ps"] is None and stats["pe"] is None and any("CNY" in n for n in notes)


def test_us_company_without_dividends(app):
    stats, payout, _ = app.market_stats_us(us_data(dividends=pd.DataFrame(), div_ttm=None), 255.0, DATE)
    assert payout.empty and stats["dps_last"] is None and stats["div_yield"] is None


def test_summary_row_and_export_tables(app, ru_all, macro):
    stats, payout, _ = rub(app, "SBER", 300.0, ru_all, macro)
    row = app.market_summary_row("SBER", stats, payout)
    assert row["Компания"] == "SBER" and row["P/S"] == round(stats["ps"], 2) and row["Payout, %"] == 49.8
    tables = app.market_tables(stats, payout)
    assert set(tables) == {"Мультипликаторы", "Дивиденды"} and tables["Дивиденды"].shape[0] == 4
