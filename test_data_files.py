"""Проверка самих файлов данных: ловит опечатки при ручном обновлении financials_ru.csv и macro_ru.csv."""
import pytest

E, W = "🔴 Ошибка", "🟡 Предупреждение"
TICKERS = ["SBER", "GAZP", "LKOH", "YDEX", "T", "GMKN"]


def test_financials_file_has_no_errors_or_warnings(app, raw_csv):
    issues, _ = app.validate_financials(raw_csv, 2026)
    bad = issues[issues["Серьезность"].isin([E, W])]
    assert bad.empty, "\n" + bad[["Серьезность", "Проверка", "Тикер", "Год", "Строка в файле", "Описание"]].to_string(index=False)


def test_every_row_of_the_file_is_read_by_the_app(raw_csv, ru_all):
    assert len(ru_all) == len(raw_csv)


def test_expected_companies_are_present(ru_all):
    assert sorted(ru_all["ticker"].unique()) == sorted(TICKERS)


@pytest.mark.parametrize("ticker", TICKERS)
def test_company_has_revenue_profit_shares_and_dividends(ru_all, ticker):
    indicators = set(ru_all[ru_all["ticker"] == ticker]["indicator"])
    assert {"revenue", "net_income", "shares", "dps"} <= indicators


@pytest.mark.parametrize("ticker", TICKERS)
def test_company_money_units_are_consistent(app, ru_all, ticker):
    _, unit, mixed = app.make_wide(ru_all[ru_all["ticker"] == ticker])
    assert mixed is False and unit in ("млрд ₽", "млн $")


@pytest.mark.parametrize("ticker", ["SBER", "T"])
def test_banks_have_roe_and_no_ebitda(ru_all, ticker):
    indicators = set(ru_all[ru_all["ticker"] == ticker]["indicator"])
    assert "roe" in indicators and "ebitda" not in indicators


def test_all_years_are_in_the_reporting_range(ru_all):
    assert ru_all["year"].between(2022, 2025).all()


def test_every_estimate_is_explained(ru_all):
    estimates = ru_all[ru_all["is_estimate"]]
    assert not estimates.empty and (estimates["note"].str.strip() != "").all()


def test_every_row_has_a_source_and_a_definition(ru_all):
    assert (ru_all["source"].str.strip() != "").all() and (ru_all["metric"].str.strip() != "").all()


def test_values_are_plausible(ru_all):
    revenue = ru_all[ru_all["indicator"] == "revenue"]
    assert (revenue["value"] > 0).all()
    assert ru_all[ru_all["indicator"] == "roe"]["value"].between(-100, 100).all()
    assert (ru_all[ru_all["indicator"] == "shares"]["value"] > 0).all() and (ru_all[ru_all["indicator"] == "dps"]["value"] >= 0).all()


def test_no_duplicate_rows(ru_all):
    assert not ru_all.duplicated(["ticker", "year", "indicator"]).any()


def test_macro_file_covers_all_data_years(macro, ru_all):
    years = set(ru_all["year"])
    assert years <= set(macro["usd_rub"]) and years <= set(macro["cpi"])


def test_macro_values_are_plausible(macro):
    assert all(40 < v < 150 for v in macro["usd_rub"].values())
    assert all(0 < v < 30 for v in macro["cpi"].values())


def test_inflation_series_has_no_gaps(app, macro):
    assert sorted(app.price_levels(macro["cpi"])) == sorted(macro["cpi"])


def test_loader_reports_a_broken_file(app, fake_st, tmp_path, monkeypatch):
    broken = tmp_path / "financials_ru.csv"
    broken.write_text("ticker,year\nSBER,2025\n", encoding="utf-8")
    monkeypatch.setattr(app, "CSV_PATH", broken)
    before = len(fake_st.messages)
    assert app.load_ru_all().empty
    assert any(name == "error" for name, _ in fake_st.messages[before:])


def test_loader_handles_a_missing_file(app, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CSV_PATH", tmp_path / "nothing.csv")
    assert app.load_ru_all().empty and app.load_revenue_csv().empty
