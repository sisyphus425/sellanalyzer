"""Сравнение компаний, разбор тикеров и данные yfinance."""
import pandas as pd
import pytest


@pytest.mark.parametrize("text, expected", [("aapl, msft;  nvda  AAPL", ["AAPL", "MSFT", "NVDA"]), ("", []), (" , ;", []),
                                            ("brk-b", ["BRK-B"])])
def test_parse_tickers(app, text, expected):
    assert app.parse_tickers(text) == expected


def test_collect_revenue_for_russian_companies(app):
    problems = []
    rev = app.collect_revenue(["SBER", "GAZP", "ZZZZ"], [], app.load_revenue_csv(), problems)
    assert set(rev["label"]) == {"SBER", "GAZP"} and len(problems) == 1 and "ZZZZ" in problems[0]
    assert {"label", "year", "revenue", "unit", "is_estimate", "growth"} <= set(rev.columns)
    sber = rev[rev["label"] == "SBER"].set_index("year")
    assert sber.loc[2023, "growth"] == pytest.approx(66.33, abs=0.01)


def test_index_revenue_with_common_base_year(app):
    problems = []
    rev = app.collect_revenue(["SBER", "GAZP"], [], app.load_revenue_csv(), problems)
    idx = app.index_revenue(rev, 2022)
    assert idx.groupby("label")["index"].first().tolist() == [100.0, 100.0]
    sber = idx[idx["label"] == "SBER"].set_index("year")
    assert sber.loc[2025, "index"] == pytest.approx(4089.0 / 1902.9 * 100)


def test_index_revenue_skips_companies_without_the_base_year(app):
    rev = app.collect_revenue(["SBER", "YDEX"], [], app.load_revenue_csv(), [])  # у Яндекса нет 2022 года
    assert set(app.index_revenue(rev, 2022)["label"]) == {"SBER"}
    own_start = app.index_revenue(rev, None)  # без базового года каждая компания стартует со своего первого года
    assert set(own_start["label"]) == {"SBER", "YDEX"} and own_start.groupby("label")["index"].first().tolist() == [100.0, 100.0]


def test_summary_table_and_numeric_summary_agree(app):
    rev = app.collect_revenue(["SBER", "YDEX"], [], app.load_revenue_csv(), [])
    common = sorted(set.intersection(*[set(g["year"]) for _, g in rev.groupby("label")]))
    assert common == [2023, 2024, 2025]
    numeric = app.summary_numeric(rev, common).set_index("Компания")
    text = app.summary_table(rev, common)
    assert numeric.loc["SBER", "CAGR весь ряд, %"] == pytest.approx(29.04, abs=0.01)
    assert numeric.loc["SBER", "CAGR общий период, %"] == pytest.approx(((4089.0 / 3165.0) ** 0.5 - 1) * 100, abs=0.01)
    assert "29.0%" in text.loc["SBER", "CAGR (весь ряд)"] and text.loc["SBER", "Рост к пред. году"].startswith("🟢")


def test_summary_without_common_period(app):
    rev = app.collect_revenue(["SBER", "LKOH"], [], app.load_revenue_csv(), [])
    common = sorted(set.intersection(*[set(g["year"]) for _, g in rev.groupby("label")]))
    numeric = app.summary_numeric(rev, [2025] if common else common)
    assert numeric["Общий период"].eq("").all()


def test_collect_profit_includes_margins_and_roe(app, ru_all):
    prof = app.collect_profit(["SBER", "GMKN"], [], ru_all)
    assert set(prof["label"]) == {"SBER", "GMKN"}
    sber = prof[prof["label"] == "SBER"].set_index("year")
    assert sber.loc[2025, "roe"] == 22.7 and sber.loc[2024, "net_margin"] == pytest.approx(1580.3 / 3506.1 * 100)
    assert prof[prof["label"] == "GMKN"]["roe"].isna().all()


def test_collect_profit_for_unknown_ticker_is_empty(app, ru_all):
    assert app.collect_profit(["ZZZZ"], [], ru_all).empty


def fin_frame():
    cols = pd.to_datetime(["2025-09-30", "2024-09-30", "2023-09-30"])
    return pd.DataFrame({cols[0]: [416.0, 112.0, 144.0], cols[1]: [391.0, 93.7, 134.0], cols[2]: [383.0, 97.0, float("nan")]},
                        index=["Total Revenue", "Net Income", "EBITDA"])


def test_us_wide_from_yfinance_style_table(app):
    w = app.us_wide(fin_frame())
    assert w["year"].tolist() == [2023, 2024, 2025]
    assert w["revenue"].tolist() == [383.0, 391.0, 416.0] and w["net_income"].tolist() == [97.0, 93.7, 112.0]
    assert w["ebitda"].isna().tolist() == [True, False, False]
    assert w["roe"].isna().all() and not w["has_estimate"].any()


def test_us_wide_without_some_rows_and_on_unrelated_table(app):
    fin = fin_frame().drop(index="EBITDA")
    assert app.us_wide(fin)["ebitda"].isna().all()
    assert app.us_wide(pd.DataFrame({pd.Timestamp("2025-09-30"): [1.0]}, index=["Something Else"])).empty
