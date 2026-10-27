"""Выгрузка в CSV и Excel."""
import io

import pandas as pd
import pytest


def test_csv_has_bom_and_dot_decimals_by_default(app):
    data = app.to_csv_bytes(pd.DataFrame({"Год": [2025], "Значение": [1.5]}))
    assert data.startswith(b"\xef\xbb\xbf") and data.decode("utf-8-sig").splitlines() == ["Год,Значение", "2025,1.5"]


def test_csv_for_russian_excel_uses_semicolon_and_decimal_comma(app):
    data = app.to_csv_bytes(pd.DataFrame({"Год": [2025], "Значение": [1.5]}), ";")
    assert data.decode("utf-8-sig").splitlines() == ["Год;Значение", "2025;1,5"]


def test_excel_has_one_sheet_per_table_and_skips_empty_ones(app):
    tables = {"Данные": pd.DataFrame({"a": [1, 2]}), "Пусто": pd.DataFrame(), "Рост": pd.DataFrame({"b": [3.5]})}
    sheets = pd.read_excel(io.BytesIO(app.to_excel_bytes(tables)), sheet_name=None)
    assert list(sheets) == ["Данные", "Рост"] and sheets["Данные"]["a"].tolist() == [1, 2]


def test_excel_sheet_names_are_cut_to_31_characters(app):
    name = "Очень длинное название листа для проверки ограничения Excel"
    sheets = pd.read_excel(io.BytesIO(app.to_excel_bytes({name: pd.DataFrame({"a": [1]})})), sheet_name=None)
    assert list(sheets) == [name[:31]]


def test_excel_is_none_without_data(app):
    assert app.to_excel_bytes({}) is None and app.to_excel_bytes({"x": pd.DataFrame()}) is None


def test_company_export_tables_for_real_data(app, ru_all):
    rows = ru_all[ru_all["ticker"] == "SBER"]
    w, unit, _ = app.make_wide(rows)
    tables = app.export_tables(w, unit, rows, "SBER")
    assert list(tables) == ["Сводка", "Данные", "Рост выручки", "Прибыльность"]
    summary = dict(zip(tables["Сводка"]["Параметр"], tables["Сводка"]["Значение"]))
    assert summary["Компания / тикер"] == "SBER" and summary["CAGR выручки, %"] == pytest.approx(29.04, abs=0.01)
    assert summary["Период CAGR выручки"] == "2022–2025"
    growth = tables["Рост выручки"]
    assert growth["Сигнал"].tolist() == ["н/д", "Рост", "Рост", "Рост"]
    # длинная таблица содержит все показатели, включая дивиденды и число акций, с источниками
    assert {"Выручка", "Чистая прибыль", "ROE", "Дивиденд на акцию", "Число акций"} <= set(tables["Данные"]["Показатель"])
    assert (tables["Данные"]["Источник"] != "").all()


def test_export_without_source_rows_uses_yahoo_label(app):
    w = pd.DataFrame({"year": [2024, 2025], "revenue": [100.0, 110.0], "net_income": [10.0, 12.0], "ebitda": [20.0, 22.0],
                      "roe": float("nan"), "has_estimate": False})
    tables = app.export_tables(w, "$", None, "AAPL")
    assert (tables["Данные"]["Источник"] == "Yahoo Finance (yfinance)").all()
    assert (tables["Данные"]["Единицы"] == "$").all() and "ROE" not in set(tables["Данные"]["Показатель"])
    assert tables["Прибыльность"]["Чистая маржа, %"].tolist() == pytest.approx([10.0, 10.91], abs=0.01)


def test_export_of_company_with_one_year_has_no_cagr(app):
    w = pd.DataFrame({"year": [2025], "revenue": [100.0], "net_income": [10.0], "ebitda": float("nan"),
                      "roe": float("nan"), "has_estimate": False})
    summary = dict(zip(*app.export_tables(w, "$", None, "X")["Сводка"].T.values))
    assert summary["CAGR выручки, %"] == "н/д"


def test_comparison_export_tables(app, ru_all):
    revenue = app.load_revenue_csv()
    problems = []
    rev = app.collect_revenue(["SBER", "GAZP"], [], revenue, problems)
    common = sorted(set.intersection(*[set(g["year"]) for _, g in rev.groupby("label")]))
    prof = app.collect_profit(["SBER", "GAZP"], [], ru_all)
    tables = app.comparison_export_tables(rev, common, prof)
    assert list(tables) == ["Выручка", "Рост и CAGR", "Прибыльность"] and problems == []
    cagr = tables["Рост и CAGR"].set_index("Компания")
    assert cagr.loc["SBER", "CAGR весь ряд, %"] == pytest.approx(29.04, abs=0.01)
    assert cagr.loc["GAZP", "Общий период"] == "2022–2025"


@pytest.mark.parametrize("name", ["Прогноз: методы", "a/b\\c", "Выручка [млрд]", "что?*", "'кавычки'"])
def test_excel_sheet_names_with_forbidden_characters_are_cleaned(app, name):
    sheets = pd.read_excel(io.BytesIO(app.to_excel_bytes({name: pd.DataFrame({"a": [1]})})), sheet_name=None)
    (title,) = list(sheets)
    assert title and not any(ch in title for ch in "\\/?*[]:") and not title.startswith("'") and len(title) <= 31


def test_excel_sheet_names_stay_unique_after_cleaning_and_truncation(app):
    long = "Очень длинное название листа для проверки ограничения"
    tables = {long + " А": pd.DataFrame({"a": [1]}), long + " Б": pd.DataFrame({"a": [2]}),
              "Прогноз: методы": pd.DataFrame({"a": [3]}), "Прогноз/методы": pd.DataFrame({"a": [4]})}
    sheets = pd.read_excel(io.BytesIO(app.to_excel_bytes(tables)), sheet_name=None)
    assert len(sheets) == 4 and len({n.lower() for n in sheets}) == 4
    assert sorted(int(df["a"].iloc[0]) for df in sheets.values()) == [1, 2, 3, 4]
