"""Своя компания: чтение файла пользователя, проверка, слияние с financials_ru.csv, хранение в сессии (data_sources/user_upload.py)."""
import importlib

import pandas as pd
import pytest

HEADER = "ticker,year,indicator,value,unit,metric,is_estimate,source,note"


@pytest.fixture
def up(app):
    return importlib.import_module("salesanalyzer.data_sources.user_upload")


def csv_bytes(*lines, sep=",", enc="utf-8-sig"):
    return "\n".join([HEADER.replace(",", sep), *lines]).encode(enc)


def test_template_is_valid(up):
    t = up.template_frame()
    rows, issues, n_err = up.prepare_rows(up.read_upload(t.to_csv(index=False).encode("utf-8-sig"), "t.csv"), 2026)
    assert n_err == 0 and len(rows) == len(t) and set(rows["ticker"]) == {"MYCO"}
    assert list(up.TEMPLATE_HELP["Колонка"]) == list(t.columns)


def test_read_csv_with_semicolon_and_decimal_comma(up):
    raw = up.read_upload(csv_bytes("myco;2025;revenue;1 234,5;млн ₽;Выручка;false;свои;", sep=";"), "x.csv")
    rows, _, n_err = up.prepare_rows(raw, 2026)
    assert n_err == 0 and rows["value"].tolist() == [1234.5] and rows["ticker"].tolist() == ["MYCO"]


def test_read_cp1251_csv(up):
    raw = up.read_upload(csv_bytes("ABC,2025,revenue,10,млн руб.,Выручка,false,свои данные,", enc="cp1251"), "x.csv")  # в cp1251 нет знака ₽
    assert raw["unit"].tolist() == ["млн руб."] and raw["source"].tolist() == ["свои данные"]


def test_read_excel_and_year_from_float(up, app):
    data = app.to_excel_bytes({"Данные": pd.DataFrame([["ABC", 2025.0, "revenue", 10.0, "млн ₽", "m", "false", "s", ""]],
                                                      columns=HEADER.split(","))})
    raw = up.read_upload(data, "my.xlsx")
    assert raw["year"].tolist() == ["2025"]
    rows, _, n_err = up.prepare_rows(raw, 2026)
    assert n_err == 0 and rows["year"].tolist() == [2025]


def test_missing_columns_and_garbage_are_reported(up):
    with pytest.raises(ValueError, match="нет колонок"):
        up.read_upload(b"ticker,year\nA,2025\n", "x.csv")
    with pytest.raises(ValueError):
        up.read_upload(b"\x00\x01\x02 not an excel", "x.xlsx")


def test_rows_with_errors_are_not_loaded(up):
    raw = up.read_upload(csv_bytes("ABC,2025,revenue,10,млн ₽,m,false,s,", "ABC,20x5,revenue,11,млн ₽,m,false,s,",
                                   "ABC,2024,revenue,abc,млн ₽,m,false,s,"), "x.csv")
    rows, issues, n_err = up.prepare_rows(raw, 2026)
    assert n_err == 2 and rows["year"].tolist() == [2025]
    assert {"Некорректный год", "Значение не число"} <= set(issues["Проверка"])


def test_merge_replaces_file_rows_for_same_ticker(up):
    file_rows = pd.DataFrame({"ticker": ["SBER", "SBER", "GAZP"], "year": [2024, 2025, 2025], "value": [1.0, 2.0, 3.0]})
    uploaded = pd.DataFrame({"ticker": ["SBER", "MYCO"], "year": [2025, 2025], "value": [20.0, 5.0]})
    out = up.merge_with_file(file_rows, uploaded)
    assert sorted(zip(out["ticker"], out["value"])) == [("GAZP", 3.0), ("MYCO", 5.0), ("SBER", 20.0)]
    assert up.merge_with_file(file_rows, pd.DataFrame()).equals(file_rows)
    assert up.merge_with_file(pd.DataFrame(), uploaded)["ticker"].tolist() == ["SBER", "MYCO"]


def test_session_storage_and_load_ru_all(up, app, monkeypatch, ru_all):
    state = {}
    monkeypatch.setattr(up.st, "session_state", state, raising=False)
    assert up.user_rows().empty and up.user_tickers() == []
    rows, _, _ = up.prepare_rows(up.read_upload(up.template_frame().to_csv(index=False).encode(), "t.csv"), 2026)
    up.set_user_rows(rows, "t.csv")
    assert up.user_tickers() == ["MYCO"] and up.user_file_name() == "t.csv"
    merged = app.load_ru_all()
    assert "MYCO" in set(merged["ticker"]) and len(merged) == len(ru_all) + len(rows)
    up.clear_user_rows()
    assert up.user_rows().empty and len(app.load_ru_all()) == len(ru_all)


def test_without_streamlit_session_nothing_breaks(up):
    assert up.user_rows().empty  # в тестах st — заглушка без session_state
    up.set_user_rows(pd.DataFrame({"ticker": ["X"]}), "x.csv")
    up.clear_user_rows()
