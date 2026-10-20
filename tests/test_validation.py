"""Проверка файла данных: каждая проверка срабатывает на испорченных данных и молчит на чистых."""
import pytest

TODAY = 2026
E, W, I = "🔴 Ошибка", "🟡 Предупреждение", "🔵 Информация"


def row(ticker="X", year=2025, indicator="revenue", value="100", unit="млрд ₽", metric="m", est="false", source="s", note=""):
    return [ticker, year, indicator, value, unit, metric, est, source, note]


def clean_rows():
    """Минимальный чистый набор: 3 года выручки и прибыли, число акций и дивиденд."""
    out = []
    for year, rev, ni in [(2023, 100, 10), (2024, 110, 11), (2025, 120, 12)]:
        out += [row(year=year, value=rev), row(year=year, indicator="net_income", value=ni)]
    out += [row(indicator="shares", value="500", unit="млн шт."), row(indicator="dps", value="3", unit="₽ на акцию")]
    return out


def check(app, mk_raw, rows):
    issues, ok = app.validate_financials(mk_raw(*rows), TODAY)
    return issues, ok


def found(issues, name, severity=None):
    sub = issues[issues["Проверка"] == name]
    return sub if severity is None else sub[sub["Серьезность"] == severity]


def test_clean_data_has_no_issues(app, mk_raw):
    issues, ok = check(app, mk_raw, clean_rows())
    assert issues.empty and len(ok) == len(clean_rows())


def test_missing_columns_are_an_error(app, mk_raw):
    raw = mk_raw(*clean_rows()).drop(columns=["unit"])
    issues, ok = app.validate_financials(raw, TODAY)
    assert found(issues, "Структура файла", E).shape[0] == 1 and ok.empty


def test_empty_file_is_an_error(app, mk_raw):
    issues, _ = app.validate_financials(mk_raw(), TODAY)
    assert found(issues, "Структура файла", E).shape[0] == 1


def test_bad_year_and_empty_ticker_rows_are_reported_and_ignored(app, mk_raw):
    issues, ok = check(app, mk_raw, clean_rows() + [row(year="20x5"), row(ticker="")])
    assert found(issues, "Некорректный год", E).shape[0] == 1 and found(issues, "Пустой тикер", E).shape[0] == 1
    assert len(ok) == len(clean_rows())


def test_value_with_space_and_decimal_comma_is_an_error(app, mk_raw):
    issues, ok = check(app, mk_raw, clean_rows() + [row(year=2022, value="4 089,0")])
    err = found(issues, "Значение не число", E)
    assert err.shape[0] == 1 and err["Год"].iloc[0] == 2022 and 2022 not in set(ok["year"])


def test_unknown_indicator_is_a_warning_and_row_is_ignored(app, mk_raw):
    issues, ok = check(app, mk_raw, clean_rows() + [row(indicator="profit")])
    assert found(issues, "Неизвестный показатель", W).shape[0] == 1 and "profit" not in set(ok["indicator"])


def test_duplicates_are_an_error_and_report_line_numbers(app, mk_raw):
    issues, ok = check(app, mk_raw, clean_rows() + [row(year=2025, value="999")])
    err = found(issues, "Дубликат", E)
    assert err.shape[0] == 1 and "6, 10" in err["Описание"].iloc[0]  # номера строк файла: 2025 г. в 6-й и 10-й строках
    assert ok[(ok["year"] == 2025) & (ok["indicator"] == "revenue")]["value"].iloc[0] == 999  # используется последняя


def test_mixed_money_units_are_an_error(app, mk_raw):
    issues, _ = check(app, mk_raw, clean_rows() + [row(year=2022, indicator="ebitda", value="30", unit="млн $")])
    assert found(issues, "Разные единицы", E).shape[0] == 1


def test_per_share_and_share_units_are_not_mixed_units(app, mk_raw):
    issues, _ = check(app, mk_raw, clean_rows())
    assert found(issues, "Разные единицы").empty


def test_roe_must_be_in_percent(app, mk_raw):
    issues, _ = check(app, mk_raw, clean_rows() + [row(indicator="roe", value="20", unit="млн")])
    assert found(issues, "Единицы ROE", W).shape[0] == 1


def test_unknown_unit_is_information(app, mk_raw):
    issues, _ = check(app, mk_raw, clean_rows() + [row(year=2022, indicator="ebitda", value="30", unit="тыс. евро")])
    assert found(issues, "Незнакомые единицы", I).shape[0] == 1


def test_gap_in_years_is_a_warning(app, mk_raw):
    rows = [r for r in clean_rows() if not (r[2] == "revenue" and r[1] == 2024)]
    issues, _ = check(app, mk_raw, rows)
    gap = found(issues, "Пропущенные годы", W)
    assert gap.shape[0] == 1 and "2024" in gap["Описание"].iloc[0]


def test_stale_data_is_a_warning(app, mk_raw):
    rows = [row(year=y, value=100 + y - 2020) for y in (2021, 2022, 2023)] + [row(indicator="shares", year=2023, unit="млн шт.", value="5"),
                                                                              row(indicator="dps", year=2023, unit="₽ на акцию", value="1"),
                                                                              row(indicator="net_income", year=2023, value="5")]
    issues, _ = check(app, mk_raw, rows)
    assert found(issues, "Устаревшие данные", W).shape[0] == 1


def test_estimate_without_note_is_a_warning_but_with_note_is_fine(app, mk_raw):
    issues, _ = check(app, mk_raw, clean_rows() + [row(year=2022, est="true")])
    assert found(issues, "Оценка без пояснения", W).shape[0] == 1
    issues, _ = check(app, mk_raw, clean_rows() + [row(year=2022, est="true", note="расчет: 100 / 2")])
    assert found(issues, "Оценка без пояснения").empty


def test_missing_source_unit_and_flag_are_warnings(app, mk_raw):
    issues, _ = check(app, mk_raw, clean_rows() + [row(year=2022, source=""), row(year=2021, unit=""), row(year=2020, est="maybe")])
    assert found(issues, "Нет источника", W).shape[0] == 1
    assert found(issues, "Нет единиц", W).shape[0] == 1 and found(issues, "Флаг оценки", W).shape[0] == 1


@pytest.mark.parametrize("indicator, value, name", [("revenue", "0", "Выручка ≤ 0"), ("revenue", "-5", "Выручка ≤ 0"),
                                                    ("shares", "0", "Число акций ≤ 0"), ("dps", "-1", "Дивиденд < 0")])
def test_impossible_values_are_errors(app, mk_raw, indicator, value, name):
    unit = {"revenue": "млрд ₽", "shares": "млн шт.", "dps": "₽ на акцию"}[indicator]
    rows = [r for r in clean_rows() if not (r[2] == indicator and r[1] == 2025)] + [row(indicator=indicator, value=value, unit=unit)]
    issues, _ = check(app, mk_raw, rows)
    assert found(issues, name, E).shape[0] == 1


def test_net_margin_above_100_percent_is_a_warning(app, mk_raw):
    rows = [r for r in clean_rows() if not (r[2] == "net_income" and r[1] == 2025)] + [row(indicator="net_income", value="500")]
    assert found(check(app, mk_raw, rows)[0], "Чистая маржа > 100%", W).shape[0] == 1


def test_ebitda_below_net_income_is_information(app, mk_raw):
    issues, _ = check(app, mk_raw, clean_rows() + [row(indicator="ebitda", value="5")])
    assert found(issues, "EBITDA < чистой прибыли", I).shape[0] == 1


def test_sharp_changes(app, mk_raw):
    rows = [r for r in clean_rows() if not (r[2] == "revenue" and r[1] == 2025)] + [row(value="400")]  # +264% к 2024
    assert found(check(app, mk_raw, rows)[0], "Резкое изменение", W).shape[0] == 1
    rows = [r for r in clean_rows() if not (r[2] == "net_income" and r[1] == 2025)] + [row(indicator="net_income", value="60")]  # +445%
    assert found(check(app, mk_raw, rows)[0], "Резкое изменение", I).shape[0] == 1


def test_coverage_hints(app, mk_raw):
    rows = [r for r in clean_rows() if r[2] not in ("shares", "dps")]
    issues, _ = check(app, mk_raw, rows)
    assert found(issues, "Нет числа акций", I).shape[0] == 1
    rows = [r for r in clean_rows() if r[2] != "dps"]
    assert found(check(app, mk_raw, rows)[0], "Нет дивидендов", I).shape[0] == 1
    rows = [r for r in clean_rows() if r[2] == "revenue" and r[1] == 2025]
    issues, _ = check(app, mk_raw, rows)
    assert found(issues, "Короткий ряд", I).shape[0] == 1 and found(issues, "Нет чистой прибыли", I).shape[0] == 1


def test_issues_are_sorted_by_severity(app, mk_raw):
    issues, _ = check(app, mk_raw, clean_rows() + [row(year="20x5"), row(year=2022, source=""), row(year=2021, indicator="ebitda", value="1", unit="тыс. евро")])
    order = {E: 0, W: 1, I: 2}
    levels = [order[s] for s in issues["Серьезность"]]
    assert levels == sorted(levels) and set(issues["Серьезность"]) == {E, W, I}


def test_coverage_matrix_marks_published_estimates_and_gaps(app, mk_raw):
    rows = [row(year=2023), row(year=2025, est="true", note="расчет")]
    _, ok = check(app, mk_raw, rows)
    cov = app.coverage_matrix(ok).set_index("Показатель")
    assert cov.loc["Выручка", "2023"] == "✓" and cov.loc["Выручка", "2024"] == "" and cov.loc["Выручка", "2025"] == "≈"
    assert app.coverage_matrix(ok.iloc[0:0]).empty
