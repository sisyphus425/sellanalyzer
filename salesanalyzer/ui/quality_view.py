"""Экран «Контроль данных»."""

from datetime import date, datetime

import pandas as pd
import streamlit as st

from salesanalyzer import config
from salesanalyzer.config import INDICATOR_NAMES, SEV_ERR, SEV_ORDER, SEV_WARN
from salesanalyzer.core.adjustments import price_levels
from salesanalyzer.core.validation import coverage_matrix, validate_financials
from salesanalyzer.data_sources.csv_store import load_macro, read_raw_csv
from salesanalyzer.ui.components import render_export
from salesanalyzer.ui.compat import show_table


def quality_view():
    st.write(f"Проверка файла **{config.CSV_PATH.name}**: структура, форматы, дубликаты, пропуски, единицы измерения, "
             "расчетные оценки и подозрительные значения. Файл перечитывается при каждом обновлении страницы, "
             "поэтому после правки CSV достаточно обновить страницу.")
    if not config.CSV_PATH.exists():
        st.error(f"Файл {config.CSV_PATH.name} не найден. Он должен лежать рядом с приложением: {config.CSV_PATH.parent}")
        return
    try:
        raw = read_raw_csv()
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")
        return

    issues, ok = validate_financials(raw, date.today().year)
    n = {sev: int((issues["Серьезность"] == sev).sum()) for sev in SEV_ORDER}

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Строк в файле", len(raw))
    m2.metric("Компаний", ok["ticker"].nunique() if not ok.empty else 0)
    m3.metric("Годы", f"{ok['year'].min()}–{ok['year'].max()}" if not ok.empty else "н/д")
    m4.metric("Расчетных оценок", int(ok["is_estimate"].sum()) if not ok.empty else 0)
    m5.metric("Ошибок / предупр.", f"{n[SEV_ERR]} / {n[SEV_WARN]}")
    st.caption("Файл изменен: " + datetime.fromtimestamp(config.CSV_PATH.stat().st_mtime).strftime("%d.%m.%Y %H:%M"))

    if n[SEV_ERR]:
        st.error(f"Найдено ошибок: {n[SEV_ERR]}. Часть строк приложение игнорирует или обрабатывает неверно. Исправьте файл.")
    elif n[SEV_WARN]:
        st.warning(f"Критичных ошибок нет, но есть предупреждения: {n[SEV_WARN]}. Стоит проверить.")
    else:
        st.success("Ошибок и предупреждений нет. Информационные заметки (если есть) — ниже.")

    coverage = coverage_matrix(ok)
    est_tbl = pd.DataFrame()
    if not ok.empty and ok["is_estimate"].any():
        e = ok[ok["is_estimate"]].sort_values(["ticker", "indicator", "year"])
        est_tbl = pd.DataFrame({
            "Тикер": e["ticker"], "Год": e["year"], "Показатель": e["indicator"].map(INDICATOR_NAMES),
            "Значение": e["value"], "Единицы": e["unit"], "Источник": e["source"], "Комментарий": e["note"]})

    tab_issues, tab_cov, tab_est, tab_raw = st.tabs(["Проблемы", "Покрытие данных", "Расчетные оценки", "Сырые данные"])
    with tab_issues:
        if issues.empty:
            st.info("Проблем не найдено.")
        else:
            chosen = st.multiselect("Показывать:", list(SEV_ORDER), default=list(SEV_ORDER))
            view = issues[issues["Серьезность"].isin(chosen)]
            show_table(view, hide_index=True)
            st.caption("Номер строки считается по файлу: строка 1 — заголовок. Информационные заметки не означают ошибку: "
                       "например, резкий рост прибыли может быть реальным, но его стоит сверить с источником.")
    with tab_cov:
        if coverage.empty:
            st.info("Нет корректных данных для матрицы покрытия.")
        else:
            show_table(coverage, hide_index=True)
            st.caption("✓ — значение есть, ≈ — расчетная оценка, пусто — данных нет.")
    with tab_est:
        if est_tbl.empty:
            st.info("Расчетных оценок в файле нет.")
        else:
            show_table(est_tbl, hide_index=True)
    with tab_raw:
        show_table(raw, hide_index=True)

    with st.expander(f"🌐 Макроданные ({config.MACRO_PATH.name})"):
        macro = load_macro()
        if not (macro["usd_rub"] or macro["cpi"]):
            st.warning(f"Файл {config.MACRO_PATH.name} не найден или пуст: пересчет в рубли и поправка на инфляцию недоступны.")
        else:
            levels = price_levels(macro["cpi"])
            fx_years, cpi_years = sorted(macro["usd_rub"]), sorted(macro["cpi"])
            st.write(f"Курс ЦБ: {fx_years[0]}–{fx_years[-1]} ({len(fx_years)} г.). "
                     f"Инфляция: {cpi_years[0]}–{cpi_years[-1]} ({len(cpi_years)} г.).")
            if len(levels) < len(cpi_years):
                st.warning("В ряду инфляции есть пропущенные годы: цепочка прерывается, поправка доступна не для всех лет.")
            data_years = set(ok["year"]) if not ok.empty else set()
            lacking_fx = sorted(data_years - set(macro["usd_rub"]))
            lacking_cpi = sorted(data_years - set(levels))
            if lacking_fx:
                st.info("Курса ЦБ нет за годы из данных: " + ", ".join(map(str, lacking_fx)))
            if lacking_cpi:
                st.info("Инфляции нет за годы из данных: " + ", ".join(map(str, lacking_cpi)))

    st.divider()
    render_export({"Проблемы": issues, "Покрытие": coverage, "Расчетные оценки": est_tbl}, "data_quality")
