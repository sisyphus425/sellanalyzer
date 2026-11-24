"""Боковая панель «Своя компания»: шаблон, загрузка файла, проверка, статус и сброс."""

import hashlib

import streamlit as st

from salesanalyzer.config import SEV_ERR, SEV_WARN
from salesanalyzer.core.exports import to_csv_bytes, to_excel_bytes
from salesanalyzer.core.validation import issues_for_display
from salesanalyzer.data_sources.user_upload import (TEMPLATE_HELP, clear_user_rows, prepare_rows, read_upload, set_user_rows,
                                                    template_frame, user_file_name, user_rows, user_tickers)
from salesanalyzer.ui.compat import rerun, show_table

ISSUES_KEY, DIGEST_KEY, UPLOADER_GEN = "user_upload_issues", "user_upload_digest", "user_upload_gen"


def render_user_data_sidebar():
    """Загрузка своей компании. Данные живут только в этой сессии браузера и на сервер не записываются."""
    with st.sidebar:
        st.header("📤 Своя компания")
        st.caption("Загрузите показатели своей компании (публичной или частной) в формате шаблона. Данные хранятся только "
                   "в этой вкладке браузера: после закрытия загрузите файл снова. На сервер они не записываются.")
        template = template_frame()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Шаблон CSV", data=to_csv_bytes(template), file_name="my_company_template.csv",
                               mime="text/csv", key="tpl_csv")
        with c2:
            xlsx = to_excel_bytes({"Данные": template, "Инструкция": TEMPLATE_HELP})
            if xlsx:
                st.download_button("Шаблон Excel", data=xlsx, file_name="my_company_template.xlsx", key="tpl_xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        gen = st.session_state.get(UPLOADER_GEN, 0)
        uploaded = st.file_uploader("Файл с данными (CSV или Excel)", type=["csv", "xlsx"], key=f"user_file_{gen}")
        if uploaded is not None:
            data = uploaded.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            if st.session_state.get(DIGEST_KEY) != digest:  # обрабатываем файл один раз, а не при каждом обновлении страницы
                st.session_state[DIGEST_KEY] = digest
                try:
                    raw = read_upload(data, uploaded.name)
                    rows, issues, _ = prepare_rows(raw)
                    st.session_state[ISSUES_KEY] = issues
                    if rows.empty:
                        clear_user_rows()
                        st.error("В файле нет ни одной корректной строки. Смотрите список проблем ниже.")
                    else:
                        set_user_rows(rows, uploaded.name)
                except ValueError as e:
                    st.session_state[ISSUES_KEY] = None
                    clear_user_rows()
                    st.error(f"Не удалось прочитать файл: {e}")

        tickers = user_tickers()
        issues = st.session_state.get(ISSUES_KEY)
        if tickers:
            rows = user_rows()
            st.success(f"Загружено из «{user_file_name()}»: {', '.join(tickers)} ({len(rows)} строк). "
                       "Выберите рынок «Россия» и введите тикер в режиме «Одна компания»; компания есть и в сравнении.")
            st.download_button("⬇️ Мои данные (CSV)", data=to_csv_bytes(rows), file_name="my_company.csv", mime="text/csv",
                               key="user_rows_dl")
            if st.button("Убрать загруженные данные", key="user_rows_clear"):
                clear_user_rows()
                st.session_state[ISSUES_KEY] = None
                st.session_state[DIGEST_KEY] = None
                st.session_state[UPLOADER_GEN] = gen + 1  # новый ключ сбрасывает выбранный файл
                rerun()
        if issues is not None and not issues.empty:
            n_err = int((issues["Серьезность"] == SEV_ERR).sum())
            n_warn = int((issues["Серьезность"] == SEV_WARN).sum())
            (st.warning if n_err else st.info)(f"Проверка файла: ошибок {n_err}, предупреждений {n_warn}. "
                                               "Строки с ошибками не загружены.")
            with st.expander("Проблемы в файле"):
                show_table(issues_for_display(issues)[["Серьезность", "Проверка", "Тикер", "Год", "Строка в файле", "Описание"]],
                           hide_index=True)
