"""Точка входа приложения. Запуск:  streamlit run streamlit_app.py

Здесь только сборка экрана: выбор режима и вызов нужного вида.
Расчеты лежат в salesanalyzer/core, источники данных в salesanalyzer/data_sources,
интерфейс в salesanalyzer/ui (см. README.md).
"""
from datetime import date

import streamlit as st

from salesanalyzer import config
from salesanalyzer.config import SEV_ERR
from salesanalyzer.core.validation import validate_financials
from salesanalyzer.data_sources.csv_store import read_raw_csv
from salesanalyzer.ui.comparison_view import comparison_view
from salesanalyzer.ui.quality_view import quality_view
from salesanalyzer.ui.single_view import single_view

st.set_page_config(page_title="Анализ Продаж Компаний", layout="wide")

st.title("📊 Анализатор продаж компаний (Мосбиржа и мировые рынки)")
mode = st.radio("Режим:", ["Одна компания", "Сравнение компаний", "Контроль данных"], horizontal=True)

if mode != "Контроль данных" and config.CSV_PATH.exists():  # быстрая проверка файла данных на каждом экране
    try:
        _issues, _ = validate_financials(read_raw_csv(), date.today().year)
        _n_err = int((_issues["Серьезность"] == SEV_ERR).sum())
        if _n_err:
            st.warning(f"⚠️ В файле {config.CSV_PATH.name} найдено ошибок: {_n_err}. Откройте режим «Контроль данных».")
    except Exception:
        pass

if mode == "Сравнение компаний":
    comparison_view()
elif mode == "Контроль данных":
    quality_view()
else:
    single_view()
