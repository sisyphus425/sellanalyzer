"""Вкладка «Прогноз»: прогноз выручки, чистой прибыли и EBITDA с диапазоном и проверкой на прошлом."""

import pandas as pd
import plotly.express as px
import streamlit as st

from salesanalyzer.config import FORECAST_HORIZONS, INDICATOR_NAMES
from salesanalyzer.core.exports import to_csv_bytes
from salesanalyzer.core.forecast import forecast_overview, forecast_series, methods_table
from salesanalyzer.core.formatting import fmt_value
from salesanalyzer.ui.compat import show_chart, show_table

REL_ICONS = {"низкая": "🟠", "средняя": "🟡", "недостаточно данных": "⚪"}
FORECAST_COLOR = "#F2B01E"
DISCLAIMER = ("Это сценарная оценка на основе небольшого числа исторических значений, а не гарантия и не инвестиционная "
              "рекомендация. Диапазон показывает разброс простых методов с поправкой на их ошибки в прошлом; реальный результат "
              "может выйти и за его пределы (смена отчётности, разовые события, курс валют).")


def _forecast_chart(res, name, unit, color):
    hist = pd.DataFrame({"Год": res["years"].astype(int).astype(str), "Значение": res["values"], "Тип": "Факт",
                         "Плюс": 0.0, "Минус": 0.0})
    sc = res["scenarios"]
    fc = pd.DataFrame({"Год": sc["Год"].astype(str), "Значение": sc["База"], "Тип": "Прогноз (база)",
                       "Плюс": sc["Высокий"] - sc["База"], "Минус": sc["База"] - sc["Низкий"]})
    fig = px.bar(pd.concat([hist, fc], ignore_index=True), x="Год", y="Значение", color="Тип", error_y="Плюс",
                 error_y_minus="Минус", text_auto=".4s", title=f"{name}: факт и прогноз (линии показывают диапазон), {unit}",
                 color_discrete_map={"Факт": color, "Прогноз (база)": FORECAST_COLOR})
    fig.update_layout(xaxis_type="category", yaxis_title=unit, legend_title="")
    show_chart(fig)


def render_forecast(w: pd.DataFrame, unit: str, color: str, key: str, estimates=None) -> dict:
    """Прогноз по одной компании. w: таблица по годам (year, revenue, net_income, ebitda).
    estimates: {показатель: годы с расчётными значениями}. Возвращает таблицы для экспорта."""
    options = {INDICATOR_NAMES[c]: c for c in ("revenue", "net_income", "ebitda") if c in w.columns and w[c].notna().any()}
    if not options:
        st.info("Нет данных для прогноза.")
        return {}
    c1, c2 = st.columns(2)
    with c1:
        name = st.selectbox("Показатель:", list(options), key=f"fc_ind_{key}")
    with c2:
        horizon = st.selectbox("Горизонт, лет:", list(FORECAST_HORIZONS), key=f"fc_h_{key}")
    col = options[name]
    series = w.dropna(subset=[col])
    res = forecast_series(series["year"], series[col], horizon, (estimates or {}).get(col, ()), nonnegative=col != "net_income")
    rel = res["reliability"]

    if res["scenarios"].empty:
        st.info("Прогноз не строится: " + "; ".join(rel["reasons"]) + ".")
        return {}

    sc, last_year, last_val = res["scenarios"], int(res["years"][-1]), float(res["values"][-1])
    first = sc.iloc[0]
    growth = (first["База"] / last_val - 1) * 100 if last_val > 0 else None
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{name}: база на {int(first['Год'])} г.", fmt_value(first["База"], unit),
              delta=None if growth is None else f"{growth:+.1f}% к {last_year} г.",
              help="Медиана трёх методов: линейный тренд, CAGR и затухающий рост.")
    m2.metric("Диапазон", f"{fmt_value(first['Низкий'], '')} – {fmt_value(first['Высокий'], unit)}",
              help="Разброс методов, расширенный на их типичную ошибку на прошлом.")
    m3.metric("Надёжность прогноза", f"{REL_ICONS.get(rel['label'], '')} {rel['label']}",
              help="Из-за малого числа значений выше «средней» надёжность не бывает.")

    if rel["label"] == "низкая":
        st.warning("Надёжность прогноза низкая: считайте его только грубым ориентиром. Причины перечислены под таблицей.")

    _forecast_chart(res, name, unit, color)

    sc_view = sc.assign(**{f"Рост базы к {last_year} г., %": [(b / last_val - 1) * 100 if last_val > 0 else None for b in sc["База"]]})
    show_table(sc_view.rename(columns={"Методов": "Методов в расчёте"}).set_index("Год").round(1))
    for reason in rel["reasons"]:
        st.caption("• " + reason)

    mt = methods_table(res)
    with st.expander("🔍 Методы и проверка на прошлом"):
        st.caption("Методы: линейный тренд (прямая по годам), CAGR (продолжение среднегодового темпа), затухающий рост "
                   "(средний темп с затуханием от года к году). Ниже каждый метод «прогнозирует» уже известные годы по более ранним данным.")
        show_table(mt.set_index("Метод").round(1))
        if not res["backtest_summary"].empty:
            show_table(res["backtest_summary"].set_index("Метод").round(1))
        if not res["backtest"].empty:
            show_table(res["backtest"].round(1).set_index("Год прогноза"))
    st.caption(DISCLAIMER)

    export = sc.assign(Показатель=name, Единицы=unit)[["Показатель", "Единицы", "Год", "Низкий", "База", "Высокий", "Методов"]]
    out = {"Прогноз": export}
    if not mt.empty:
        out["Прогноз - методы"] = mt
    if not res["backtest"].empty:
        out["Прогноз - проверка на прошлом"] = res["backtest"]
    return out


def render_forecast_comparison(rev: pd.DataFrame, prof: pd.DataFrame):
    """Сводка прогнозов по нескольким компаниям. rev: label, year, revenue, unit, is_estimate; prof: label, year, net_income, unit, has_estimate."""
    st.caption("Базовый прогноз = медиана трёх простых методов. Подробности, диапазоны и проверку на прошлом смотрите в режиме «Одна компания».")
    c1, c2 = st.columns(2)
    with c1:
        indicator = st.radio("Показатель:", ["Выручка", "Чистая прибыль"], horizontal=True, key="fc_cmp_ind")
    with c2:
        horizon = st.selectbox("Горизонт, лет:", list(FORECAST_HORIZONS), key="fc_cmp_h")
    series, units = {}, {}
    if indicator == "Выручка":
        for label, g in rev.groupby("label", sort=False):
            series[label] = (g["year"], g["revenue"], g.loc[g["is_estimate"], "year"])
            units[label] = g["unit"].iloc[-1]
    elif not prof.empty:
        for label, g in prof.dropna(subset=["net_income"]).groupby("label", sort=False):
            series[label] = (g["year"], g["net_income"], g.loc[g["has_estimate"], "year"])
            units[label] = g["unit"].iloc[-1]
    if not series:
        st.info("Нет данных для прогноза по выбранному показателю.")
        return
    overview = forecast_overview(series, horizon, nonnegative=indicator == "Выручка")
    overview.insert(2, "Единицы", overview["Компания"].map(units))
    show_table(overview.set_index("Компания").round(1))
    chart = overview.dropna(subset=["Рост базы к последнему году, %"])
    if chart.empty:
        st.info("Ни по одной компании прогноз не построен: недостаточно данных.")
    else:
        fig = px.bar(chart, x="Компания", y="Рост базы к последнему году, %", color="Надежность", text_auto=".1f",
                     title=f"{indicator}: базовый прогноз, рост к последнему году, %")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(yaxis_title="%", legend_title="")
        show_chart(fig)
    st.caption("Рост считается по базовому сценарию на последний год горизонта относительно последнего известного значения. "
               "У компаний с рядом короче трёх значений прогноз не строится. " + DISCLAIMER)
    st.download_button("⬇️ Скачать сводку прогнозов (CSV)", key="dl_cmp_forecast", mime="text/csv",
                       data=to_csv_bytes(overview), file_name="comparison_forecast.csv")
