"""Экран «Сравнение компаний»."""

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from salesanalyzer import config
from salesanalyzer.config import GROWTH_DOWN, GROWTH_UP, MAX_COMPANIES
from salesanalyzer.core.adjustments import adjust_long
from salesanalyzer.core.comparison import index_revenue, parse_tickers, summary_table
from salesanalyzer.core.exports import comparison_export_tables, to_csv_bytes
from salesanalyzer.core.market import market_stats_ru, market_stats_us, market_summary_row
from salesanalyzer.core.prices import last_close, price_metrics_table
from salesanalyzer.core.profitability import latest_value
from salesanalyzer.data_sources.csv_store import load_macro, load_revenue_csv, load_ru_all
from salesanalyzer.data_sources.moex import get_moex_candles
from salesanalyzer.data_sources.yahoo import get_us_data, get_us_prices
from salesanalyzer.services import collect_prices, collect_profit, collect_revenue
from salesanalyzer.ui.components import adjustment_controls, render_adjustment_info, render_export


def comparison_view():
    st.write(f"Выберите от 2 до {MAX_COMPANIES} компаний. Российские берутся из {config.CSV_PATH.name}, "
             "зарубежные вводятся тикерами.")
    revenue_all = load_revenue_csv()
    ru_all = load_ru_all()
    ru_options = sorted(revenue_all["ticker"].unique()) if not revenue_all.empty else []

    c1, c2 = st.columns(2)
    with c1:
        default = [t for t in ("SBER", "GAZP", "YDEX") if t in ru_options]
        ru_sel = st.multiselect("Российские компании:", ru_options, default=default)
    with c2:
        us_text = st.text_input("Зарубежные тикеры через запятую (например: AAPL, MSFT):", value="")
    us_sel = parse_tickers(us_text)

    total = len(ru_sel) + len(us_sel)
    if total < 2:
        st.info("Выберите минимум две компании.")
        return
    if total > MAX_COMPANIES:
        st.warning(f"Выбрано {total} компаний, максимум — {MAX_COMPANIES}. Уберите лишние.")
        return

    ru_sel_rows = ru_all[ru_all["ticker"].isin(ru_sel)] if not ru_all.empty else ru_all
    to_rub, real, base_year, macro = adjustment_controls(list(ru_sel_rows["unit"].unique()), "cmp")
    ru_all_adj, adj_notes = adjust_long(ru_sel_rows, to_rub, real, base_year, macro)
    revenue_all_adj = (ru_all_adj[ru_all_adj["indicator"] == "revenue"]
                       .rename(columns={"value": "revenue"}).drop(columns=["indicator"]))
    render_adjustment_info(macro, to_rub, real, base_year, adj_notes)
    if (to_rub or real) and us_sel:
        st.caption("ℹ️ Зарубежные компании не пересчитываются: их значения остаются в долларах (российская инфляция к ним не применима).")

    problems = []
    with st.spinner("Загружаем данные..."):
        rev = collect_revenue(ru_sel, us_sel, revenue_all_adj, problems)
    if rev.empty:
        st.error("Не удалось получить данные о выручке ни по одной компании.")
        for p in problems:
            st.write(f"- {p}")
        return

    labels = list(rev["label"].unique())
    year_sets = [set(g["year"]) for _, g in rev.groupby("label")]
    common = sorted(set.intersection(*year_sets))

    prof = collect_profit(ru_sel, us_sel, ru_all_adj)

    tab_idx, tab_growth, tab_abs, tab_prof, tab_mkt, tab_price = st.tabs(
        ["Выручка (индекс)", "Рост и CAGR", "Выручка в единицах", "Прибыльность", "Мультипликаторы", "Цена акции (индекс)"])

    with tab_idx:
        st.caption("Индекс выручки: значение в базовом году = 100. Так можно сравнивать компании "
                   "с разными масштабами и валютами.")
        if common:
            base = st.selectbox("Базовый год (= 100):", common, index=0)
            idx = index_revenue(rev, base)
        else:
            st.warning("Нет года, по которому есть данные у всех компаний: каждая индексируется по "
                       "своему первому году, сравнивать осторожно.")
            idx = index_revenue(rev, None)
        if idx.empty:
            st.info("Недостаточно данных для индекса.")
        else:
            fig = px.line(idx, x="year", y="index", color="label", markers=True,
                          title="Динамика выручки (базовый год = 100)")
            fig.add_hline(y=100, line_dash="dot", line_color="gray")
            fig.update_layout(xaxis=dict(dtick=1), xaxis_title="Год", yaxis_title="Индекс", legend_title="")
            st.plotly_chart(fig, use_container_width=True)

    with tab_growth:
        st.dataframe(summary_table(rev, common), use_container_width=True)
        st.caption(f"Светофор: 🟢 от +{GROWTH_UP:.0f}%, 🟡 от {GROWTH_DOWN:.0f}% до +{GROWTH_UP:.0f}%, "
                   f"🔴 ниже {GROWTH_DOWN:.0f}%. Рост номинальный, в валюте отчетности каждой компании. "
                   "CAGR по общему периоду сопоставим между компаниями, CAGR по всему ряду — нет, "
                   "если длина рядов разная.")
        gdf = rev.dropna(subset=["growth"])
        if gdf.empty:
            st.info("Для годового роста нужны данные минимум за два соседних года.")
        else:
            fig = px.bar(gdf.assign(Год=gdf["year"].astype(str)), x="Год", y="growth", color="label",
                         barmode="group", text_auto=".1f", title="Рост выручки к предыдущему году, %")
            fig.update_layout(xaxis_type="category", yaxis_title="%", legend_title="")
            st.plotly_chart(fig, use_container_width=True)

    with tab_abs:
        st.caption("Абсолютные значения сравнимы только внутри одной единицы измерения, "
                   "поэтому компании сгруппированы по единицам.")
        for unit, g in rev.groupby("unit"):
            fig = px.bar(g.assign(Год=g["year"].astype(str)), x="Год", y="revenue", color="label",
                         barmode="group", text_auto=".4s", title=f"Выручка, {unit}")
            fig.update_layout(xaxis_type="category", yaxis_title=unit, legend_title="")
            st.plotly_chart(fig, use_container_width=True)
        if rev["is_estimate"].any():
            est = rev[rev["is_estimate"]].groupby("label")["year"].apply(lambda s: ", ".join(map(str, s)))
            st.caption("⚠️ Расчетные оценки: " + "; ".join(f"{k} ({v})" for k, v in est.items()))

    with tab_prof:
        if prof.empty:
            st.info("Нет данных о прибыльности по выбранным компаниям.")
        else:
            summary = []
            for label, g in prof.groupby("label", sort=False):
                rec = {"Компания": label}
                for col, name in (("net_margin", "Чистая маржа"), ("ebitda_margin", "EBITDA-маржа"), ("roe", "ROE")):
                    lv = latest_value(g, col)
                    rec[name] = "н/д" if lv is None else f"{lv[1]:.1f}% ({lv[0]})"
                summary.append(rec)
            st.dataframe(pd.DataFrame(summary).set_index("Компания"), use_container_width=True)

            metric = st.radio("Показатель на графике:", ["Чистая маржа", "EBITDA-маржа", "ROE"], horizontal=True)
            col = {"Чистая маржа": "net_margin", "EBITDA-маржа": "ebitda_margin", "ROE": "roe"}[metric]
            chart = prof.dropna(subset=[col])
            if chart.empty:
                st.info(f"Нет данных по показателю «{metric}» у выбранных компаний.")
            else:
                fig = px.bar(chart.assign(Год=chart["year"].astype(str)), x="Год", y=col, color="label",
                             barmode="group", text_auto=".1f", title=f"{metric}, %")
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.update_layout(xaxis_type="category", yaxis_title="%", legend_title="")
                st.plotly_chart(fig, use_container_width=True)
            st.caption("В скобках — последний год с данными. Маржа = показатель / выручка; для банков (Сбербанк, "
                       "Т-Технологии) «выручка» — операционный доход, EBITDA не применяется, вместо нее ROE. "
                       "Для Яндекса прибыль и EBITDA скорректированные; для Лукойла прибыль от продолжающейся деятельности. "
                       "Сравнивать уровни маржи между банками и нефинансовыми компаниями некорректно.")

    with tab_mkt:
        st.caption("Текущие мультипликаторы: последняя цена × число акций / показатели последнего отчетного года. "
                   "Российские компании — по данным файла (число акций и дивиденды), зарубежные — по Yahoo Finance. "
                   "P/S и P/E безразмерны, поэтому сравнимы между рынками; капитализация указана в валюте листинга.")
        macro_m = load_macro()
        rows_m, notes_all = [], []
        with st.spinner("Загружаем цены и считаем мультипликаторы..."):
            for t in ru_sel:
                try:
                    price_c, pdate_c = last_close(get_moex_candles(t))
                except requests.RequestException as e:
                    price_c, pdate_c = None, None
                    notes_all.append(f"{t}: ошибка MOEX ({e})")
                stats_c, payout_c, n_c = market_stats_ru(ru_all[ru_all["ticker"] == t], macro_m, price_c, pdate_c)
                notes_all += [f"{t}: {x}" for x in n_c]
                if stats_c is not None:
                    rows_m.append(market_summary_row(t, stats_c, payout_c))
            for t in us_sel:
                try:
                    data_c = get_us_data(t)
                    price_c, pdate_c = last_close(get_us_prices(t))
                except Exception as e:
                    notes_all.append(f"{t}: ошибка загрузки ({e})")
                    continue
                if data_c is None:
                    continue
                stats_c, payout_c, n_c = market_stats_us(data_c, price_c, pdate_c)
                notes_all += [f"{t}: {x}" for x in n_c]
                rows_m.append(market_summary_row(t, stats_c, payout_c))
        if not rows_m:
            st.info("Нет данных для расчета мультипликаторов.")
        else:
            mdf = pd.DataFrame(rows_m).set_index("Компания")
            st.dataframe(mdf, use_container_width=True)
            metric_m = st.radio("Показатель на графике:", ["P/S", "P/E", "Дивдоходность, %", "Payout, %"], horizontal=True,
                                key="mkt_metric")
            cm = mdf[metric_m].dropna()
            if cm.empty:
                st.info(f"Нет данных по показателю «{metric_m}».")
            else:
                fig = px.bar(cm.reset_index(), x="Компания", y=metric_m, text_auto=".2f", title=metric_m)
                fig.update_layout(yaxis_title=metric_m)
                st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Скачать таблицу (CSV)", key="dl_cmp_mkt", mime="text/csv",
                               data=to_csv_bytes(mdf.reset_index()), file_name="comparison_multiples.csv")
        for n in dict.fromkeys(notes_all):
            st.caption("⚠️ " + n)

    with tab_price:
        years = st.selectbox("Период:", [1, 2, 3], index=0, format_func=lambda y: f"{y} г.")
        with st.spinner("Загружаем цены..."):
            prices = collect_prices(ru_sel, us_sel, years, problems)
        if prices.empty:
            st.info("История цен недоступна.")
        else:
            start_date = prices["begin"].min().date()
            fig = px.line(prices, x="begin", y="index", color="label",
                          title=f"Цена закрытия (старт с {start_date} = 100)")
            fig.add_hline(y=100, line_dash="dot", line_color="gray")
            fig.update_layout(xaxis_title="Дата", yaxis_title="Индекс", legend_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Метрики доходности и риска")
            pm = price_metrics_table(prices)
            if not pm.empty:
                st.dataframe(pm.set_index("Компания"), use_container_width=True)
                st.caption("Считаются по ценам за выбранный период (с общей даты старта). Волатильность — "
                           "годовая; просадка — от максимума внутри периода. MA200 на коротком периоде может быть недоступна (н/д).")
                st.download_button("⬇️ Скачать метрики (CSV)", key="dl_cmp_pm", mime="text/csv",
                                   data=to_csv_bytes(pm), file_name="comparison_price_metrics.csv")
            st.download_button("⬇️ Скачать индекс цен (CSV)", key="dl_cmp_prices", mime="text/csv",
                               data=to_csv_bytes(prices.rename(columns={"begin": "Дата", "label": "Компания",
                                                                        "close": "Цена закрытия", "index": "Индекс"})),
                               file_name="comparison_prices.csv")
            st.caption("Цены каждой бумаги в ее валюте (₽ или $), поэтому сравниваются изменения в процентах, "
                       "а не сами цены. Старт — общая дата, с которой есть данные по всем выбранным бумагам.")

    st.divider()
    render_export(comparison_export_tables(rev, common, prof), "comparison")

    if problems:
        with st.expander("⚠️ Что не удалось загрузить"):
            for p in dict.fromkeys(problems):
                st.write(f"- {p}")
