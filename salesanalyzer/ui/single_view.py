"""Экран «Одна компания»."""

import streamlit as st

from salesanalyzer import config
from salesanalyzer.config import BLUE, GREEN, GREEN_LIGHT
from salesanalyzer.core.adjustments import adjust_long
from salesanalyzer.core.exports import export_tables
from salesanalyzer.core.market import market_stats_ru, market_stats_us
from salesanalyzer.core.prices import last_close
from salesanalyzer.core.profitability import make_wide
from salesanalyzer.data_sources.csv_store import load_macro, load_revenue_csv, load_ru_all
from salesanalyzer.data_sources.moex import get_moex_candles, get_moex_name
from salesanalyzer.data_sources.yahoo import get_us_data, get_us_prices
from salesanalyzer.ui.forecast import render_forecast
from salesanalyzer.ui.dividends import render_dividend_forecast
from salesanalyzer.core.dividends import policy_from_rows
from salesanalyzer.config import UNIT_TO_BN
from salesanalyzer.ui.price_range import render_price_range
from salesanalyzer.ui.whatif import render_whatif
from salesanalyzer.core.whatif import build_base
from salesanalyzer.ui.components import adjustment_controls, render_adjustment_info, render_export, render_market_stats, render_price, render_profitability, render_revenue, show_data_error
from salesanalyzer.data_sources.errors import DataSourceError


def single_view():
    st.write("Введите только тикер: для РФ — код Мосбиржи (SBER, GAZP, LKOH, YDEX, T, GMKN), "
             "для США — AAPL, TSLA и т.д.")
    col_market, col_input = st.columns(2)
    with col_market:
        market = st.radio("Рынок:", ["Россия (Мосбиржа)", "США / Международный"], horizontal=True)
    is_ru = market == "Россия (Мосбиржа)"
    with col_input:
        ticker = st.text_input("Тикер:", value="SBER" if is_ru else "AAPL",
                               key=f"ticker_{'ru' if is_ru else 'us'}").strip().upper()
    if not ticker:
        return

    if is_ru:
        revenue_all = load_revenue_csv()
        ru_all = load_ru_all()
        if revenue_all.empty:
            st.error(f"Файл {config.CSV_PATH.name} не найден или пуст. Он должен лежать в папке данных: {config.CSV_PATH.parent}")
            st.stop()

        rows = revenue_all[revenue_all["ticker"] == ticker]
        name = None
        try:
            name = get_moex_name(ticker)
        except DataSourceError as e:
            st.warning(f"{e} Название компании и цены недоступны; финансовые данные из файла показаны.")

        if name is None and rows.empty:
            st.error(f"❌ Тикер **{ticker}** не найден на Мосбирже и отсутствует в {config.CSV_PATH.name}.")
            st.stop()

        st.subheader(f"🏢 {name or ticker} ({ticker})")

        ru_orig = ru_all[ru_all["ticker"] == ticker]
        ru_long = ru_orig
        market_tables_, forecast_tables_, range_tables_, whatif_tables_, div_tables_ = {}, {}, {}, {}, {}
        if not ru_long.empty:
            to_rub, real, base_year, macro = adjustment_controls(list(ru_long["unit"].unique()), f"single_{ticker}")
            ru_long, adj_notes = adjust_long(ru_long, to_rub, real, base_year, macro)
            render_adjustment_info(macro, to_rub, real, base_year, adj_notes)
        rows = ru_long[ru_long["indicator"] == "revenue"].rename(columns={"value": "revenue"}).drop(columns=["indicator"])

        tab_rev, tab_prof, tab_fc, tab_div, tab_wi, tab_mkt, tab_price, tab_pr = st.tabs(["Выручка (МСФО)", "Прибыльность", "Прогноз", "Прогноз дивидендов", "Что если", "Мультипликаторы и дивиденды", "Цена акции", "Диапазон цены"])

        with tab_rev:
            if rows.empty:
                st.warning(f"Для **{ticker}** нет данных о выручке в {config.CSV_PATH.name}. "
                           "Добавьте строки в файл — код менять не нужно.")
                st.caption("Тикеры в файле: " + ", ".join(sorted(revenue_all["ticker"].unique())))
            else:
                units = rows["unit"].unique()
                unit = units[-1]
                if len(units) > 1:
                    st.warning("В файле для этого тикера указаны разные единицы измерения — проверьте колонку unit.")
                st.caption(f"Показатель: {rows['metric'].iloc[-1]}. Данные из файла {config.CSV_PATH.name}; "
                           "светлые столбцы — расчетные оценки.")
                render_revenue(rows, unit, GREEN, GREEN_LIGHT, f"Выручка {ticker} по годам")

        with tab_prof:
            ru_rows = ru_long
            if ru_rows.empty:
                st.warning(f"Для **{ticker}** нет данных о прибыли в {config.CSV_PATH.name}. "
                           "Добавьте строки (indicator: net_income, ebitda, roe) — код менять не нужно.")
            else:
                w, unit_p, mixed = make_wide(ru_rows)
                render_profitability(w, unit_p, GREEN, ru_rows, mixed)

        with tab_fc:
            if ru_long.empty:
                st.warning(f"Для **{ticker}** нет данных в {config.CSV_PATH.name}: прогноз не строится.")
            else:
                w_f, unit_f, _ = make_wide(ru_long)
                est_f = {ind: set(g["year"]) for ind, g in ru_long[ru_long["is_estimate"]].groupby("indicator")}
                forecast_tables_ = render_forecast(w_f, unit_f, GREEN, f"ru_{ticker}", est_f)

        with tab_div:
            if ru_orig.empty or not (ru_orig["indicator"] == "shares").any():
                st.warning(f"Для **{ticker}** в {config.CSV_PATH.name} нет числа акций (indicator = shares): прогноз дивидендов недоступен.")
            else:
                rows_d, _ = adjust_long(ru_orig, True, False, None, load_macro())
                w_d, unit_d, _ = make_wide(rows_d)
                shares_d = w_d.dropna(subset=["shares"])["shares"].iloc[-1] if "shares" in w_d.columns and w_d["shares"].notna().any() else None
                try:
                    price_d, _date_d = last_close(get_moex_candles(ticker))
                except DataSourceError as e:
                    price_d = None
                    st.warning(f"{e} Доходность не считается без цены.")
                est_d = {ind: set(g["year"]) for ind, g in rows_d[rows_d["is_estimate"]].groupby("indicator")}
                div_tables_ = render_dividend_forecast(w_d, UNIT_TO_BN.get(unit_d, 1.0), shares_d, price_d, policy_from_rows(ru_orig), f"ru_{ticker}", GREEN, est_d)

        with tab_wi:
            if ru_orig.empty:
                st.warning(f"Для **{ticker}** нет данных в {config.CSV_PATH.name}: калькулятор недоступен.")
            else:
                rows_w, _ = adjust_long(ru_orig, True, False, None, load_macro())   # в рубли по среднему курсу, без поправки на инфляцию
                w_w, unit_w, _ = make_wide(rows_w)
                try:
                    price_w, _date = last_close(get_moex_candles(ticker))
                except DataSourceError as e:
                    price_w = None
                    st.warning(f"{e} Цена не загружена: её можно ввести вручную.")
                base_w, notes_w = build_base(w_w, unit_w, price_w, "₽")
                rev_w = w_w.dropna(subset=["revenue"])
                whatif_tables_ = render_whatif(base_w, notes_w, (rev_w["year"], rev_w["revenue"]), f"ru_{ticker}", GREEN)

        with tab_mkt:
            if ru_orig.empty or not (ru_orig["indicator"] == "shares").any():
                st.warning(f"Для **{ticker}** в {config.CSV_PATH.name} нет числа акций (indicator = shares). "
                           "Добавьте строку — мультипликаторы и дивиденды появятся без изменения кода.")
            else:
                try:
                    price_m, pdate_m = last_close(get_moex_candles(ticker))
                except DataSourceError as e:
                    price_m, pdate_m = None, None
                    show_data_error(e, f"mkt_{ticker}")
                stats_m, payout_m, notes_m = market_stats_ru(ru_orig, load_macro(), price_m, pdate_m)
                market_tables_ = render_market_stats(stats_m, payout_m, notes_m, GREEN)

        with tab_price:
            try:
                render_price(get_moex_candles(ticker), ticker, "₽", GREEN)
            except DataSourceError as e:
                show_data_error(e, f"price_{ticker}")

        with tab_pr:
            try:
                range_prices = get_moex_candles(ticker)
            except DataSourceError as e:
                range_prices = None
                show_data_error(e, f"range_{ticker}")
            if range_prices is not None:
                range_tables_ = render_price_range(range_prices, ticker, "₽", GREEN, f"ru_{ticker}")

        ru_export_rows = ru_long
        if not ru_export_rows.empty:
            st.divider()
            w_e, unit_e, _ = make_wide(ru_export_rows)
            render_export({**export_tables(w_e, unit_e, ru_export_rows, ticker), **market_tables_, **forecast_tables_, **range_tables_, **whatif_tables_, **div_tables_}, f"{ticker}_financials")
    else:
        try:
            with st.spinner(f"Загружаем данные для {ticker}..."):
                data = get_us_data(ticker)
        except Exception as e:
            show_data_error(e, f"us_{ticker}")
            st.stop()

        if data is None:
            st.error(f"❌ Тикер **{ticker}** не найден.")
            return
        st.subheader(f"🏢 {data['name']}")
        site = data["website"]
        st.write(f"**Сектор:** {data['sector']} | **Сайт:** {f'[{site}]({site})' if site else '—'}")
        market_tables_, forecast_tables_, range_tables_, whatif_tables_ = {}, {}, {}, {}
        tab_rev, tab_prof, tab_fc, tab_wi, tab_mkt, tab_price, tab_pr = st.tabs(["Выручка", "Прибыльность", "Прогноз", "Что если", "Мультипликаторы и дивиденды", "Цена акции", "Диапазон цены"])

        with tab_rev:
            if data["df"] is None:
                st.warning(f"⚠️ Профиль {ticker} найден, но открытых данных по выручке нет.")
            else:
                render_revenue(data["df"].assign(is_estimate=False), "$", BLUE, BLUE,
                               f"История продаж {ticker} по годам")
        with tab_prof:
            if data["wide"].empty:
                st.warning(f"⚠️ Нет открытых данных о прибыли для {ticker}.")
            else:
                render_profitability(data["wide"], "$", BLUE)

        with tab_fc:
            if data["wide"].empty:
                st.warning(f"⚠️ Нет открытых данных для прогноза по {ticker}.")
            else:
                forecast_tables_ = render_forecast(data["wide"], "$", BLUE, f"us_{ticker}")

        with tab_wi:
            if data["wide"].empty:
                st.warning(f"⚠️ Нет открытых данных для калькулятора по {ticker}.")
            else:
                try:
                    price_w, _date = last_close(get_us_prices(ticker))
                except Exception as e:
                    price_w = None
                    st.warning(f"{e} Цена не загружена: её можно ввести вручную.")
                div_df = data.get("dividends")
                dps_by_year = dict(zip(div_df["year"], div_df["dps"])) if div_df is not None and not div_df.empty else None
                shares_w = data["shares"] / 1e6 if data.get("shares") else None
                base_w, notes_w = build_base(data["wide"], "$", price_w, data["currency"], shares_mn=shares_w, dps_by_year=dps_by_year)
                rev_w = data["wide"].dropna(subset=["revenue"])
                whatif_tables_ = render_whatif(base_w, notes_w, (rev_w["year"], rev_w["revenue"]), f"us_{ticker}", BLUE)

        with tab_mkt:
            try:
                price_m, pdate_m = last_close(get_us_prices(ticker))
            except Exception as e:
                price_m, pdate_m = None, None
                show_data_error(e, f"mkt_{ticker}")
            stats_m, payout_m, notes_m = market_stats_us(data, price_m, pdate_m)
            market_tables_ = render_market_stats(stats_m, payout_m, notes_m, BLUE)

        with tab_price:
            try:
                render_price(get_us_prices(ticker), ticker, data["currency"], BLUE)
            except Exception as e:
                show_data_error(e, f"price_{ticker}")

        with tab_pr:
            try:
                range_prices = get_us_prices(ticker)
            except Exception as e:
                range_prices = None
                show_data_error(e, f"range_{ticker}")
            if range_prices is not None:
                range_tables_ = render_price_range(range_prices, ticker, data["currency"], BLUE, f"us_{ticker}")

        if not data["wide"].empty:
            st.divider()
            render_export({**export_tables(data["wide"], "$", None, ticker), **market_tables_, **forecast_tables_, **range_tables_, **whatif_tables_}, f"{ticker}_financials")
