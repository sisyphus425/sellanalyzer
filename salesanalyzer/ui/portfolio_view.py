"""Экран «Портфель»: состав (количество бумаг или доли), стоимость в рублях, доходность и риск, сравнение с индексом,
вклад бумаг, дивидендный поток и сценарии стоимости. Портфель хранится только в сессии браузера; его можно скачать и загрузить файлом."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from salesanalyzer.config import BENCHMARKS, GREEN, PRICE_HISTORY_YEARS, PRICE_PERIOD_OPTIONS
from salesanalyzer.core.benchmark import beta_label, compare_with_benchmark
from salesanalyzer.core.exports import to_csv_bytes
from salesanalyzer.core.portfolio import (build_panel, contributions, correlation_matrix, dividend_calendar, holdings_table,
                                          monthly_dividends, normalize_positions, portfolio_dividend_events, read_portfolio_csv,
                                          value_series, weights_to_qty)
from salesanalyzer.core.prices import add_price_indicators, price_metrics
from salesanalyzer.core.valuation import curve_yield
from salesanalyzer.data_sources.errors import DataSourceError
from salesanalyzer.data_sources.moex import get_moex_index_candles, get_ofz_curve
from salesanalyzer.services import collect_portfolio
from salesanalyzer.ui.compat import rerun, show_chart, show_table
from salesanalyzer.ui.components import render_export
from salesanalyzer.ui.price_range import render_price_range

MODE_QTY, MODE_WEIGHT = "Количество бумаг", "Доли, %"
MARKETS = {"РФ": "ru", "США": "us"}
STATE_DF, STATE_GEN = "portfolio_table", "portfolio_editor_gen"
GRAY = "#8A8F98"
DEFAULT = pd.DataFrame({"Тикер": ["SBER", "LKOH", "YDEX", "GMKN"], "Рынок": ["РФ"] * 4,
                        "Количество": [100.0, 5.0, 10.0, 200.0], "Доля, %": [40.0, 25.0, 20.0, 15.0],
                        "Цена покупки": [np.nan] * 4})
FILE_COLS = {"Тикер": "ticker", "Рынок": "market", "Количество": "qty", "Доля, %": "weight", "Цена покупки": "buy_price"}


def _pct(v, sign=False):
    return "н/д" if v is None or pd.isna(v) else (f"{v:+.1f}%" if sign else f"{v:.1f}%")


def read_portfolio_file(data: bytes) -> pd.DataFrame:
    """Файл портфеля -> таблица редактора (колонки DEFAULT)."""
    df = read_portfolio_csv(data)
    back = {v: k for k, v in FILE_COLS.items()}
    out = df.rename(columns=back)
    out["Рынок"] = df["market"].map({v: k for k, v in MARKETS.items()})
    return out[list(DEFAULT.columns)]


def _input_block():
    """Ввод состава. Возвращает (таблица позиций для расчета, режим, сумма для режима долей)."""
    if STATE_DF not in st.session_state:
        st.session_state[STATE_DF] = DEFAULT.copy()
    c1, c2 = st.columns([2, 2])
    with c1:
        mode = st.radio("Как задан состав:", [MODE_QTY, MODE_WEIGHT], horizontal=True, key="pf_mode",
                        help="«Количество бумаг» — ваш реальный портфель. «Доли» — модельный портфель: доли переводятся в количество "
                             "бумаг на указанную сумму по текущим ценам.")
    amount = None
    with c2:
        if mode == MODE_WEIGHT:
            amount = st.number_input("Сумма портфеля, ₽", min_value=1_000.0, value=1_000_000.0, step=100_000.0, key="pf_amount")
    cols = ["Тикер", "Рынок", "Количество", "Цена покупки"] if mode == MODE_QTY else ["Тикер", "Рынок", "Доля, %"]
    gen = st.session_state.get(STATE_GEN, 0)
    # Исходные данные редактора не меняются, пока не загружен файл и не переключен режим: иначе Streamlit сбрасывает правки.
    if st.session_state.get("pf_last_mode") != mode:
        for k in [k for k in st.session_state.keys() if str(k).startswith("pf_base_")]:
            del st.session_state[k]
        st.session_state["pf_last_mode"] = mode
    base_key = f"pf_base_{gen}_{mode}"
    if base_key not in st.session_state:
        st.session_state[base_key] = st.session_state[STATE_DF][cols].reset_index(drop=True)
    edited = st.data_editor(
        st.session_state[base_key], num_rows="dynamic", hide_index=True, key=f"pf_editor_{gen}_{mode}",
        column_config={
            "Тикер": st.column_config.TextColumn("Тикер", help="SBER, LKOH, AAPL…", required=True),
            "Рынок": st.column_config.SelectboxColumn("Рынок", options=list(MARKETS), required=True, default="РФ"),
            "Количество": st.column_config.NumberColumn("Количество", min_value=0.0, help="Число бумаг (можно дробное)"),
            "Доля, %": st.column_config.NumberColumn("Доля, %", min_value=0.0, max_value=100.0,
                                                     help="Если сумма долей не 100%, они нормируются"),
            "Цена покупки": st.column_config.NumberColumn("Цена покупки", min_value=0.0,
                                                          help="Необязательно; в валюте бумаги. Нужна для прибыли/убытка"),
        })
    table = edited.reindex(columns=list(DEFAULT.columns)).reset_index(drop=True)
    prev = st.session_state[STATE_DF].dropna(subset=["Тикер"]).drop_duplicates("Тикер", keep="last").set_index("Тикер")
    for c in DEFAULT.columns:
        if c not in cols:  # колонки другого режима берем из прежней таблицы по тикеру
            table[c] = table["Тикер"].map(prev[c].to_dict()) if c in prev.columns else np.nan
    st.session_state[STATE_DF] = table

    f1, f2 = st.columns(2)
    with f1:
        file_df = st.session_state[STATE_DF].rename(columns=FILE_COLS).assign(
            market=lambda d: d["market"].map(MARKETS).fillna("ru"))
        st.download_button("⬇️ Сохранить портфель (CSV)", data=to_csv_bytes(file_df), file_name="portfolio.csv", mime="text/csv",
                           key="pf_save")
    with f2:
        up = st.file_uploader("Загрузить портфель (CSV)", type=["csv"], key=f"pf_upload_{gen}")
        if up is not None:
            try:
                st.session_state[STATE_DF] = read_portfolio_file(up.getvalue())
                st.session_state[STATE_GEN] = gen + 1
                rerun()
            except Exception as e:
                st.error(f"Не удалось прочитать файл портфеля: {e}")

    src = st.session_state[STATE_DF]
    positions = pd.DataFrame({"ticker": src["Тикер"], "market": src["Рынок"].map(MARKETS),
                              "qty": src["Количество"] if mode == MODE_QTY else src["Доля, %"],
                              "buy_price": src["Цена покупки"] if mode == MODE_QTY else np.nan})
    return normalize_positions(positions), mode, amount


def _performance(prices, tr, qty, color):
    """Доходность и риск портфеля за период; сравнение с индексом МосБиржи. Возвращает таблицы для выгрузки."""
    v, t = value_series(prices, qty), value_series(tr, qty)
    mp, mt = price_metrics(add_price_indicators(v)), price_metrics(add_price_indicators(t))
    if mp is None or mt is None:
        st.info("Недостаточно данных за выбранный период.")
        return {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Доходность по цене", _pct(mp["total"]))
    c2.metric("С дивидендами", _pct(mt["total"]), delta=f"{mt['total'] - mp['total']:+.1f} п.п. дивиденды",
              help="Дивиденды каждой бумаги реинвестируются в нее же в день отсечки. Налог не учитывается.")
    c3.metric("Годовая (с дивидендами)", _pct(mt["annual"]))
    c4.metric("Волатильность (год)", _pct(mt["vol"]), help="По стоимости портфеля с дивидендами: дивидендные гэпы не завышают риск.")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Макс. просадка", _pct(mt["max_dd"]), help=f"С дивидендами: пик {mt['dd_peak']:%d.%m.%Y}, дно {mt['dd_trough']:%d.%m.%Y}.")
    d2.metric("Текущая просадка", _pct(mt["cur_dd"]))
    base = v["close"].iloc[0]
    long = pd.DataFrame({"Дата": list(v["begin"]) * 2, "Индекс": list(v["close"] / base * 100) + list(t["close"] / base * 100),
                         "Серия": ["Только цена"] * len(v) + ["С дивидендами"] * len(t)})

    tables = {"Портфель - стоимость": pd.DataFrame({"Дата": v["begin"], "Стоимость, ₽": v["close"], "С дивидендами, ₽": t["close"]})}
    total_mode = st.radio("Сравнение с индексом:", ["С дивидендами (против MCFTR)", "По цене (против IMOEX)"], horizontal=True,
                          key="pf_bench_mode")
    total = total_mode.startswith("С дивидендами")
    secid, bname = BENCHMARKS["ru"]["total" if total else "price"]
    try:
        bench = get_moex_index_candles(secid, PRICE_HISTORY_YEARS)
        m, frame = compare_with_benchmark(t if total else v, bench[bench["begin"] >= v["begin"].min()])
    except DataSourceError as e:
        st.warning(f"Индекс {bname} не загружен: {e}")
        m, frame = None, pd.DataFrame()
    if m is not None:
        d3.metric(f"Разница с {secid}", f"{m['excess']:+.1f} п.п.", help=f"Доходность портфеля минус {bname} за общий период.")
        d4.metric("Бета к индексу", "н/д" if m["beta"] is None else f"{m['beta']:.2f}",
                  help=f"{beta_label(m['beta']).capitalize()}. Корреляция с индексом: "
                       + ("н/д" if m["corr"] is None else f"{m['corr']:.2f}") + ".")
        b = frame.set_index("begin")["bench"]
        long = pd.concat([long, pd.DataFrame({"Дата": b.index, "Индекс": b.to_numpy(), "Серия": bname})], ignore_index=True)
        tables["Портфель - сравнение с индексом"] = frame.rename(columns={"begin": "Дата", "stock": "Портфель", "bench": bname,
                                                                          "relative": "Относительная сила"})
    fig = px.line(long, x="Дата", y="Индекс", color="Серия", title="Портфель и индекс (старт = 100)",
                  color_discrete_map={"Только цена": GRAY, "С дивидендами": color, bname: "#1f2937"})
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.update_layout(legend_title="")
    show_chart(fig)
    dd = (t["close"] / t["close"].cummax() - 1) * 100
    dfig = px.area(x=t["begin"], y=dd, title="Просадка портфеля (с дивидендами), %", color_discrete_sequence=["#D64545"])
    dfig.update_layout(xaxis_title="Дата", yaxis_title="%")
    show_chart(dfig)
    st.caption("История — «если бы нынешний состав держали весь период»: стоимость = Σ количество × цена в рублях "
               "(валютные бумаги пересчитаны по курсу Мосбиржи на каждую дату). Ребалансировка, покупки и продажи не учитываются.")
    return tables


def _contributions_block(prices, tr, qty, color):
    c = contributions(prices, tr, qty)
    if c.empty:
        return {}
    left, right = st.columns(2)
    with left:
        fig = px.bar(c, x="Тикер", y="Вклад в доходность с дивидендами, п.п.", text_auto=".1f",
                     title="Вклад в доходность (с дивидендами), п.п.", color_discrete_sequence=[color])
        show_chart(fig)
    with right:
        long = c.melt(id_vars="Тикер", value_vars=["Доля сейчас, %", "Вклад в риск, %"], var_name="Показатель", value_name="%")
        fig = px.bar(long, x="Тикер", y="%", color="Показатель", barmode="group", text_auto=".0f",
                     title="Доля в портфеле и вклад в риск, %", color_discrete_sequence=[GRAY, "#D64545"])
        show_chart(fig)
    show_table(c.set_index("Тикер").round(2))
    st.caption("Вклад в доходность — какую часть доходности портфеля (в п.п. от стоимости на старте) дала бумага; сумма вкладов равна "
               "доходности портфеля. Вклад в риск — доля бумаги в дисперсии портфеля по текущим долям: бумага с долей 20% и "
               "вкладом в риск 40% «раскачивает» портфель вдвое сильнее своего веса.")
    out = {"Портфель - вклад бумаг": c}
    corr = correlation_matrix(tr)
    if not corr.empty:
        fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                        title="Корреляция дневных доходностей")
        show_chart(fig)
        st.caption("Чем ниже корреляция между бумагами, тем сильнее они гасят колебания друг друга (диверсификация).")
        out["Портфель - корреляции"] = corr.reset_index().rename(columns={"index": "Тикер"})
    return out


def _dividends_block(data, qty, value, last_date, color):
    cal = dividend_calendar(data["orig_dividends"], qty, data["last_rate"], last_date)
    if cal.empty:
        st.info("Ожидаемых выплат на 12 месяцев нет: за последний год бумаги портфеля не платили дивиденды, объявленных выплат нет "
                "(для РФ даты берутся из dividends_ru.csv, у YDEX и T их нет).")
        return {}
    total = float(cal["Сумма, ₽"].sum())
    yld = total / value * 100 if value > 0 else None
    try:
        ofz = curve_yield(get_ofz_curve(None), 1)
    except DataSourceError:
        ofz = None
    c1, c2, c3 = st.columns(3)
    c1.metric("Ожидаемые дивиденды за 12 мес.", f"{total:,.0f} ₽")
    c2.metric("Дивидендная доходность портфеля", _pct(yld), help="Ожидаемые выплаты / текущая стоимость портфеля.")
    c3.metric("ОФЗ 1 год", _pct(ofz), delta=None if ofz is None or yld is None else f"{yld - ofz:+.2f} п.п. портфель к ОФЗ")
    monthly = monthly_dividends(cal)
    fig = px.bar(monthly.assign(Месяц=monthly["Месяц"].dt.strftime("%Y-%m")), x="Месяц", y="Сумма, ₽", text_auto=",.0f",
                 title="Ожидаемые дивиденды по месяцам, ₽", color_discrete_sequence=[color])
    fig.update_layout(xaxis_type="category")
    show_chart(fig)
    view = cal.assign(**{"Дата отсечки": pd.to_datetime(cal["Дата отсечки"]).dt.strftime("%d.%m.%Y")})
    show_table(view.round(2), hide_index=True)
    st.caption("«Как год назад» — допущение: выплаты последних 12 месяцев повторены через год с той же суммой; компания может "
               "изменить сумму, дату или отменить выплату. Валютные выплаты пересчитаны по текущему курсу. Налог не учитывается.")
    return {"Портфель - дивиденды": cal}


def portfolio_view():
    st.write("Задайте состав портфеля: российские бумаги — тикеры Мосбиржи, зарубежные — тикеры Yahoo Finance. "
             "Все суммы пересчитываются в рубли. Портфель хранится только в этой вкладке браузера: сохраните его файлом.")
    positions, mode, amount = _input_block()
    if positions.empty:
        st.info("Добавьте в таблицу хотя бы одну бумагу с количеством (или долей) больше нуля.")
        return
    years = st.selectbox("Период истории:", list(PRICE_PERIOD_OPTIONS), index=len(PRICE_PERIOD_OPTIONS) - 1,
                         format_func=lambda y: f"{y} г.", key="pf_years")
    problems = []
    with st.spinner("Загружаем цены, дивиденды и курсы валют..."):
        data = collect_portfolio(positions, PRICE_HISTORY_YEARS, problems)
    pos = data["positions"]
    if pos.empty:
        st.error("Ни по одной бумаге не удалось получить цены.")
        for p in problems:
            st.write(f"- {p}")
        return
    if mode == MODE_WEIGHT:
        rub_price = pd.Series({t: data["last_price"][t] * data["last_rate"][t] for t in pos["ticker"]})
        pos = pos.assign(qty=weights_to_qty(pos.set_index("ticker")["qty"], rub_price, amount).to_numpy())
    qty = pos.set_index("ticker")["qty"]

    holdings = holdings_table(pos, data["last_price"], data["last_rate"], data["currency"])
    value = float(holdings["Стоимость, ₽"].sum())
    full_prices, full_tr = build_panel(data["rub"])
    prices, tr = build_panel(data["rub"], years)
    if prices.empty:
        st.error("У бумаг нет общего периода цен.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Стоимость портфеля", f"{value:,.0f} ₽")
    m2.metric("Бумаг", len(holdings))
    pnl = holdings["Прибыль/убыток, ₽"].dropna()
    if mode == MODE_QTY and not pnl.empty:
        cost = float((holdings["Стоимость, ₽"] - holdings["Прибыль/убыток, ₽"]).loc[pnl.index].sum())
        m3.metric("Прибыль/убыток (где указана цена покупки)", f"{pnl.sum():+,.0f} ₽",
                  delta=f"{pnl.sum() / cost * 100:+.1f}%" if cost > 0 else None,
                  help="К цене покупки, в рублях по текущему курсу (без учета дивидендов и комиссий).")
    if prices.index[0] > prices.index[-1] - pd.DateOffset(years=years) + pd.Timedelta(days=10):
        st.caption(f"ℹ️ Общая история всех бумаг начинается {prices.index[0]:%d.%m.%Y}: период короче выбранного.")

    tabs = st.tabs(["Состав", "Доходность и риск", "Вклад бумаг", "Дивиденды", "Диапазон стоимости"])
    tables = {"Портфель - состав": holdings}
    with tabs[0]:
        left, right = st.columns([3, 2])
        with left:
            show_table(holdings.set_index("Тикер").round(2))
        with right:
            fig = px.pie(holdings, names="Тикер", values="Стоимость, ₽", title="Доли по стоимости", hole=0.45)
            show_chart(fig)
        by_cur = holdings.groupby("Валюта", as_index=False)["Стоимость, ₽"].sum()
        if len(by_cur) > 1:
            st.caption("По валютам: " + "; ".join(f"{r['Валюта']} {r['Стоимость, ₽'] / value * 100:.0f}%" for _, r in by_cur.iterrows()))
    with tabs[1]:
        tables.update(_performance(prices, tr, qty, GREEN))
    with tabs[2]:
        tables.update(_contributions_block(prices, tr, qty, GREEN))
    with tabs[3]:
        tables.update(_dividends_block(data, qty, value, prices.index[-1], GREEN))
    with tabs[4]:
        st.caption("Сценарии стоимости нынешнего состава портфеля, построенные по истории его стоимости за весь загруженный период "
                   "(до 3 лет) той же моделью, что и вкладка «Диапазон цены» у одной компании.")
        hist_value = value_series(full_prices, qty)
        div_events = portfolio_dividend_events({t: a["dividends"] for t, a in data["rub"].items()}, qty)
        tables.update({k.replace("Диапазон цены", "Портфель - диапазон"): v for k, v in
                       render_price_range(hist_value, "Портфель", "₽", GREEN, "portfolio", dividends=div_events).items()})

    if problems:
        with st.expander("⚠️ Что не удалось загрузить"):
            for p in dict.fromkeys(problems):
                st.write(f"- {p}")
    st.divider()
    render_export(tables, "portfolio")
    st.caption("Справочная аналитика, не инвестиционная рекомендация.")
