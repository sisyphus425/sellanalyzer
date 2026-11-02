"""Общие блоки интерфейса: вкладки выручки, прибыльности, цены, мультипликаторов, экспорт, пересчет."""

import pandas as pd
import plotly.express as px
import streamlit as st

from salesanalyzer import config
from salesanalyzer.config import GROWTH_DOWN, GROWTH_UP, INDICATOR_NAMES, SIGNAL_COLORS, USD_UNITS
from salesanalyzer.core.adjustments import price_levels
from salesanalyzer.core.exports import to_csv_bytes, to_excel_bytes
from salesanalyzer.core.formatting import fmt_value
from salesanalyzer.core.growth import add_growth, calc_cagr, signal_icon, signal_name
from salesanalyzer.core.market import market_tables
from salesanalyzer.core.prices import add_price_indicators, price_metrics, trend_label
from salesanalyzer.core.profitability import add_margins, latest_value
from salesanalyzer.data_sources.csv_store import load_macro
from salesanalyzer.ui.compat import rerun, show_chart, show_table
from salesanalyzer.config import PRICE_PERIOD_OPTIONS


def render_revenue(df: pd.DataFrame, unit: str, color: str, color_est: str, title: str):
    """df: колонки year, revenue, is_estimate (+ необязательно metric, source, note)."""
    df = add_growth(df)
    last = df.iloc[-1]
    cagr = calc_cagr(df)
    span = int(df["year"].iloc[-1] - df["year"].iloc[0])

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Выручка за {int(last['year'])} г.",
              ("≈ " if last["is_estimate"] else "") + fmt_value(last["revenue"], unit))
    g = last["growth"]
    if pd.isna(g):
        m2.metric("Рост к прошлому году", "н/д")
    else:
        m2.metric(f"Рост к прошлому году {signal_icon(g)}", f"{g:.2f}%", delta=f"{g:.2f}%")
    if cagr is None:
        m3.metric("CAGR", "н/д", help="Нужны данные минимум за два разных года")
    else:
        m3.metric(f"CAGR за {span} г. {signal_icon(cagr)}", f"{cagr:.2f}%", delta=f"{cagr:.2f}%",
                  help="Среднегодовой темп роста между первым и последним годом: "
                       "(последнее / первое) ^ (1 / число лет) − 1")

    plot = df.assign(Год=df["year"].astype(str),
                     Тип=df["is_estimate"].map({True: "Расчетная оценка", False: "Опубликовано"}),
                     Значение=df["revenue"])
    fig = px.bar(plot, x="Год", y="Значение", color="Тип", text_auto=".4s", title=title,
                 color_discrete_map={"Опубликовано": color, "Расчетная оценка": color_est})
    fig.update_layout(xaxis_type="category", yaxis_title=unit)
    show_chart(fig)

    # ---- Динамика роста ----
    st.markdown("#### Динамика роста")
    gdf = df.dropna(subset=["growth"]).copy()
    if gdf.empty:
        st.info("Для расчета годового роста нужны данные минимум за два соседних года.")
    else:
        gdf["Год"] = gdf["year"].astype(str)
        gdf["Сигнал"] = gdf["growth"].map(signal_name)
        c1, c2 = st.columns([3, 2])
        with c1:
            gfig = px.bar(gdf, x="Год", y="growth", color="Сигнал", text_auto=".1f",
                          color_discrete_map=SIGNAL_COLORS, title="Рост выручки к предыдущему году, %")
            gfig.update_layout(xaxis_type="category", yaxis_title="%")
            show_chart(gfig)
        with c2:
            table = pd.DataFrame({
                "Год": gdf["Год"],
                f"Выручка ({unit})": gdf["revenue"].map(lambda v: f"{v:,.1f}"),
                "Рост г/г": gdf["growth"].map(lambda v: f"{v:+.1f}%"),
                "": gdf["growth"].map(signal_icon),
            }).set_index("Год")
            show_table(table)

        grew = int((gdf["growth"] > 0).sum())
        st.caption(f"Лет с ростом: {grew} из {len(gdf)}. "
                   f"Светофор: 🟢 рост от +{GROWTH_UP:.0f}%, 🟡 от {GROWTH_DOWN:.0f}% до +{GROWTH_UP:.0f}%, "
                   f"🔴 падение больше {abs(GROWTH_DOWN):.0f}%. Рост номинальный: без поправки на инфляцию.")

    notes = []
    if df["is_estimate"].any():
        notes.append("в расчете участвуют расчетные оценки")
    if (df["year"].diff().dropna() > 1).any():
        notes.append("в данных есть пропущенные годы: рост для них не считается, "
                     "а CAGR берется между первым и последним годом")
    if notes:
        st.caption("⚠️ Учтите: " + "; ".join(notes) + ".")

    with st.expander("📄 Исходная таблица данных, источники и комментарии"):
        cols = {"year": "Год", "revenue": f"Выручка ({unit})", "metric": "Показатель",
                "source": "Источник", "note": "Комментарий"}
        table = df[[c for c in cols if c in df.columns]].rename(columns=cols)
        show_table(table.set_index("Год"))


def render_profitability(w: pd.DataFrame, unit: str, color: str, source_rows=None, mixed: bool = False):
    """Вкладка 'Прибыльность': чистая прибыль, EBITDA, маржа, ROE и рост чистой прибыли."""
    if mixed:
        st.warning("В данных смешаны разные единицы измерения: маржа не рассчитывается. Проверьте колонку unit.")
    w = add_margins(w, enabled=not mixed)
    if not any(w[c].notna().any() for c in ("net_income", "ebitda", "roe")):
        st.info("Нет данных о прибыли, EBITDA или ROE для этой компании.")
        return

    ni, nm, em, roe = (latest_value(w, c) for c in ("net_income", "net_margin", "ebitda_margin", "roe"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Чистая прибыль за {ni[0]} г." if ni else "Чистая прибыль",
              fmt_value(ni[1], unit) if ni else "н/д")
    m2.metric(f"Чистая маржа за {nm[0]} г." if nm else "Чистая маржа",
              f"{nm[1]:.1f}%" if nm else "н/д", help="Чистая прибыль / выручка")
    m3.metric(f"EBITDA-маржа за {em[0]} г." if em else "EBITDA-маржа",
              f"{em[1]:.1f}%" if em else "н/д", help="EBITDA / выручка. Для банков не применяется")
    m4.metric(f"ROE за {roe[0]} г." if roe else "ROE",
              f"{roe[1]:.1f}%" if roe else "н/д", help="Рентабельность капитала. Приводится для банков")

    # Чистая прибыль и EBITDA в единицах
    value_cols = [c for c in ("net_income", "ebitda") if w[c].notna().any()]
    if value_cols:
        vals = w.melt(id_vars="year", value_vars=value_cols, var_name="Показатель", value_name="Значение")
        vals = vals.dropna(subset=["Значение"])
        vals["Показатель"] = vals["Показатель"].map(INDICATOR_NAMES)
        vals["Год"] = vals["year"].astype(str)
        fig = px.bar(vals, x="Год", y="Значение", color="Показатель", barmode="group", text_auto=".4s",
                     title=f"Чистая прибыль и EBITDA, {unit}",
                     color_discrete_map={"Чистая прибыль": color, "EBITDA": "#8A8F98"})
        fig.update_layout(xaxis_type="category", yaxis_title=unit, legend_title="")
        show_chart(fig)

    # Маржа и ROE в процентах
    margin_cols = [c for c in ("net_margin", "ebitda_margin") if w[c].notna().any()]
    has_roe = w["roe"].notna().any()
    left, right = (st.columns(2) if margin_cols and has_roe else (st.container(), st.container()))
    if margin_cols:
        with left:
            mm = w.melt(id_vars="year", value_vars=margin_cols, var_name="Показатель", value_name="Значение")
            mm = mm.dropna(subset=["Значение"])
            mm["Показатель"] = mm["Показатель"].map({"net_margin": "Чистая маржа", "ebitda_margin": "EBITDA-маржа"})
            fig = px.line(mm, x="year", y="Значение", color="Показатель", markers=True, title="Маржа, % от выручки")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(xaxis=dict(dtick=1), xaxis_title="Год", yaxis_title="%", legend_title="")
            show_chart(fig)
    if has_roe:
        with right:
            rd = w.dropna(subset=["roe"]).assign(Год=lambda d: d["year"].astype(str))
            fig = px.bar(rd, x="Год", y="roe", text_auto=".1f", title="ROE (рентабельность капитала), %",
                         color_discrete_sequence=[color])
            fig.update_layout(xaxis_type="category", yaxis_title="%")
            show_chart(fig)

    # Рост чистой прибыли
    ni_df = w.dropna(subset=["net_income"])[["year", "net_income"]]
    if len(ni_df) > 1:
        g = add_growth(ni_df, "net_income")
        cagr = calc_cagr(ni_df, "net_income")
        st.markdown("#### Динамика чистой прибыли")
        table = pd.DataFrame({
            "Год": g["year"].astype(str),
            f"Чистая прибыль ({unit})": g["net_income"].map(lambda v: f"{v:,.1f}"),
            "Рост г/г": g["growth"].map(lambda v: "н/д" if pd.isna(v) else f"{v:+.1f}%"),
            "": g["growth"].map(signal_icon),
        }).set_index("Год")
        show_table(table)
        cagr_text = ("CAGR чистой прибыли за {a}–{b}: {c:.1f}% {i}. ".format(
            a=int(ni_df["year"].iloc[0]), b=int(ni_df["year"].iloc[-1]), c=cagr, i=signal_icon(cagr))
            if cagr is not None else "CAGR чистой прибыли не рассчитан (убыток в первом или последнем году ряда). ")
        st.caption(cagr_text + "Рост считается, только если прибыль положительна в обоих годах; "
                   "при смене знака или убытке показывается «н/д».")

    notes = ["маржа = показатель / выручка (для банков «выручка» — операционный доход)"]
    if w["has_estimate"].any():
        notes.append("в данных есть расчетные оценки (см. таблицу ниже)")
    st.caption("ℹ️ " + "; ".join(notes) + ".")

    if source_rows is not None and not source_rows.empty:
        with st.expander("📄 Исходные данные, источники и комментарии"):
            t = source_rows.sort_values(["indicator", "year"]).copy()
            t["Показатель"] = t["indicator"].map(INDICATOR_NAMES).fillna(t["indicator"])
            t["Оценка"] = t["is_estimate"].map({True: "да", False: ""})
            t = t.rename(columns={"year": "Год", "value": "Значение", "unit": "Единицы",
                                  "metric": "Определение", "source": "Источник", "note": "Комментарий"})
            show_table(t[["Показатель", "Год", "Значение", "Единицы", "Определение", "Оценка", "Источник", "Комментарий"]]
                         .set_index("Показатель"))


def show_data_error(error, key: str):
    """Сообщение об ошибке источника данных и кнопка «Повторить»."""
    st.error(str(error))
    if st.button("🔄 Повторить", key=f"retry_{key}"):
        rerun()


def render_export(tables: dict, file_stem: str):
    """Блок кнопок скачивания: CSV (любая таблица) и Excel (все таблицы на отдельных листах)."""
    tables = {k: v for k, v in tables.items() if v is not None and not v.empty}
    if not tables:
        return
    st.markdown("#### 📥 Экспорт данных")
    left, right = st.columns(2)
    with left:
        sheet = st.selectbox("Таблица для CSV:", list(tables), key=f"csv_sheet_{file_stem}")
        sep_label = st.radio("Разделитель CSV:", ["запятая (,)", "точка с запятой (;) — для русского Excel"],
                             key=f"csv_sep_{file_stem}")
        sep = ";" if sep_label.startswith("точка") else ","
        st.download_button("⬇️ Скачать CSV", data=to_csv_bytes(tables[sheet], sep),
                           file_name=f"{file_stem}_{sheet}.csv".replace(" ", "_"), mime="text/csv",
                           key=f"csv_btn_{file_stem}")
    with right:
        xlsx = to_excel_bytes(tables)
        st.write("**Excel:** все таблицы в одном файле, каждая на своем листе.")
        if xlsx is None:
            st.info("Для выгрузки в Excel установите библиотеку: `pip install openpyxl`")
        else:
            st.download_button("⬇️ Скачать Excel (все таблицы)", data=xlsx, file_name=f"{file_stem}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"xlsx_btn_{file_stem}")
        st.caption("Данные выгружаются «как на экране»: с пометками об оценках и источниками (для РФ).")


def render_market_stats(stats, payout: pd.DataFrame, notes: list, color: str) -> dict:
    """Вкладка 'Мультипликаторы и дивиденды'. Возвращает таблицы для экспорта."""
    for n in notes:
        st.caption("⚠️ " + n)
    if stats is None:
        st.info("Недостаточно данных для расчета.")
        return {}
    cur = stats["currency"]

    def f(v, fmt):
        return "н/д" if v is None else fmt.format(v)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Цена акции" + ("" if stats["price_date"] is None else f" на {stats['price_date']:%d.%m.%Y}"),
              f(stats["price"], "{:,.2f} " + cur))
    c2.metric("Капитализация", f(stats["mcap"], "{:,.1f} млрд " + cur),
              help="Цена акции × число акций из файла (для США — данные Yahoo Finance).")
    c3.metric("P/S", f(stats["ps"], "{:.2f}"), help=f"Капитализация / выручка за {stats['y_rev']} г. ({stats['rev_metric']})")
    pe_text = f(stats["pe"], "{:.1f}") if stats["pe"] is not None else ("н/д (убыток)" if stats["y_ni"] else "н/д")
    c4.metric("P/E", pe_text, help=f"Капитализация / чистая прибыль за {stats['y_ni']} г. ({stats['ni_metric']})")

    d1, d2, d3, d4 = st.columns(4)
    dl = stats["dps_last"]
    d1.metric(f"Дивиденд на акцию за {dl[0]} г." if dl else "Дивиденд на акцию",
              "н/д" if dl is None else f"{dl[1]:,.2f} {cur}" + (" ≈" if stats["dps_est"] else ""))
    d2.metric("Дивидендная доходность", f(stats["div_yield"], "{:.2f}%"),
              help=f"Дивиденд / текущая цена ({stats['div_basis']}).")
    pr = payout["Payout, %"].dropna()
    d3.metric("Payout за последний год с данными", f(None if pr.empty else float(pr.iloc[-1]), "{:.0f}%"),
              help="Доля чистой прибыли, направленная на дивиденды (по текущему числу акций).")
    d4.metric("Число акций, млн", f(stats["shares_mn"], "{:,.1f}"))

    if not payout.empty:
        left, right = st.columns(2)
        with left:
            fig = px.bar(payout.assign(Год=payout["Год"].astype(str)), x="Год", y="Дивиденд на акцию", text_auto=".2f",
                         title=f"Дивиденд на акцию по годам, {cur}", color_discrete_sequence=[color])
            fig.update_layout(xaxis_type="category", yaxis_title=cur)
            show_chart(fig)
        with right:
            pp = payout.dropna(subset=["Payout, %"])
            if pp.empty:
                st.info("Payout не рассчитан: нет прибыли или числа акций.")
            else:
                fig = px.bar(pp.assign(Год=pp["Год"].astype(str)), x="Год", y="Payout, %", text_auto=".0f",
                             title="Payout: доля чистой прибыли на дивиденды, %", color_discrete_sequence=[color])
                fig.update_layout(xaxis_type="category", yaxis_title="%")
                show_chart(fig)
        show_table(payout.set_index("Год"))
    else:
        st.info("Данных о дивидендах нет.")

    st.caption("Мультипликаторы текущие: последняя цена, число акций из файла и показатели последнего отчетного года. "
               "«Дивиденды всего» и payout считаются по текущему числу акций, поэтому для прошлых лет это приближение. "
               "При необычной прибыли (разовые статьи, убыток) P/E может быть неинформативным — смотрите определение прибыли в подсказке. "
               "Справочная аналитика, не рекомендация.")
    return market_tables(stats, payout)


def render_price(prices: pd.DataFrame, ticker: str, currency: str, color: str):
    """Вкладка 'Цена акции': метрики риска и доходности, график со скользящими средними, график просадки."""
    if prices.empty:
        st.info("История цен недоступна для этого тикера.")
        return

    years = st.selectbox("Период:", list(PRICE_PERIOD_OPTIONS), index=len(PRICE_PERIOD_OPTIONS) - 1, format_func=lambda y: f"{y} г.", key=f"price_years_{ticker}")
    full = add_price_indicators(prices)  # средние считаем по всей истории, чтобы линии были на старте периода
    p = full[full["begin"] >= full["begin"].max() - pd.DateOffset(years=years)].reset_index(drop=True)
    m = price_metrics(p)
    if m is None:
        st.info("Недостаточно данных за выбранный период.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Последняя цена закрытия, {currency}", f"{m['last']:,.2f}")
    c2.metric("Доходность за период", f"{m['total']:.1f}%", delta=f"{m['total']:.1f}%")
    c3.metric("Волатильность (год)", "н/д" if m["vol"] is None else f"{m['vol']:.1f}%",
              help="Стандартное отклонение дневной доходности × √252. Чем выше, тем сильнее колебания цены.")
    c4.metric("Макс. просадка", f"{m['max_dd']:.1f}%",
              help=f"Наибольшее падение от предыдущего максимума: пик {m['dd_peak']:%d.%m.%Y}, "
                   f"минимум {m['dd_trough']:%d.%m.%Y}.")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Годовая доходность", "н/д" if m["annual"] is None else f"{m['annual']:.1f}%",
              help="Доходность за период, пересчитанная на год (нужен период от 3 месяцев).")
    d2.metric("Текущая просадка", f"{m['cur_dd']:.1f}%", help="Отставание последней цены от максимума за период.")
    d3.metric("MA50 / MA200", "н/д" if m["ma50"] is None or m["ma200"] is None
              else f"{m['ma50']:,.1f} / {m['ma200']:,.1f}")
    d4.metric("Тренд", trend_label(m), help="Расположение последней цены относительно 200-дневной средней.")

    show_ma = st.checkbox("Показать скользящие средние (50 и 200 дней)", value=True, key=f"price_ma_{ticker}")
    cols = ["close"] + (["ma50", "ma200"] if show_ma else [])
    long = p.melt(id_vars="begin", value_vars=cols, var_name="Серия", value_name="Цена").dropna(subset=["Цена"])
    long["Серия"] = long["Серия"].map({"close": "Цена закрытия", "ma50": "MA50", "ma200": "MA200"})
    fig = px.line(long, x="begin", y="Цена", color="Серия", title=f"Цена закрытия {ticker} ({currency})",
                  color_discrete_map={"Цена закрытия": color, "MA50": "#F2B01E", "MA200": "#8A8F98"})
    fig.update_layout(xaxis_title="Дата", yaxis_title=f"Цена, {currency}", legend_title="")
    show_chart(fig)

    p["drawdown"] = (p["close"] / p["close"].cummax() - 1) * 100
    dfig = px.area(p, x="begin", y="drawdown", title="Просадка от максимума, %", color_discrete_sequence=["#D64545"])
    dfig.update_layout(xaxis_title="Дата", yaxis_title="%")
    show_chart(dfig)

    st.caption("MA — среднее цен закрытия за 50 и 200 торговых дней. Просадка — падение от предыдущего максимума. "
               "Прошлые показатели не гарантируют будущих результатов; это справочная аналитика, а не рекомендация.")
    out = p.assign(drawdown=p["drawdown"].round(2)).rename(columns={
        "begin": "Дата", "close": f"Цена закрытия, {currency}", "ma50": "MA50", "ma200": "MA200", "drawdown": "Просадка, %"})
    st.download_button("⬇️ Скачать цены и метрики (CSV)", data=to_csv_bytes(out), mime="text/csv",
                       key=f"dl_price_{ticker}", file_name=f"{ticker}_prices.csv")


def adjustment_controls(units: list, key: str):
    """Переключатели пересчета. Возвращает (to_rub, real, base_year, macro)."""
    macro = load_macro()
    levels = price_levels(macro["cpi"])
    has_usd = any(u in USD_UNITS for u in units)
    base_year = max(levels) if levels else None
    c1, c2, c3 = st.columns([3, 3, 2])
    to_rub = real = False
    with c1:
        if has_usd:
            to_rub = st.checkbox("Пересчитать из $ в ₽ (средний курс ЦБ за год)", key=f"fx_{key}")
    with c2:
        real = st.checkbox("Поправка на инфляцию (реальные значения)", key=f"real_{key}")
    with c3:
        if real and levels:
            base_year = st.selectbox("В ценах года:", sorted(levels), index=len(levels) - 1, key=f"base_{key}")
    if (to_rub or real) and not (macro["usd_rub"] or macro["cpi"]):
        st.warning(f"Файл {config.MACRO_PATH.name} не найден или пуст: пересчет недоступен.")
        return False, False, base_year, macro
    return to_rub, real, base_year, macro


def render_adjustment_info(macro: dict, to_rub: bool, real: bool, base_year, notes: list):
    if not (to_rub or real):
        return
    for n in notes:
        st.caption("⚠️ " + n)
    parts = []
    if to_rub:
        parts.append("значения в $ пересчитаны в ₽ по среднегодовому курсу ЦБ за соответствующий год")
    if real:
        parts.append(f"значения в ₽ приведены к ценам {base_year} г. по ИПЦ Росстата (уровень цен в середине года)")
    st.caption("ℹ️ " + "; ".join(parts) + ". Маржа и ROE от пересчета не меняются.")
    with st.expander("🔢 Коэффициенты пересчета"):
        levels = price_levels(macro["cpi"])
        tbl = []
        for y in sorted(set(macro["usd_rub"]) | set(macro["cpi"])):
            tbl.append({"Год": y, "Курс ЦБ, ₽ за $": macro["usd_rub"].get(y),
                        "Инфляция дек/дек, %": macro["cpi"].get(y),
                        f"Коэф. к ценам {base_year} г.": round(levels[base_year] / levels[y], 4)
                        if real and y in levels and base_year in levels else None})
        show_table(pd.DataFrame(tbl).set_index("Год"))
        st.caption("Курс — простое среднее официальных курсов ЦБ за год. Уровень цен в середине года — среднее геометрическое "
                   "уровней на конец предыдущего и текущего года (приближение для годовых потоков). "
                   "Данные лежат в macro_ru.csv, их можно обновлять без изменения кода.")
