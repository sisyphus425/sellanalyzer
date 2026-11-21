"""Блоки вкладки «Мультипликаторы и дивиденды»: мультипликаторы по годам и сравнение доходности акции с ОФЗ."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from salesanalyzer.config import OFZ_TERMS
from salesanalyzer.core.valuation import compare_to_history, curve_yield, history_vs_ofz, spread_rows, trailing_dividends
from salesanalyzer.data_sources.errors import DataSourceError
from salesanalyzer.ui.compat import show_chart, show_table

GRAY = "#8A8F98"
OFZ_COLOR = "#D64545"


def _fmt(v, fmt):
    return "н/д" if v is None or pd.isna(v) else fmt.format(v)


def render_multiples_history(history: pd.DataFrame, stats, currency: str, color: str, notes=()) -> dict:
    """Мультипликаторы на конец прошлых лет и сравнение текущих с медианой. Возвращает таблицы для экспорта."""
    st.markdown("#### 📅 Мультипликаторы по годам")
    for n in notes:
        st.caption("ℹ️ " + n)
    if history.empty:
        st.info("Недостаточно данных: нужны цена на конец года и выручка или прибыль за этот год.")
        return {}
    current = {"P/S": stats.get("ps") if stats else None, "P/E": stats.get("pe") if stats else None}
    cmp = compare_to_history(history, current)
    if cmp:
        cols = st.columns(len(cmp))
        for col, c in zip(cols, cmp):
            col.metric(f"{c['Показатель']} сейчас", f"{c['Сейчас']:.2f}",
                       delta=None if c["Отклонение от медианы, %"] is None else f"{c['Отклонение от медианы, %']:+.0f}% к медиане",
                       delta_color="inverse",
                       help=f"Медиана на конец {c['Лет']} прошлых лет: {c['Медиана прошлых лет']:.2f}. Вывод: {c['Вывод']}. "
                            "Ниже медианы — акция оценена дешевле, чем обычно для нее самой (это не значит, что она недооценена).")
        st.caption("Сравнение текущих мультипликаторов с медианой прошлых лет: " +
                   "; ".join(f"{c['Показатель']} {c['Вывод']}" for c in cmp) + ".")

    chart = history[["Год", "P/E", "P/S"]].copy()
    chart["Год"] = chart["Год"].astype(str)
    if stats and (stats.get("pe") or stats.get("ps")):
        chart = pd.concat([chart, pd.DataFrame([{"Год": "Сейчас", "P/E": stats.get("pe"), "P/S": stats.get("ps")}])], ignore_index=True)
    left, right = st.columns(2)
    for box, col in ((left, "P/E"), (right, "P/S")):
        data = chart.dropna(subset=[col])
        with box:
            if data.empty:
                st.info(f"{col} по годам не рассчитан (нет прибыли или выручки).")
                continue
            fig = px.bar(data, x="Год", y=col, text_auto=".1f", title=f"{col} на конец года",
                         color=data["Год"].eq("Сейчас").map({True: "Сейчас", False: "Конец года"}),
                         color_discrete_map={"Конец года": color, "Сейчас": GRAY})
            med = history[col].dropna()
            if len(med) >= 2:
                fig.add_hline(y=float(med.median()), line_dash="dot", line_color=GRAY, annotation_text="медиана")
            fig.update_layout(xaxis_type="category", yaxis_title=col, legend_title="", showlegend=False)
            show_chart(fig)

    view = history.copy()
    view["Дата цены"] = pd.to_datetime(view["Дата цены"]).dt.strftime("%d.%m.%Y")
    show_table(view.set_index("Год").round(2))
    st.caption(f"Цена — закрытие последнего торгового дня года, {currency}; капитализация = цена × текущее число акций. "
               "Цены биржа пересчитывает на дробления, а число акций в файле указано после дробления, поэтому дробления "
               "(T 10:1, GMKN 1:100) не искажают прошлые значения. Выкупы и допэмиссии не учтены: для прошлых лет это приближение. "
               "P/E на конец года считается по прибыли этого года, которую компания публикует позже.")
    out = {"Мультипликаторы по годам": history}
    if cmp:
        out["Мультипликаторы к истории"] = pd.DataFrame(cmp)
    return out


def render_ofz_comparison(stats, history: pd.DataFrame, events, curve_loader, key: str, color: str) -> dict:
    """Дивидендная доходность и доходность прибыли (E/P) против доходности ОФЗ (G-кривая Мосбиржи), сейчас и по годам.
    curve_loader(дата или None) -> G-кривая. events: выплаты с датами отсечки (для доходности за 12 месяцев)."""
    st.markdown("#### 🏦 Доходность акции против ОФЗ")
    term = st.selectbox("Срок ОФЗ для сравнения:", list(OFZ_TERMS), format_func=lambda t: f"{t} г.", key=f"ofz_term_{key}",
                        help="Доходность берется с кривой бескупонной доходности ОФЗ Мосбиржи (G-кривая) на этот срок.")
    try:
        curve = curve_loader(None)
    except DataSourceError as e:
        st.warning(f"Не удалось загрузить доходность ОФЗ: {e}")
        return {}
    ofz = curve_yield(curve, term)
    if ofz is None:
        st.info("Мосбиржа не вернула кривую доходности ОФЗ.")
        return {}
    curve_date = curve["tradedate"].iloc[0]

    price, pe = (stats or {}).get("price"), (stats or {}).get("pe")
    items = []
    if stats and stats.get("div_yield") is not None:
        dl = stats.get("dps_last")
        items.append(("Дивдоходность (последний финансовый год)", stats["div_yield"],
                      f"дивиденд за {dl[0]} г. / текущая цена" if dl else "дивиденд / текущая цена"))
    if price and events is not None and len(events):
        ttm = trailing_dividends(events, pd.Timestamp.today())
        items.append(("Дивдоходность (выплаты за 12 месяцев)", ttm / price * 100, "сумма выплат с отсечкой за последние 365 дней / текущая цена"))
    if pe:
        items.append(("Доходность прибыли (E/P)", 100 / pe, "1 / P/E: какая доля цены приходится на годовую прибыль"))
    table = spread_rows(items, ofz)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"ОФЗ {term} г. на {curve_date:%d.%m.%Y}", f"{ofz:.2f}%")
    for col, (_, r) in zip((c2, c3), table.head(2).iterrows()):
        col.metric(r["Показатель"], f"{r['Доходность, %']:.2f}%", delta=f"{r['Спред к ОФЗ, п.п.']:+.2f} п.п. к ОФЗ")
    if not table.empty:
        show_table(table.set_index("Показатель").round(2))
    else:
        st.info("Нет цены или дивидендов: сравнить с ОФЗ нечего.")

    hist_table = pd.DataFrame()
    if history is not None and not history.empty:
        ofz_by_year = {}
        try:
            for _, r in history.iterrows():
                c = curve_loader(pd.Timestamp(r["Дата цены"]).date().isoformat())
                y = curve_yield(c, term)
                if y is not None:
                    ofz_by_year[int(r["Год"])] = y
        except DataSourceError as e:
            st.caption(f"⚠️ История доходности ОФЗ не загружена: {e}")
        hist_table = history_vs_ofz(history, ofz_by_year)
        plot = hist_table.dropna(subset=["ОФЗ, %"])
        if not plot.empty:
            fig = go.Figure()
            years = plot["Год"].astype(str)
            fig.add_bar(x=years, y=plot["Дивдоходность, %"], name="Дивдоходность", marker_color=color)
            fig.add_bar(x=years, y=plot["Доходность прибыли (E/P), %"], name="Доходность прибыли (E/P)", marker_color=GRAY)
            fig.add_scatter(x=years, y=plot["ОФЗ, %"], name=f"ОФЗ {term} г.", mode="lines+markers", line=dict(color=OFZ_COLOR, width=3))
            fig.update_layout(title=f"Доходность акции и ОФЗ {term} г. на конец года, %", barmode="group",
                              xaxis_type="category", yaxis_title="%", legend_title="")
            show_chart(fig)
            show_table(plot.set_index("Год").round(2))

    st.caption("ОФЗ дают почти гарантированную доходность в рублях, а дивиденды и прибыль компании меняются и не гарантированы, "
               "поэтому инвесторы обычно ждут от акций доходность выше ОФЗ (премию за риск). Отрицательный спред значит, "
               "что сегодня ОФЗ платят больше, чем дивиденды или прибыль на рубль цены; рост прибыли и цены акции в сравнении не учтён. "
               "Налоги не учитываются. Справочная аналитика, не рекомендация.")
    out = {}
    if not table.empty:
        out["Доходность против ОФЗ"] = table.assign(**{"Срок ОФЗ, лет": term, "Дата кривой ОФЗ": f"{curve_date:%Y-%m-%d}"})
    if not hist_table.empty:
        out["Доходность против ОФЗ по годам"] = hist_table
    return out
