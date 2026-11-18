"""Блоки вкладки «Цена акции»: полная доходность с дивидендами и сравнение с индексом рынка."""

import pandas as pd
import plotly.express as px
import streamlit as st

from salesanalyzer.config import BENCHMARKS
from salesanalyzer.core.benchmark import add_total_return, beta_label, compare_with_benchmark, total_return_metrics
from salesanalyzer.data_sources.errors import DataSourceError
from salesanalyzer.ui.compat import show_chart, show_table

INDEX_COLOR = "#8A8F98"


def _pct(v, digits=1, sign=False):
    if v is None or pd.isna(v):
        return "н/д"
    return f"{v:+.{digits}f}%" if sign else f"{v:.{digits}f}%"


def render_total_return(p: pd.DataFrame, dividends, ticker: str, currency: str, color: str, no_data_hint: str = ""):
    """Полная доходность за выбранный период. p: цены периода (begin, close). Возвращает таблицу add_total_return
    (с колонкой tr) или None, если данных о дивидендах нет."""
    st.markdown("#### 💰 Полная доходность (цена + дивиденды)")
    if dividends is None or len(dividends) == 0:
        st.info("Нет данных о датах выплаты дивидендов: полная доходность не считается, показана только доходность по цене. "
                + no_data_hint)
        return None
    tr = add_total_return(p, dividends)
    m = total_return_metrics(tr)
    if m is None:
        st.info("Недостаточно данных за выбранный период.")
        return None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Полная доходность", _pct(m["tr_total"]), help="Цена + дивиденды, вложенные в ту же бумагу в день отсечки. Налог не учитывается.")
    c2.metric("Только цена", _pct(m["price_total"]))
    c3.metric("Вклад дивидендов", _pct(m["div_effect"], sign=True).replace("%", " п.п."),
              help="Полная доходность минус доходность по цене.")
    c4.metric("Годовая полная доходность", _pct(m["tr_annual"]), help="Нужен период от 3 месяцев.")
    if m["n_dividends"] == 0:
        st.caption("За выбранный период отсечек не было: полная доходность равна доходности по цене.")
    else:
        st.caption(f"Учтено выплат: {m['n_dividends']}, всего {m['dividends_sum']:,.2f} {currency} на акцию.")
        base = tr["close"].iloc[0]
        long = pd.DataFrame({"Дата": list(tr["begin"]) * 2,
                             "Значение": list(tr["close"] / base * 100) + list(tr["tr"] / base * 100),
                             "Серия": ["Только цена"] * len(tr) + ["Цена + дивиденды"] * len(tr)})
        fig = px.line(long, x="Дата", y="Значение", color="Серия", title=f"{ticker}: доходность с дивидендами и без (старт = 100)",
                      color_discrete_map={"Только цена": INDEX_COLOR, "Цена + дивиденды": color})
        fig.add_hline(y=100, line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title="Дата", yaxis_title="Индекс", legend_title="")
        show_chart(fig)
        with st.expander("Выплаты, учтённые в расчёте"):
            ev = tr[tr["div"] > 0][["begin", "div", "close"]].rename(
                columns={"begin": "Дата отсечки", "div": f"Дивиденд, {currency}", "close": f"Цена закрытия, {currency}"})
            ev["Доходность выплаты, %"] = (ev[f"Дивиденд, {currency}"] / (ev[f"Цена закрытия, {currency}"] + ev[f"Дивиденд, {currency}"]) * 100).round(2)
            ev["Дата отсечки"] = ev["Дата отсечки"].dt.strftime("%d.%m.%Y")
            show_table(ev, hide_index=True)
    return tr


def render_vs_market(p: pd.DataFrame, tr, market: str, loader, ticker: str, color: str, key: str):
    """Сравнение с индексом рынка. p: цены периода; tr: результат render_total_return или None;
    loader(тикер индекса) -> цены индекса (begin, close). Возвращает таблицу для выгрузки или None."""
    st.markdown("#### 📈 Сравнение с рынком")
    spec = BENCHMARKS[market]
    options = ["По цене"] + (["С дивидендами (полная доходность)"] if tr is not None else [])
    mode = st.radio("Как сравнивать:", options, horizontal=True, key=f"bm_mode_{key}",
                    help="«По цене» — бумага против обычного индекса. «С дивидендами» — полная доходность бумаги "
                         "против индекса полной доходности, где дивиденды тоже реинвестируются.")
    total = mode != "По цене"
    bench_id, bench_name = spec["total"] if total else spec["price"]
    try:
        bench = loader(bench_id)
    except DataSourceError as e:
        st.warning(f"Не удалось загрузить {bench_name}: {e}")
        return None
    if bench is None or bench.empty:
        st.info(f"Нет данных по индексу {bench_name}.")
        return None
    stock = tr[["begin", "tr"]].rename(columns={"tr": "close"}) if total else p[["begin", "close"]]
    bench = bench[bench["begin"] >= p["begin"].min()]
    m, frame = compare_with_benchmark(stock, bench)
    if m is None:
        st.info(f"Мало общих торговых дней с индексом {bench_name}: сравнение не строится.")
        return None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{ticker}", _pct(m["stock_total"]))
    c2.metric(bench_name, _pct(m["bench_total"]))
    c3.metric("Разница с рынком", _pct(m["excess"], sign=True).replace("%", " п.п."),
              delta=_pct(m["excess"], sign=True).replace("%", " п.п."), help="Доходность бумаги минус доходность индекса за общий период.")
    c4.metric("Бета", "н/д" if m["beta"] is None else f"{m['beta']:.2f}",
              help="Насколько в среднем бумага меняется, когда индекс меняется на 1%. "
                   "1 — как рынок, больше 1 — сильнее рынка, меньше 1 — слабее. Считается по дневным доходностям.")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Годовая доходность бумаги", _pct(m["stock_annual"]))
    d2.metric("Годовая доходность индекса", _pct(m["bench_annual"]))
    d3.metric("Корреляция с индексом", "н/д" if m["corr"] is None else f"{m['corr']:.2f}",
              help="От −1 до 1: насколько синхронно бумага и индекс меняются день ко дню.")
    d4.metric("Отклонение от индекса (год)", _pct(m["tracking_error"]),
              help="Стандартное отклонение разницы дневных доходностей × √252 (tracking error).")
    st.caption(f"Бета {beta_label(m['beta'])}. Период {m['start']:%d.%m.%Y}–{m['end']:%d.%m.%Y}, общих торговых дней: {m['days']}.")

    long = frame.melt(id_vars="begin", value_vars=["stock", "bench"], var_name="Серия", value_name="Индекс")
    long["Серия"] = long["Серия"].map({"stock": ticker + (" (с дивидендами)" if total else ""), "bench": bench_name})
    fig = px.line(long, x="begin", y="Индекс", color="Серия", title=f"{ticker} и {bench_name} (старт = 100)",
                  color_discrete_sequence=[color, INDEX_COLOR])
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Дата", yaxis_title="Индекс", legend_title="")
    show_chart(fig)
    rfig = px.line(frame, x="begin", y="relative", title=f"Относительная сила: {ticker} к {bench_name} (старт = 100)",
                   color_discrete_sequence=[color])
    rfig.add_hline(y=100, line_dash="dot", line_color="gray")
    rfig.update_layout(xaxis_title="Дата", yaxis_title="Бумага / индекс × 100")
    show_chart(rfig)
    st.caption("Линия относительной силы растёт, когда бумага обгоняет индекс, и падает, когда отстаёт. "
               "Прошлые показатели не гарантируют будущих результатов.")
    return frame.rename(columns={"begin": "Дата", "stock": f"{ticker}, индекс", "bench": f"{bench_name}, индекс",
                                 "relative": "Относительная сила"})
