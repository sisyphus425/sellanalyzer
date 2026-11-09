"""Исследовательская вкладка: простые модели цены (дрейф, AR, ARIMA, гребневая регрессия) против наивного прогноза."""

import pandas as pd
import plotly.express as px
import streamlit as st

from salesanalyzer.config import LAB_HORIZONS_DAYS, LAB_MIN_TRAIN
from salesanalyzer.core.pricelab import (MIN_PRICES, NAIVE, VERDICT_BASE, latest_forecasts, overall_conclusion, summarize, walk_forward)
from salesanalyzer.core.pricesim import clean_prices
from salesanalyzer.ui.compat import show_chart, show_table

VERDICT_ICONS = {"лучше наивного": "🟢", "хуже наивного": "🔴", "не отличается от наивного": "⚪", "мало проверок": "🟠", VERDICT_BASE: "▫️"}
DISCLAIMER = ("Исследовательская вкладка, не прогноз и не рекомендация. Цены акций близки к случайному блужданию, и простые модели редко "
              "превосходят предположение «цена не изменится». Здесь показано, как они проявили себя на прошлом, когда видели только данные до "
              "каждой даты проверки. Если какая-то модель выиграла, это может быть случайностью: при сравнении нескольких моделей "
              "одна из них выигрывает и без реальной силы, а будущее может отличаться от прошлого.")


def render_pricelab(prices: pd.DataFrame, ticker: str, currency: str, color: str, key: str) -> dict:
    """Лаборатория моделей цены. Возвращает таблицы для экспорта."""
    st.caption("Проверка честная: на каждой дате модель обучается только на прошлом и прогнозирует доходность вперёд; ошибки сравниваются "
               "с наивным прогнозом «цена не изменится». Если модель не точнее наивного, её прогнозу доверять нельзя.")
    history = clean_prices(prices) if prices is not None and not prices.empty else pd.DataFrame()
    if len(history) < MIN_PRICES:
        st.info(f"Истории цен мало (нужно хотя бы {MIN_PRICES} торговых дней, примерно {MIN_PRICES // 21} месяцев): проверка не строится.")
        return {}

    labels = {1: "1 день", 21: "1 месяц (21 торговый день)"}
    horizon = st.selectbox("Горизонт прогноза:", list(LAB_HORIZONS_DAYS), format_func=lambda h: labels.get(h, f"{h} дн."), key=f"lab_h_{key}")
    with st.spinner("Проверяем модели на прошлом..."):
        origins = walk_forward(history, horizon)
    summary = summarize(origins, horizon)
    if summary.empty:
        st.info(f"Для проверки нужно не меньше {LAB_MIN_TRAIN} дней истории до первой даты проверки плюс горизонт.")
        return {}

    n = int(summary["Проверок"].max())
    first, last = origins["Дата"].min(), origins["Дата"].max()
    st.markdown(f"**Итог для горизонта {labels.get(horizon, horizon)}:** {overall_conclusion(summary)}")
    st.caption(f"Проверок на прошлом: {n} (с {pd.to_datetime(first):%d.%m.%Y} по {pd.to_datetime(last):%d.%m.%Y}); окна проверки не перекрываются.")

    view = summary.assign(Вывод=[f"{VERDICT_ICONS.get(v, '')} {v}" for v in summary["Вывод"]]).drop(columns=["Горизонт, дней"])
    show_table(view.set_index("Модель").round(2))
    st.caption("«RMSE к наивному» меньше 1: модель в среднем точнее наивного; «Вывод» «лучше» ставится только при заметной разнице "
               "(|t| ≥ 2). «Знак угадан» сравнивайте с долей роста в выборке: если рынок рос в 55% периодов, то 55% угадываний ничего не значат.")

    chart = summary[summary["Модель"] != NAIVE]
    if not chart.empty:
        fig = px.bar(chart, x="Модель", y="RMSE к наивному", color="Вывод", text_auto=".3f",
                     title=f"{ticker}: ошибка модели относительно наивного прогноза (меньше 1 = лучше), {labels.get(horizon, horizon)}")
        fig.add_hline(y=1, line_dash="dot", line_color="gray")
        fig.update_layout(legend_title="", xaxis_title="")
        show_chart(fig)

    latest = latest_forecasts(history, horizon)
    verdicts = dict(zip(summary["Модель"], summary["Вывод"]))
    rows = [{"Модель": m, "Доходность, %": ret, f"Цена, {currency}": price, "Проверка на прошлом": f"{VERDICT_ICONS.get(verdicts.get(m, ''), '')} {verdicts.get(m, '')}"}
            for m, (ret, price) in latest.items()]
    with st.expander("📌 Что модели «говорят» сегодня (только для справки)"):
        st.warning("Это не прогноз. Если у модели нет отметки «лучше наивного», её число не лучше, чем «цена не изменится».")
        show_table(pd.DataFrame(rows).set_index("Модель").round(2))
    with st.expander("🔍 Как устроены модели"):
        st.markdown("- **Наивный:** цена через h дней равна текущей.\n- **Средний дрейф:** продолжает среднюю историческую доходность.\n"
                    "- **AR(5):** предсказывает доходность по пяти последним дневным доходностям.\n"
                    "- **ARIMA(1,1,1):** авторегрессия и скользящее среднее ошибок по доходностям (оценка Ханнана–Риссанена).\n"
                    "- **Гребневая регрессия (ML):** линейная регрессия с регуляризацией по пяти признакам: недавний импульс за 1, 5, 21 день, "
                    "волатильность и отклонение от средней за 50 дней.")
    st.caption(DISCLAIMER)

    return {"Лаборатория - сводка": summary, "Лаборатория - проверки": origins}
