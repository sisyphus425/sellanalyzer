"""Вкладка «Диапазон цены»: сценарии будущей цены (Монте-Карло) и проверка модели на прошлом."""

import pandas as pd
import plotly.express as px
import streamlit as st

from salesanalyzer.config import MC_DAYS_PER_MONTH, MC_HORIZON_MONTHS
from salesanalyzer.core.pricesim import (DRIFT_HIST, DRIFT_ZERO, MODEL_BOOTSTRAP, MODEL_GBM, VOL_EWMA, VOL_FULL, calibrate,
                                         calibration_verdict, clean_prices, price_range)
from salesanalyzer.ui.compat import show_chart, show_table

MODELS = {"Нормальная модель (GBM)": MODEL_GBM, "Бутстрап исторических доходностей": MODEL_BOOTSTRAP}
VOLS = {"Свежая (EWMA)": VOL_EWMA, "За весь период": VOL_FULL}
DRIFTS = {"Без роста (ожидаемая цена = текущей)": DRIFT_ZERO, "Историческая доходность": DRIFT_HIST}
DISCLAIMER = ("Это не прогноз цены, а сценарии: какие цены были бы возможны при историческом разбросе доходностей. Модель не знает о новостях, "
              "дивидендных гэпах, санкциях и других разовых событиях; реальная цена может выйти за любые коридоры. "
              "Справочная аналитика, не инвестиционная рекомендация.")


def _round_numeric(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    """Округляет только числовые колонки (колонки с датами не трогает)."""
    out = df.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(digits)
    return out


def _fan_chart(res, history, currency, color, ticker):
    hist = history.tail(126)
    parts = [pd.DataFrame({"Дата": hist["begin"], "Цена": hist["close"], "Серия": "История"})]
    q = res["quantiles"]
    for col, name in (("p10", "Нижняя граница 80%"), ("p25", "Нижняя граница 50%"), ("p50", "Медиана"),
                      ("p75", "Верхняя граница 50%"), ("p90", "Верхняя граница 80%")):
        parts.append(pd.DataFrame({"Дата": res["dates"], "Цена": q[col].to_numpy(), "Серия": name}))
    fig = px.line(pd.concat(parts, ignore_index=True), x="Дата", y="Цена", color="Серия",
                  title=f"{ticker}: история и коридоры возможных цен, {currency}",
                  color_discrete_map={"История": color, "Медиана": "#1f2937", "Нижняя граница 50%": "#F2B01E", "Верхняя граница 50%": "#F2B01E",
                                      "Нижняя граница 80%": "#D64545", "Верхняя граница 80%": "#D64545"})
    fig.update_layout(yaxis_title=f"Цена, {currency}", legend_title="")
    show_chart(fig)


def render_price_range(prices: pd.DataFrame, ticker: str, currency: str, color: str, key: str) -> dict:
    """Диапазон будущей цены по истории цен. Возвращает таблицы для экспорта."""
    history = clean_prices(prices) if prices is not None and not prices.empty else pd.DataFrame()
    if history.empty:
        st.info("История цен недоступна: диапазон не строится.")
        return {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        months = st.selectbox("Горизонт:", list(MC_HORIZON_MONTHS), format_func=lambda m: f"{m} мес.", key=f"pr_h_{key}")
    with c2:
        model_label = st.selectbox("Модель:", list(MODELS), key=f"pr_m_{key}")
    with c3:
        vol_label = st.selectbox("Волатильность:", list(VOLS), key=f"pr_v_{key}",
                                 help="Для бутстрапа не используется: разброс берётся прямо из истории.")
    with c4:
        drift_label = st.selectbox("Дрейф:", list(DRIFTS), key=f"pr_d_{key}",
                                   help="«Без роста» не продолжает прошлую динамику. «Историческая доходность» переносит среднюю доходность прошлого в будущее.")
    model, vol, drift = MODELS[model_label], VOLS[vol_label], DRIFTS[drift_label]
    days = months * MC_DAYS_PER_MONTH

    res = price_range(history, days, model, vol, drift)
    if res is None:
        st.info("Слишком мало данных (нужно не меньше двух-трёх месяцев торгов) для построения диапазона.")
        return {}
    m, p0, params = res["metrics"], res["p0"], res["params"]

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Текущая цена", f"{p0:,.2f} {currency}")
    a2.metric(f"Медиана через {months} мес.", f"{m['median']:,.2f} {currency}", delta=f"{(m['median'] / p0 - 1) * 100:+.1f}%",
              help="Половина смоделированных траекторий оканчивается ниже этой цены, половина выше.")
    a3.metric("80%-коридор", f"{m['p10']:,.2f} – {m['p90']:,.2f}",
              help="В 80% смоделированных случаев цена через горизонт окажется внутри этого коридора.")
    a4.metric("Вероятность быть ниже текущей", f"{m['prob_below_current'] * 100:.0f}%")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Просадка 10% и более в любой момент", f"{m['prob_touch_drop_10'] * 100:.0f}%",
              help="Вероятность того, что цена хотя бы раз за горизонт опустится на 10% и более от текущей.")
    b2.metric("Просадка 20% и более в любой момент", f"{m['prob_touch_drop_20'] * 100:.0f}%")
    b3.metric("VaR 95%", f"{m['var95_pct']:.1f}%", help=f"В 5% случаев потеря за {months} мес. будет не меньше этой величины. В среднем в этих худших случаях: {m['cvar95_pct']:.1f}% (CVaR).")
    b4.metric("Волатильность (год)", f"{params['sigma_annual'] * 100:.1f}%")

    _fan_chart(res, history, currency, color, ticker)

    hist_fig = px.histogram(x=res["terminal"], nbins=40, title=f"Распределение цены через {months} мес., {currency}",
                            color_discrete_sequence=[color])
    hist_fig.add_vline(x=p0, line_dash="dot", line_color="gray")
    hist_fig.update_layout(xaxis_title=f"Цена, {currency}", yaxis_title="Число траекторий", showlegend=False)
    show_chart(hist_fig)

    table = pd.DataFrame({"Процентиль": ["5%", "10%", "25%", "50% (медиана)", "75%", "90%", "95%"],
                          f"Цена, {currency}": [float(res["quantiles"][c].iloc[-1]) for c in ("p5", "p10", "p25", "p50", "p75", "p90", "p95")]})
    table["К текущей цене, %"] = (table[f"Цена, {currency}"] / p0 - 1) * 100
    show_table(table.set_index("Процентиль").round(2))

    cal_table, cal = calibrate(history, days, model, vol, drift)
    verdict = calibration_verdict(cal)
    with st.expander("🔍 Проверка модели на прошлом"):
        if cal["n"] == 0:
            st.info("Истории слишком мало для проверки на прошлом (нужно больше года торгов плюс горизонт).")
        else:
            st.caption("На каждой дате модель обучается только на более ранних данных и строит коридоры на выбранный горизонт; затем сравнивается с фактической ценой.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Проверок", cal["n"], help=f"Даты проверки идут с шагом около месяца, окна перекрываются: независимых проверок примерно {cal['n_eff']:.0f}.")
            k2.metric("Факт в 80%-коридоре", f"{cal['coverage80'] * 100:.0f}%", delta=f"{(cal['coverage80'] - 0.8) * 100:+.0f} п.п. к ожидаемым 80%")
            k3.metric("Факт в 50%-коридоре", f"{cal['coverage50'] * 100:.0f}%", delta=f"{(cal['coverage50'] - 0.5) * 100:+.0f} п.п. к ожидаемым 50%")
            k4.metric("Факт ниже медианы", f"{cal['below_median'] * 100:.0f}%", help="При хорошей калибровке около 50%.")
            margin = f"около ±{cal['margin80'] * 100:.0f} п.п. для 80%-коридора и ±{cal['margin50'] * 100:.0f} п.п. для 50%-коридора"
            if verdict == "мало проверок":
                st.warning(f"Независимых проверок мало (около {cal['n_eff']:.0f}): вывод о качестве модели ненадёжен.")
            elif verdict == "отклоняется":
                st.warning(f"Доля попаданий отличается от ожидаемой сильнее случайного разброса ({margin}). Модель на этой истории откалибрована плохо: "
                           "считайте коридоры грубым ориентиром.")
            else:
                st.info(f"Доля попаданий в пределах случайного разброса ({margin}). При таком числе независимых проверок (около {cal['n_eff']:.0f}) "
                        "заметны только крупные отклонения, поэтому это не доказательство качества модели.")
            show_table(_round_numeric(cal_table, 2).set_index("Дата прогноза"))
    st.caption(f"Параметры: модель «{model_label}», волатильность «{vol_label}», дрейф «{drift_label}»; {len(res['terminal']):,} траекторий; "
               f"дневная волатильность {params['sigma'] * 100:.2f}%. " + DISCLAIMER)

    export_q = _round_numeric(res["quantiles"].assign(Дата=res["dates"]), 4)
    out = {"Диапазон цены": export_q}
    if cal["n"]:
        out["Диапазон цены - проверка"] = cal_table
    return out
