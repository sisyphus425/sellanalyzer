"""Вкладка «Что если»: сценарный калькулятор. Пользователь задает допущения, приложение считает результат."""

import pandas as pd
import plotly.express as px
import streamlit as st

from salesanalyzer.config import WHATIF_MAX_YEARS
from salesanalyzer.core.whatif import (LIMITS, PRESETS, SENSITIVITY_METRICS, default_growths, growth_source_reliability, make_params, scenario,
                                       sensitivity, with_price)
from salesanalyzer.ui.compat import show_chart, show_table

FACT_NAME, SCENARIO_NAME = "Факт", "Сценарий"
DISCLAIMER = ("Калькулятор не предсказывает будущее: он считает следствия ваших допущений. Результат хорош ровно настолько, насколько реалистичны допущения. "
              "«Цена при неизменном P/E» это арифметика (текущий P/E × EPS сценария), а не оценка справедливой цены и не рекомендация.")


def _fmt(value, template="{:,.1f}"):
    return "н/д" if value is None or pd.isna(value) else template.format(value)


def _delta(new, old):
    """Рост в % между положительными значениями для подписи под метрикой."""
    if new is None or old is None or pd.isna(new) or pd.isna(old) or new <= 0 or old <= 0:
        return None
    return f"{(new / old - 1) * 100:+.1f}% к факту"


def _bar(df, column, title, unit, color):
    data = pd.DataFrame({"Год": df["Год"].astype(int).astype(str), column: df[column],
                         "Тип": [FACT_NAME] + [SCENARIO_NAME] * (len(df) - 1)}).dropna(subset=[column])
    if data.empty:
        return
    fig = px.bar(data, x="Год", y=column, color="Тип", text_auto=".4s", title=title,
                 color_discrete_map={FACT_NAME: color, SCENARIO_NAME: "#F2B01E"})
    fig.update_layout(xaxis_type="category", yaxis_title=unit, legend_title="")
    show_chart(fig)


def render_whatif(base, notes, series, key: str, color: str) -> dict:
    """Калькулятор «что если». base: результат build_base (или None), notes: замечания, series: (годы, выручка) для
    начальных значений роста из базового прогноза. Возвращает таблицы для экспорта."""
    for n in notes:
        st.caption("⚠️ " + n)
    if base is None:
        st.info("Калькулятор недоступен: не хватает данных (нужны выручка и чистая прибыль хотя бы за один год).")
        return {}
    cur = base["currency"]
    unit = f"млрд {cur}"
    st.caption("Задайте допущения по годам, и калькулятор посчитает прибыль, прибыль на акцию (EPS), P/E и дивиденд. "
               "Это арифметика допущений, а не прогноз.")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric(f"Факт {base['year']} г.: выручка", f"{base['revenue']:,.1f} {unit}")
    a2.metric("Чистая прибыль", f"{base['net_income']:,.1f} {unit}", delta=f"маржа {base['margin']:.1f}%", delta_color="off")
    a3.metric("EPS", _fmt(base["eps"], "{:,.2f} " + cur), help="Чистая прибыль / число акций.")
    a4.metric("Дивиденд на акцию", _fmt(base["dps"], "{:,.2f} " + cur),
              delta=None if base["payout"] is None else f"payout {base['payout']:.0f}%", delta_color="off")

    c1, c2, c3 = st.columns(3)
    with c1:
        preset = st.selectbox("Набор допущений:", list(PRESETS), key=f"wi_preset_{key}",
                              help="Готовые наборы; значения ниже можно менять вручную. «Как в базовом прогнозе» берёт рост выручки из вкладки «Прогноз».")
    with c2:
        horizon = st.selectbox("Горизонт, лет:", list(range(1, WHATIF_MAX_YEARS + 1)), key=f"wi_h_{key}")
    with c3:
        price_in = st.number_input(f"Цена акции для расчёта, {cur}", value=float(base["price"] or 0.0), min_value=0.0, step=1.0,
                                   key=f"wi_price_{key}", help="Подставлена последняя цена закрытия; 0 означает «цена неизвестна».")
    calc_base = with_price(base, price_in) if price_in != (base["price"] or 0.0) else base

    growths = default_growths(series[0], series[1], horizon) if series is not None else None
    defaults = make_params(calc_base, horizon, preset, growths)
    if growths is not None and preset in (PRESETS[0], PRESETS[3], PRESETS[4]):
        reliability = growth_source_reliability(series[0], series[1])
        st.caption(f"Начальный рост выручки взят из базового прогноза (надёжность прогноза: {reliability}). Проверьте допущения и измените их под свой взгляд.")
    if base["payout"] is not None and base["payout"] > LIMITS["payout"][1]:
        st.caption(f"⚠️ Payout базового года ({base['payout']:.0f}%) превышает допустимый предел {LIMITS['payout'][1]:.0f}%: в допущениях стоит предел "
                   "(у компании прибыль в базовом году нетипично мала или определена иначе, см. README).")
    st.markdown("**Допущения по годам**")
    user = []
    for i, d in enumerate(defaults, start=1):
        year = base["year"] + i
        k = f"{key}_{preset}_{horizon}"
        cols = st.columns(4)
        with cols[0]:
            g = st.number_input(f"Рост выручки в {year} г., %", value=round(d["growth"], 1), min_value=LIMITS["growth"][0], max_value=LIMITS["growth"][1], step=1.0, key=f"wi_g{i}_{k}")
        with cols[1]:
            m = st.number_input(f"Чистая маржа в {year} г., %", value=round(d["margin"], 1), min_value=LIMITS["margin"][0], max_value=LIMITS["margin"][1], step=0.5, key=f"wi_m{i}_{k}")
        with cols[2]:
            s = st.number_input(f"Изменение числа акций в {year} г., %", value=0.0, min_value=LIMITS["shares_change"][0], max_value=LIMITS["shares_change"][1], step=0.5, key=f"wi_s{i}_{k}",
                                help="Минус: выкуп акций. Плюс: дополнительная эмиссия.")
        with cols[3]:
            p = st.number_input(f"Payout в {year} г., %", value=round(d["payout"], 1), min_value=LIMITS["payout"][0], max_value=LIMITS["payout"][1], step=5.0, key=f"wi_p{i}_{k}",
                                help="Доля чистой прибыли, направляемая на дивиденды (при убытке дивиденды не платятся).")
        user.append({"growth": g, "margin": m, "shares_change": s, "payout": p})

    df = scenario(calc_base, user)
    final, year_f = df.iloc[-1], int(df["Год"].iloc[-1])

    st.markdown(f"#### Результат на {year_f} г.")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Чистая прибыль", f"{final['Чистая прибыль']:,.1f} {unit}", delta=_delta(final["Чистая прибыль"], base["net_income"]))
    r2.metric("EPS", _fmt(final["EPS"], "{:,.2f} " + cur), delta=_delta(final["EPS"], base["eps"]))
    r3.metric("P/E по цене акции", _fmt(final["P/E"], "{:.1f}"), help="Цена для расчёта / EPS сценария. При убытке не определён.")
    r4.metric("Дивиденд на акцию", _fmt(final["Дивиденд на акцию"], "{:,.2f} " + cur),
              delta=None if pd.isna(final["Дивидендная доходность, %"]) else f"доходность {final['Дивидендная доходность, %']:.1f}%", delta_color="off")
    s1, s2, s3 = st.columns(3)
    s1.metric("Выручка", f"{final['Выручка']:,.1f} {unit}", delta=_delta(final["Выручка"], base["revenue"]))
    s2.metric("Дивиденды всего", f"{final['Дивиденды всего']:,.1f} {unit}")
    implied = final["Цена при неизменном P/E"]
    s3.metric("Цена при неизменном P/E", _fmt(implied, "{:,.2f} " + cur),
              delta=None if pd.isna(implied) or not calc_base["price"] else f"{(implied / calc_base['price'] - 1) * 100:+.1f}% к цене",
              help="Текущий P/E × EPS сценария: чему равнялась бы цена, если бы мультипликатор не изменился. Это арифметика, не оценка стоимости.")

    left, right = st.columns(2)
    with left:
        _bar(df, "Чистая прибыль", f"Чистая прибыль: факт и сценарий, {unit}", unit, color)
    with right:
        _bar(df, "EPS", f"EPS: факт и сценарий, {cur}", cur, color)

    show_table(df.drop(columns=[]).round(2).set_index("Год"))

    metric = st.radio("Чувствительность показателя:", list(SENSITIVITY_METRICS), horizontal=True, key=f"wi_sens_{key}")
    grid = sensitivity(calc_base, user, metric)
    if grid.notna().any().any():
        fig = px.imshow(grid, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                        title=f"{metric} в {year_f} г.: что будет, если рост выручки и маржа окажутся другими (п.п. к вашим допущениям)")
        show_chart(fig)
    else:
        st.info("Для выбранного показателя таблица чувствительности недоступна (нет числа акций, цены или прибыль отрицательна).")
    st.caption(DISCLAIMER)

    assumptions = pd.DataFrame([{"Год": base["year"] + i, "Рост выручки, %": u["growth"], "Чистая маржа, %": u["margin"],
                                 "Изменение числа акций, %": u["shares_change"], "Payout, %": u["payout"]} for i, u in enumerate(user, start=1)])
    out = {"Сценарий - допущения": assumptions, "Сценарий - результат": df}
    if grid.notna().any().any():
        out["Сценарий - чувствительность"] = grid.reset_index(names=f"{metric}: изменение")
    return out
