"""Вкладка «Прогноз дивидендов»: прогнозная чистая прибыль × payout ÷ число акций."""

import pandas as pd
import plotly.express as px
import streamlit as st

from salesanalyzer.config import FORECAST_HORIZONS
from salesanalyzer.core.dividends import (dividend_forecast, historical_payouts, payout_choices, policy_gap)
from salesanalyzer.core.forecast import forecast_series
from salesanalyzer.ui.compat import show_chart, show_table

DISCLAIMER = ("Это арифметика допущений, а не обещание выплат: дивиденд решает совет директоров и собрание акционеров, "
              "а политика компании это ориентир, который может измениться или не выполняться. Прогноз прибыли сам неточен "
              "(см. вкладку «Прогноз»), поэтому дивиденд показан диапазоном.")
YELLOW = "#F2B01E"


def render_dividend_forecast(w, unit_to_bn, shares_mn, price, policy, key, color, estimates=None) -> dict:
    """w: таблица по годам в рублях (year, net_income, dps); unit_to_bn: множитель единиц в млрд; shares_mn: число акций, млн;
    price: цена акции или None; policy: результат policy_from_rows или None. Возвращает таблицы для экспорта."""
    if "net_income" not in w.columns or not w["net_income"].notna().any():
        st.info("Нет данных о чистой прибыли: прогноз дивидендов не строится.")
        return {}
    if not shares_mn:
        st.info("Не указано число акций (indicator = shares): дивиденд на акцию посчитать нельзя.")
        return {}
    ni = w.dropna(subset=["net_income"])
    ni_bn = {int(r["year"]): float(r["net_income"]) * unit_to_bn for _, r in ni.iterrows()}
    dps_rows = w.dropna(subset=["dps"]) if "dps" in w.columns else pd.DataFrame(columns=["year", "dps"])
    dps_by_year = {int(r["year"]): float(r["dps"]) for _, r in dps_rows.iterrows()}
    history = historical_payouts(ni_bn, dps_by_year, shares_mn)
    avg, policy_pct, warn = policy_gap(history, policy["pct"] if policy else None)
    last_dps = dps_by_year[max(dps_by_year)] if dps_by_year else None

    if policy:
        st.markdown(f"**Политика компании:** payout {policy['pct']:.0f}% ({policy['year']} г.)")
        st.caption(f"Источник: {policy['source']}. {policy['note']}")
    else:
        st.info("Для этой компании в файле данных нет целевого payout (indicator = payout_target): "
                "используйте средний фактический или введите значение вручную. Политика может быть не привязана к чистой прибыли "
                "(например, к свободному денежному потоку), тогда простой расчёт не подходит.")
    if warn:
        st.warning("⚠️ " + warn[0].upper() + warn[1:] + ".")

    choices = payout_choices(policy_pct, avg)
    options = list(choices) + ["Вручную"]
    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("Payout:", options, key=f"dv_mode_{key}")
    with c2:
        default = choices.get(mode, policy_pct if policy_pct is not None else (avg if avg is not None else 30.0))
        payout = st.number_input("Значение payout, %", min_value=0.0, max_value=300.0, value=float(min(max(default, 0.0), 300.0)),
                                 step=5.0, key=f"dv_val_{key}_{mode}", disabled=mode != "Вручную")
        if mode != "Вручную":
            payout = choices[mode]
    with c3:
        horizon = st.selectbox("Горизонт, лет:", list(FORECAST_HORIZONS), key=f"dv_h_{key}")

    res = forecast_series(ni["year"], ni["net_income"], horizon, (estimates or {}).get("net_income", ()))
    scen = res["scenarios"]
    if scen.empty:
        st.info("Прогноз прибыли не строится: " + "; ".join(res["reliability"]["reasons"]) + ".")
        return {}
    scen = scen.assign(**{c: scen[c] * unit_to_bn for c in ("Низкий", "База", "Высокий")})
    fc = dividend_forecast(scen, shares_mn, payout, price, last_dps)
    first = fc.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Дивиденд на акцию, база, {int(first['Год'])} г.", f"{first['Дивиденд на акцию (база)']:,.2f} ₽",
              delta=None if pd.isna(first["Рост к последнему факту, %"]) else f"{first['Рост к последнему факту, %']:+.1f}% к последнему факту",
              help="Прогнозная чистая прибыль (база) × payout ÷ число акций.")
    m2.metric("Диапазон", f"{first['Дивиденд на акцию (низкий)']:,.2f} – {first['Дивиденд на акцию (высокий)']:,.2f} ₽",
              help="По низкому и высокому сценарию прибыли с прогноза.")
    m3.metric("Доходность к цене (база)", "н/д" if pd.isna(first["Дивидендная доходность (база), %"]) else f"{first['Дивидендная доходность (база), %']:.1f}%",
              help="Дивиденд базового сценария / текущая цена.")
    if res["reliability"]["label"] == "низкая":
        st.warning("Надёжность прогноза прибыли низкая: считайте дивиденд грубым ориентиром.")

    hist = pd.DataFrame({"Год": history["Год"].astype(str), "Дивиденд на акцию": history["Дивиденд на акцию"], "Тип": "Факт"})
    fut = pd.DataFrame({"Год": fc["Год"].astype(str), "Дивиденд на акцию": fc["Дивиденд на акцию (база)"], "Тип": "Прогноз (база)"})
    chart = pd.concat([hist, fut], ignore_index=True)
    if not chart.empty:
        fig = px.bar(chart, x="Год", y="Дивиденд на акцию", color="Тип", text_auto=".2f", title="Дивиденд на акцию: факт и прогноз, ₽",
                     color_discrete_map={"Факт": color, "Прогноз (база)": YELLOW})
        fig.update_layout(xaxis_type="category", yaxis_title="₽", legend_title="")
        show_chart(fig)

    show_table(fc.set_index("Год").round(2))
    if not history.empty:
        with st.expander("📜 Фактический payout по годам"):
            st.caption("Считается по текущему числу акций, поэтому для прошлых лет это приближение.")
            show_table(history.set_index("Год").round(1))
    st.caption(DISCLAIMER)

    out = {"Прогноз дивидендов": fc.assign(**{"Payout, %": payout})}
    if not history.empty:
        out["Дивиденды - фактический payout"] = history
    return out
