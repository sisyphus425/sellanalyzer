"""
Анализатор продаж компаний.

РФ:  выручка берется из файла revenue_ru.csv (лежит рядом с этим скриптом),
     название компании и история цен - из MOEX ISS.
США: выручка и цены - из yfinance.

Чтобы обновить данные по РФ, редактируйте только revenue_ru.csv (код трогать не нужно).
Колонки CSV: ticker, year, revenue, unit, metric, is_estimate, source, note
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Анализ Продаж Компаний", layout="wide")

ISS = "https://iss.moex.com/iss"
CACHE_TTL = 60 * 60
CSV_PATH = Path(__file__).with_name("revenue_ru.csv")
GREEN, GREEN_LIGHT, BLUE = "#00A86B", "#9AD8BF", "#1f77b4"


# ============ Данные РФ: CSV ============

@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    """mtime в аргументах: при изменении файла кэш сбрасывается сам."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": str})
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["year"] = df["year"].astype(int)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["is_estimate"] = df["is_estimate"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "да"])
    for col in ("unit", "metric", "source", "note"):
        df[col] = df[col].fillna("").astype(str)
    return df.dropna(subset=["revenue"])


def load_revenue_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame()
    return _read_csv(str(CSV_PATH), CSV_PATH.stat().st_mtime)


# ============ MOEX ISS ============

def iss_get(path: str, **params) -> dict:
    r = requests.get(f"{ISS}/{path}.json", params={"iss.meta": "off", **params}, timeout=15)
    r.raise_for_status()
    return {k: pd.DataFrame(v["data"], columns=v["columns"])
            for k, v in r.json().items() if isinstance(v, dict) and "columns" in v}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_moex_name(ticker: str):
    """Полное название бумаги на Мосбирже или None, если тикер неизвестен."""
    desc = iss_get(f"securities/{ticker}", **{"iss.only": "description"}).get("description")
    if desc is None or desc.empty:
        return None
    return dict(zip(desc["name"], desc["value"])).get("NAME", ticker)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_moex_candles(ticker: str, years: int = 3) -> pd.DataFrame:
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    frames, cursor = [], 0
    while True:  # ISS отдает максимум 500 свечей за запрос
        df = iss_get(f"engines/stock/markets/shares/securities/{ticker}/candles",
                     interval=24, **{"from": start, "start": cursor}).get("candles")
        if df is None or df.empty:
            break
        frames.append(df)
        cursor += len(df)
        if len(df) < 500:
            break
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["begin"] = pd.to_datetime(out["begin"])
    return out


# ============ США: yfinance ============

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_us_data(ticker: str):
    company = yf.Ticker(ticker)
    info = company.info or {}
    name = info.get("longName")
    if not name:
        return None
    df = None
    fin = company.financials
    if fin is not None and not fin.empty and "Total Revenue" in fin.index:
        rev = fin.loc["Total Revenue"].dropna()
        if not rev.empty:
            df = pd.DataFrame({"year": rev.index.year, "revenue": rev.values}).sort_values("year")
    return {"name": name, "sector": info.get("sector") or "Не указан",
            "website": info.get("website"), "df": df,
            "currency": info.get("currency") or "USD"}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_us_prices(ticker: str, years: int = 3) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=f"{years}y", interval="1d")
    if hist is None or hist.empty:
        return pd.DataFrame()
    out = hist[["Close"]].reset_index()
    out.columns = ["begin", "close"]
    out["begin"] = pd.to_datetime(out["begin"]).dt.tz_localize(None)
    return out


# ============ Отрисовка ============

def fmt_value(v: float, unit: str) -> str:
    return f"{v:,.1f} {unit}".strip() if abs(v) < 100 else f"{v:,.0f} {unit}".strip()


def render_revenue(df: pd.DataFrame, unit: str, color: str, color_est: str, title: str):
    """df: колонки year, revenue, is_estimate (+ необязательно metric, source, note)."""
    df = df.sort_values("year").reset_index(drop=True)
    last = df.iloc[-1]
    growth = None
    if len(df) > 1 and df["year"].iloc[-1] - df["year"].iloc[-2] == 1 and df["revenue"].iloc[-2]:
        prev = df["revenue"].iloc[-2]
        growth = (last["revenue"] - prev) / prev * 100

    m1, m2 = st.columns(2)
    m1.metric(f"Выручка за {int(last['year'])} г.", ("≈ " if last["is_estimate"] else "") + fmt_value(last["revenue"], unit))
    if growth is None:
        m2.metric("Годовая динамика", "н/д")
    else:
        m2.metric("Годовая динамика", f"{growth:.2f}%", delta=f"{growth:.2f}%")

    plot = df.assign(Год=df["year"].astype(str),
                     Тип=df["is_estimate"].map({True: "Расчетная оценка", False: "Опубликовано"}),
                     Значение=df["revenue"])
    fig = px.bar(plot, x="Год", y="Значение", color="Тип", text_auto=".4s", title=title,
                 color_discrete_map={"Опубликовано": color, "Расчетная оценка": color_est})
    fig.update_layout(xaxis_type="category", yaxis_title=unit)
    st.plotly_chart(fig, use_container_width=True)  # в новых версиях Streamlit: width="stretch"

    with st.expander("📄 Исходная таблица данных, источники и комментарии"):
        cols = {"year": "Год", "revenue": f"Выручка ({unit})", "metric": "Показатель",
                "source": "Источник", "note": "Комментарий"}
        table = df[[c for c in cols if c in df.columns]].rename(columns=cols)
        st.dataframe(table.set_index("Год"), use_container_width=True)


def render_price(prices: pd.DataFrame, ticker: str, currency: str, color: str):
    if prices.empty:
        st.info("История цен недоступна для этого тикера.")
        return
    last, first = prices["close"].iloc[-1], prices["close"].iloc[0]
    p1, p2 = st.columns(2)
    p1.metric(f"Последняя цена закрытия, {currency}", f"{last:,.2f}")
    if first:
        change = (last - first) / first * 100
        p2.metric("Изменение за период", f"{change:.2f}%", delta=f"{change:.2f}%")
    fig = px.line(prices, x="begin", y="close", title=f"Цена закрытия {ticker} ({currency})",
                  color_discrete_sequence=[color])
    fig.update_layout(xaxis_title="Дата", yaxis_title=f"Цена, {currency}")
    st.plotly_chart(fig, use_container_width=True)


# ============ UI ============

st.title("📊 Анализатор продаж компаний (РФ и Мир)")
st.write("Введите только тикер: для РФ — код Мосбиржи (SBER, GAZP, LKOH, YDEX, T, GMKN), "
         "для США — AAPL, TSLA и т.д.")

col_market, col_input = st.columns(2)
with col_market:
    market = st.radio("Рынок:", ["Россия (Мосбиржа)", "США / Международный"], horizontal=True)
is_ru = market == "Россия (Мосбиржа)"
with col_input:
    ticker = st.text_input("Тикер:", value="SBER" if is_ru else "AAPL",
                           key=f"ticker_{'ru' if is_ru else 'us'}").strip().upper()

if ticker and is_ru:
    revenue_all = load_revenue_csv()
    if revenue_all.empty:
        st.error(f"Файл {CSV_PATH.name} не найден или пуст. Он должен лежать рядом с приложением: {CSV_PATH.parent}")
        st.stop()

    rows = revenue_all[revenue_all["ticker"] == ticker]

    name = None
    try:
        name = get_moex_name(ticker)
    except requests.RequestException:
        st.warning("Не удалось связаться с MOEX: название и цены могут быть недоступны.")

    if name is None and rows.empty:
        st.error(f"❌ Тикер **{ticker}** не найден на Мосбирже и отсутствует в {CSV_PATH.name}.")
        st.stop()

    st.subheader(f"🏢 {name or ticker} ({ticker})")
    tab_rev, tab_price = st.tabs(["Выручка (МСФО)", "Цена акции"])

    with tab_rev:
        if rows.empty:
            st.warning(f"Для **{ticker}** нет данных о выручке в {CSV_PATH.name}. "
                       "Добавьте строки в файл — код менять не нужно.")
            st.caption("Тикеры в файле: " + ", ".join(sorted(revenue_all["ticker"].unique())))
        else:
            units = rows["unit"].unique()
            unit = units[-1]
            if len(units) > 1:
                st.warning("В файле для этого тикера указаны разные единицы измерения — проверьте колонку unit.")
            st.caption(f"Показатель: {rows['metric'].iloc[-1]}. Данные из файла {CSV_PATH.name}; "
                       "светлые столбцы — расчетные оценки.")
            render_revenue(rows, unit, GREEN, GREEN_LIGHT, f"Выручка {ticker} по годам")

    with tab_price:
        try:
            render_price(get_moex_candles(ticker), ticker, "₽", GREEN)
        except requests.RequestException as e:
            st.error(f"Не удалось загрузить историю цен: {e}")

elif ticker:
    try:
        with st.spinner(f"Загружаем данные для {ticker}..."):
            data = get_us_data(ticker)
    except Exception as e:
        st.error(f"Не удалось загрузить данные: {e}")
        st.stop()

    if data is None:
        st.error(f"❌ Тикер **{ticker}** не найден.")
    else:
        st.subheader(f"🏢 {data['name']}")
        site = data["website"]
        st.write(f"**Сектор:** {data['sector']} | **Сайт:** {f'[{site}]({site})' if site else '—'}")
        tab_rev, tab_price = st.tabs(["Выручка", "Цена акции"])

        with tab_rev:
            if data["df"] is None:
                st.warning(f"⚠️ Профиль {ticker} найден, но открытых данных по выручке нет.")
            else:
                df = data["df"].assign(is_estimate=False)
                render_revenue(df, "$", BLUE, BLUE, f"История продаж {ticker} по годам")

        with tab_price:
            try:
                render_price(get_us_prices(ticker), ticker, data["currency"], BLUE)
            except Exception as e:
                st.error(f"Не удалось загрузить историю цен: {e}")
