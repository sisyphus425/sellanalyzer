"""
Анализатор продаж компаний (рост, CAGR, светофор, сравнение, прибыльность и ROE, экспорт CSV/Excel, контроль качества данных, метрики цены, пересчет в рубли и инфляция, мультипликаторы и дивиденды).

РФ:  финансовые показатели (выручка, чистая прибыль, EBITDA, ROE) берутся из файла financials_ru.csv
     (лежит рядом с этим скриптом), название компании и история цен - из MOEX ISS.
США: выручка и цены - из yfinance.

Чтобы обновить данные по РФ, редактируйте только financials_ru.csv (код трогать не нужно).
Колонки CSV: ticker, year, indicator, value, unit, metric, is_estimate, source, note
Курсы ЦБ и инфляция для пересчета: macro_ru.csv (year, indicator = usd_rub | cpi, value, unit, source, note)
indicator: revenue | net_income | ebitda | roe
"""
import re
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Анализ Продаж Компаний", layout="wide")

ISS = "https://iss.moex.com/iss"
CACHE_TTL = 60 * 60
CSV_PATH = Path(__file__).with_name("financials_ru.csv")
REQUIRED_COLS = ["ticker", "year", "indicator", "value", "unit", "metric", "is_estimate", "source", "note"]
MACRO_PATH = Path(__file__).with_name("macro_ru.csv")
MACRO_COLS = ["year", "indicator", "value", "unit", "source", "note"]
GREEN, GREEN_LIGHT, BLUE = "#00A86B", "#9AD8BF", "#1f77b4"


# ============ Данные РФ: CSV ============

@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    """mtime в аргументах: при изменении файла кэш сбрасывается сам."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": str})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("в файле нет колонок: " + ", ".join(missing))
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["indicator"] = df["indicator"].astype(str).str.strip().str.lower()
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["is_estimate"] = df["is_estimate"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "да"])
    for col in ("unit", "metric", "source", "note"):
        df[col] = df[col].fillna("").astype(str)
    return df.dropna(subset=["value"])


def load_ru_all() -> pd.DataFrame:
    """Все показатели РФ в длинном формате (ticker, year, indicator, value, ...)."""
    if not CSV_PATH.exists():
        return pd.DataFrame()
    try:
        return _read_csv(str(CSV_PATH), CSV_PATH.stat().st_mtime)
    except ValueError as e:
        st.error(f"Ошибка в файле {CSV_PATH.name}: {e}")
        return pd.DataFrame()


def load_revenue_csv() -> pd.DataFrame:
    """Только выручка; значение лежит в колонке revenue."""
    df = load_ru_all()
    if df.empty:
        return df
    rev = df[df["indicator"] == "revenue"].rename(columns={"value": "revenue"})
    return rev.drop(columns=["indicator"]).reset_index(drop=True)


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

def us_wide(fin: pd.DataFrame) -> pd.DataFrame:
    """Годовая отчетность yfinance -> таблица по годам: revenue, net_income, ebitda, roe (NaN, если нет)."""
    rows = {"revenue": "Total Revenue", "net_income": "Net Income", "ebitda": "EBITDA"}
    data = {}
    for key, label in rows.items():
        if label in fin.index:
            ser = fin.loc[label].dropna()
            if not ser.empty:
                ser = pd.Series(ser.values, index=ser.index.year)
                data[key] = ser[~ser.index.duplicated(keep="last")]
    if not data:
        return pd.DataFrame()
    w = pd.DataFrame(data)
    w.index.name = "year"
    w = w.reset_index().sort_values("year").reset_index(drop=True)
    for c in ("revenue", "net_income", "ebitda", "roe"):
        if c not in w.columns:
            w[c] = float("nan")
    w["has_estimate"] = False
    return w


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_us_data(ticker: str):
    company = yf.Ticker(ticker)
    info = company.info or {}
    name = info.get("longName")
    if not name:
        return None
    df, wide = None, pd.DataFrame()
    fin = company.financials
    if fin is not None and not fin.empty:
        wide = us_wide(fin)
        if "Total Revenue" in fin.index:
            rev = fin.loc["Total Revenue"].dropna()
            if not rev.empty:
                df = pd.DataFrame({"year": rev.index.year, "revenue": rev.values}).sort_values("year")
    div_df, div_ttm = pd.DataFrame(), None
    try:
        div = company.dividends
        if div is not None and not div.empty:
            div_df = div.groupby(div.index.year).sum().rename("dps").rename_axis("year").reset_index()
            div_ttm = float(div[div.index > pd.Timestamp.now(tz=div.index.tz) - pd.Timedelta(days=365)].sum())
    except Exception:
        pass
    return {"name": name, "sector": info.get("sector") or "Не указан",
            "website": info.get("website"), "df": df, "wide": wide,
            "currency": info.get("currency") or "USD", "fin_currency": info.get("financialCurrency"),
            "market_cap": info.get("marketCap"), "shares": info.get("sharesOutstanding"),
            "dividends": div_df, "div_ttm": div_ttm}


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


GROWTH_UP, GROWTH_DOWN = 5.0, -5.0  # пороги "светофора", % (меняйте под себя)
SIGNAL_COLORS = {"Рост": "#00A86B", "Стагнация": "#F2B01E", "Падение": "#D64545"}


def signal_name(g) -> str:
    if g is None or pd.isna(g):
        return "н/д"
    if g >= GROWTH_UP:
        return "Рост"
    if g <= GROWTH_DOWN:
        return "Падение"
    return "Стагнация"


def signal_icon(g) -> str:
    return {"Рост": "🟢", "Стагнация": "🟡", "Падение": "🔴"}.get(signal_name(g), "⚪")


def add_growth(df: pd.DataFrame, col: str = "revenue") -> pd.DataFrame:
    """Колонка growth (% к предыдущему году). Только для соседних лет и положительных значений в обоих годах."""
    df = df.sort_values("year").reset_index(drop=True).copy()
    prev = df[col].shift(1)
    consecutive = df["year"].diff() == 1
    df["growth"] = ((df[col] - prev) / prev * 100).where(consecutive & (prev > 0) & (df[col] > 0))
    return df


def calc_cagr(df: pd.DataFrame, col: str = "revenue"):
    """Среднегодовой темп роста между первым и последним годом, % (или None)."""
    if len(df) < 2:
        return None
    first, last = df.iloc[0], df.iloc[-1]
    years = int(last["year"] - first["year"])
    if years < 1 or first[col] <= 0 or last[col] <= 0:
        return None
    return ((last[col] / first[col]) ** (1 / years) - 1) * 100


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
    st.plotly_chart(fig, use_container_width=True)  # в новых версиях Streamlit: width="stretch"

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
            st.plotly_chart(gfig, use_container_width=True)
        with c2:
            table = pd.DataFrame({
                "Год": gdf["Год"],
                f"Выручка ({unit})": gdf["revenue"].map(lambda v: f"{v:,.1f}"),
                "Рост г/г": gdf["growth"].map(lambda v: f"{v:+.1f}%"),
                "": gdf["growth"].map(signal_icon),
            }).set_index("Год")
            st.dataframe(table, use_container_width=True)

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
        st.dataframe(table.set_index("Год"), use_container_width=True)


# ============ Прибыльность ============

INDICATOR_NAMES = {"revenue": "Выручка", "net_income": "Чистая прибыль", "ebitda": "EBITDA", "roe": "ROE",
                   "dps": "Дивиденд на акцию", "shares": "Число акций"}
MONEY_INDICATORS = ("revenue", "net_income", "ebitda")  # денежные показатели одной единицы измерения


def make_wide(rows: pd.DataFrame):
    """Длинная таблица (year, indicator, value, unit, is_estimate) -> по годам. Возвращает (wide, unit, mixed_units)."""
    if rows.empty:
        return pd.DataFrame(), "", False
    p = rows.pivot_table(index="year", columns="indicator", values="value", aggfunc="last")
    for c in ("revenue", "net_income", "ebitda", "roe", "dps", "shares"):
        if c not in p.columns:
            p[c] = float("nan")
    p["has_estimate"] = rows.groupby("year")["is_estimate"].any()
    units = rows.loc[rows["indicator"].isin(MONEY_INDICATORS), "unit"].unique()
    unit = units[-1] if len(units) else ""
    return p.reset_index(), unit, len(units) > 1


def add_margins(w: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    """Чистая маржа и EBITDA-маржа в % от выручки."""
    w = w.sort_values("year").reset_index(drop=True).copy()
    if enabled:
        base = w["revenue"].where(w["revenue"] > 0)
        w["net_margin"] = w["net_income"] / base * 100
        w["ebitda_margin"] = w["ebitda"] / base * 100
    else:
        w["net_margin"] = float("nan")
        w["ebitda_margin"] = float("nan")
    return w


def latest_value(w: pd.DataFrame, col: str):
    s_ = w.dropna(subset=[col])
    return None if s_.empty else (int(s_["year"].iloc[-1]), float(s_[col].iloc[-1]))


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
        st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)
    if has_roe:
        with right:
            rd = w.dropna(subset=["roe"]).assign(Год=lambda d: d["year"].astype(str))
            fig = px.bar(rd, x="Год", y="roe", text_auto=".1f", title="ROE (рентабельность капитала), %",
                         color_discrete_sequence=[color])
            fig.update_layout(xaxis_type="category", yaxis_title="%")
            st.plotly_chart(fig, use_container_width=True)

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
        st.dataframe(table, use_container_width=True)
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
            st.dataframe(t[["Показатель", "Год", "Значение", "Единицы", "Определение", "Оценка", "Источник", "Комментарий"]]
                         .set_index("Показатель"), use_container_width=True)


# ============ Экспорт ============

def to_csv_bytes(df: pd.DataFrame, sep: str = ",") -> bytes:
    """CSV в UTF-8 с BOM (русский текст корректно открывается в Excel). При ';' десятичный разделитель — запятая."""
    return df.to_csv(index=False, sep=sep, decimal="," if sep == ";" else ".").encode("utf-8-sig")


def to_excel_bytes(tables: dict):
    """Книга Excel: один лист на таблицу. None, если нет openpyxl или нет данных."""
    try:
        from openpyxl.utils import get_column_letter
    except ImportError:
        return None
    tables = {k: v for k, v in tables.items() if v is not None and not v.empty}
    if not tables:
        return None
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for name, df in tables.items():
            sheet = name[:31]
            df.to_excel(xw, sheet_name=sheet, index=False)
            ws = xw.sheets[sheet]
            for i, col in enumerate(df.columns, 1):
                width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)]) + 2
                ws.column_dimensions[get_column_letter(i)].width = min(width, 60)
            ws.freeze_panes = "A2"
    return bio.getvalue()


def export_tables(w: pd.DataFrame, unit: str, source_rows=None, name: str = "") -> dict:
    """Таблицы для выгрузки по одной компании: Сводка, Данные, Рост выручки, Прибыльность."""
    w = add_margins(w)
    tables = {}

    # Данные (длинный формат)
    if source_rows is not None and not source_rows.empty:
        t = source_rows.sort_values(["indicator", "year"])
        data = pd.DataFrame({
            "Показатель": t["indicator"].map(INDICATOR_NAMES).fillna(t["indicator"]),
            "Год": t["year"], "Значение": t["value"], "Единицы": t["unit"], "Определение": t["metric"],
            "Оценка": t["is_estimate"].map({True: "да", False: ""}),
            "Источник": t["source"], "Комментарий": t["note"],
        })
    else:
        cols = [c for c in ("revenue", "net_income", "ebitda", "roe") if w[c].notna().any()]
        if cols:
            long = (w.melt(id_vars="year", value_vars=cols, var_name="indicator", value_name="Значение")
                    .dropna(subset=["Значение"]).sort_values(["indicator", "year"]))
            data = pd.DataFrame({
                "Показатель": long["indicator"].map(INDICATOR_NAMES), "Год": long["year"],
                "Значение": long["Значение"],
                "Единицы": long["indicator"].map(lambda i: "%" if i == "roe" else unit),
                "Источник": "Yahoo Finance (yfinance)",
            })
        else:
            data = pd.DataFrame()

    # Рост выручки
    rev = w.dropna(subset=["revenue"])[["year", "revenue"]]
    growth_tbl = pd.DataFrame()
    if not rev.empty:
        g = add_growth(rev, "revenue")
        growth_tbl = pd.DataFrame({"Год": g["year"], f"Выручка ({unit})": g["revenue"],
                                   "Рост г/г, %": g["growth"].round(2), "Сигнал": g["growth"].map(signal_name)})

    # Прибыльность
    prof_tbl = w[["year", "revenue", "net_income", "ebitda", "roe", "net_margin", "ebitda_margin"]].rename(columns={
        "year": "Год", "revenue": "Выручка", "net_income": "Чистая прибыль", "ebitda": "EBITDA", "roe": "ROE, %",
        "net_margin": "Чистая маржа, %", "ebitda_margin": "EBITDA-маржа, %"}).round(2)
    prof_tbl.insert(1, "Единицы", unit)
    if not any(w[c].notna().any() for c in ("net_income", "ebitda", "roe")):
        prof_tbl = pd.DataFrame()

    # Сводка
    def span(df):
        return f"{int(df['year'].iloc[0])}–{int(df['year'].iloc[-1])}" if len(df) > 1 else "н/д"

    def pct(v):
        return "н/д" if v is None else round(v, 2)

    ni = w.dropna(subset=["net_income"])[["year", "net_income"]]
    summary = pd.DataFrame([
        ("Компания / тикер", name), ("Единицы (денежные показатели)", unit),
        ("Дата выгрузки", date.today().isoformat()),
        ("CAGR выручки, %", pct(calc_cagr(rev, "revenue"))), ("Период CAGR выручки", span(rev)),
        ("CAGR чистой прибыли, %", pct(calc_cagr(ni, "net_income"))), ("Период CAGR чистой прибыли", span(ni)),
    ], columns=["Параметр", "Значение"])

    return {"Сводка": summary, "Данные": data, "Рост выручки": growth_tbl, "Прибыльность": prof_tbl}


def summary_numeric(rev: pd.DataFrame, common_years: list) -> pd.DataFrame:
    """Числовая версия сводки роста и CAGR для выгрузки."""
    c_from, c_to = (min(common_years), max(common_years)) if len(common_years) >= 2 else (None, None)
    rows = []
    for label, g in rev.groupby("label", sort=False):
        g = g.sort_values("year")
        last = g.iloc[-1]
        cagr = calc_cagr(g)
        cagr_common = calc_cagr(g[(g["year"] >= c_from) & (g["year"] <= c_to)]) if c_from is not None else None
        rows.append({
            "Компания": label,
            "Период данных": f"{int(g['year'].iloc[0])}–{int(last['year'])}",
            "Рост к пред. году, %": None if pd.isna(last["growth"]) else round(last["growth"], 2),
            "CAGR весь ряд, %": None if cagr is None else round(cagr, 2),
            "CAGR общий период, %": None if cagr_common is None else round(cagr_common, 2),
            "Общий период": f"{c_from}–{c_to}" if c_from is not None else "",
            "Единицы": last["unit"],
            "Есть оценки": "да" if g["is_estimate"].any() else "нет",
        })
    return pd.DataFrame(rows)


def comparison_export_tables(rev: pd.DataFrame, common: list, prof: pd.DataFrame) -> dict:
    r = rev.sort_values(["label", "year"])
    tables = {
        "Выручка": pd.DataFrame({
            "Компания": r["label"], "Год": r["year"], "Выручка": r["revenue"], "Единицы": r["unit"],
            "Оценка": r["is_estimate"].map({True: "да", False: ""}),
            "Рост г/г, %": r["growth"].round(2), "Сигнал": r["growth"].map(signal_name)}),
        "Рост и CAGR": summary_numeric(rev, common),
    }
    if not prof.empty:
        p_ = prof.sort_values(["label", "year"])
        tables["Прибыльность"] = pd.DataFrame({
            "Компания": p_["label"], "Год": p_["year"], "Единицы": p_["unit"], "Выручка": p_["revenue"],
            "Чистая прибыль": p_["net_income"], "EBITDA": p_["ebitda"], "ROE, %": p_["roe"],
            "Чистая маржа, %": p_["net_margin"].round(2), "EBITDA-маржа, %": p_["ebitda_margin"].round(2),
            "Оценка": p_["has_estimate"].map({True: "да", False: ""})})
    return tables


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


# ============ Мультипликаторы и дивиденды ============

UNIT_TO_BN = {"млрд ₽": 1.0, "млн ₽": 1 / 1000, "₽": 1e-9, "млрд $": 1.0, "млн $": 1 / 1000, "$": 1e-9}
PAYOUT_COLS = ["Год", "Дивиденд на акцию", "Дивиденды всего, млрд ≈", "Чистая прибыль, млрд", "Payout, %", "Оценка"]


def last_close(prices: pd.DataFrame):
    """(последняя цена закрытия, дата) или (None, None)."""
    if prices is None or prices.empty:
        return None, None
    r = prices.sort_values("begin").iloc[-1]
    return float(r["close"]), r["begin"]


def _metric_of(rows: pd.DataFrame, indicator: str) -> str:
    sub = rows[rows["indicator"] == indicator].sort_values("year")
    return "" if sub.empty else sub["metric"].iloc[-1]


def _payout_rows(dps_rows, shares_mn, ni_bn_by_year, est_years, divisor):
    """Таблица дивидендов по годам: DPS, сумма выплат (≈ по текущему числу акций), payout к чистой прибыли."""
    out = []
    for _, r in dps_rows.sort_values("year").iterrows():
        y, dps = int(r["year"]), float(r["dps"])
        total = dps * shares_mn / divisor if shares_mn else None
        ni_bn = ni_bn_by_year.get(y)
        payout = total / ni_bn * 100 if total is not None and ni_bn and ni_bn > 0 else None
        out.append({"Год": y, "Дивиденд на акцию": dps,
                    "Дивиденды всего, млрд ≈": None if total is None else round(total, 1),
                    "Чистая прибыль, млрд": None if ni_bn is None else round(ni_bn, 1),
                    "Payout, %": None if payout is None else round(payout, 1),
                    "Оценка": "да" if y in est_years else ""})
    return pd.DataFrame(out, columns=PAYOUT_COLS)


def market_stats_ru(rows_long: pd.DataFrame, macro: dict, price, price_date):
    """Капитализация, P/S, P/E, дивиденды для российской компании. Денежные показатели в $ переводятся в ₽ по среднему курсу."""
    rows_rub, notes = adjust_long(rows_long, True, False, None, macro)
    w, unit, _ = make_wide(rows_rub)
    if w.empty:
        return None, pd.DataFrame(columns=PAYOUT_COLS), notes + ["Нет данных по компании."]
    k = UNIT_TO_BN.get(unit)
    if k is None:
        notes.append(f"Единицы «{unit}» не поддерживаются: мультипликаторы не считаются.")
    shares, rev, ni = latest_value(w, "shares"), latest_value(w, "revenue"), latest_value(w, "net_income")
    if shares is None:
        notes.append("Не указано число акций (indicator = shares): капитализация и мультипликаторы не считаются.")
    if price is None:
        notes.append("Нет цены акции: капитализация и мультипликаторы не считаются.")
    mcap = price * shares[1] / 1000 if price and shares else None  # млн шт. × ₽ = млн ₽ -> млрд ₽
    ps = mcap / (rev[1] * k) if mcap and rev and k and rev[1] > 0 else None
    pe = mcap / (ni[1] * k) if mcap and ni and k and ni[1] > 0 else None

    dps_rows = w.dropna(subset=["dps"])
    est_years = set(rows_rub[(rows_rub["indicator"] == "dps") & rows_rub["is_estimate"]]["year"].astype(int))
    ni_bn = {int(r["year"]): r["net_income"] * k for _, r in w.dropna(subset=["net_income"]).iterrows()} if k else {}
    payout = _payout_rows(dps_rows, shares[1] if shares else None, ni_bn, est_years, 1000)
    dps_last = None if dps_rows.empty else (int(dps_rows["year"].iloc[-1]), float(dps_rows["dps"].iloc[-1]))
    return {
        "price": price, "price_date": price_date, "currency": "₽", "mcap": mcap, "shares_mn": shares[1] if shares else None,
        "ps": ps, "pe": pe, "y_rev": rev[0] if rev else None, "y_ni": ni[0] if ni else None,
        "rev_metric": _metric_of(rows_rub, "revenue"), "ni_metric": _metric_of(rows_rub, "net_income"),
        "div_yield": dps_last[1] / price * 100 if dps_last and price else None,
        "dps_last": dps_last, "dps_est": bool(dps_last and dps_last[0] in est_years), "div_basis": "последний год из файла",
    }, payout, notes


def market_stats_us(data: dict, price, price_date):
    """Те же показатели для зарубежной компании по данным yfinance."""
    notes, w, cur = [], data["wide"], data["currency"]
    mcap_raw = data.get("market_cap")
    if data.get("fin_currency") and data["fin_currency"] != cur:
        notes.append(f"Валюта отчетности ({data['fin_currency']}) отличается от валюты торгов ({cur}): мультипликаторы не считаются.")
        mcap_raw = None
    if not mcap_raw:
        notes.append("Yahoo не вернул рыночную капитализацию: мультипликаторы не считаются.")
    rev = latest_value(w, "revenue") if not w.empty else None
    ni = latest_value(w, "net_income") if not w.empty else None
    ps = mcap_raw / rev[1] if mcap_raw and rev and rev[1] > 0 else None
    pe = mcap_raw / ni[1] if mcap_raw and ni and ni[1] > 0 else None
    shares = data.get("shares")
    shares_mn = shares / 1e6 if shares else None
    ni_bn = {int(r["year"]): r["net_income"] / 1e9 for _, r in w.dropna(subset=["net_income"]).iterrows()} if not w.empty else {}
    div_df = data.get("dividends")
    if div_df is None or div_df.empty:
        payout = pd.DataFrame(columns=PAYOUT_COLS)
    else:
        payout = _payout_rows(div_df.rename(columns={"dps": "dps"}), shares_mn, ni_bn, set(), 1000)
    ttm = data.get("div_ttm")
    return {
        "price": price, "price_date": price_date, "currency": cur, "mcap": mcap_raw / 1e9 if mcap_raw else None,
        "shares_mn": shares_mn, "ps": ps, "pe": pe, "y_rev": rev[0] if rev else None, "y_ni": ni[0] if ni else None,
        "rev_metric": "Total Revenue (Yahoo Finance)", "ni_metric": "Net Income (Yahoo Finance)",
        "div_yield": ttm / price * 100 if ttm and price else None,
        "dps_last": (int(div_df["year"].iloc[-1]), float(div_df["dps"].iloc[-1])) if div_df is not None and not div_df.empty else None,
        "dps_est": False, "div_basis": "выплаты за последние 12 месяцев",
    }, payout, notes


def market_tables(stats: dict, payout: pd.DataFrame) -> dict:
    cur = stats["currency"]
    kv = [(f"Цена, {cur}", stats["price"]), ("Дата цены", None if stats["price_date"] is None else f"{stats['price_date']:%Y-%m-%d}"),
          ("Число акций, млн", stats["shares_mn"]), (f"Капитализация, млрд {cur}", stats["mcap"]),
          (f"P/S (выручка {stats['y_rev']})", stats["ps"]), (f"P/E (прибыль {stats['y_ni']})", stats["pe"]),
          ("Дивдоходность, %", stats["div_yield"])]
    out = {"Мультипликаторы": pd.DataFrame(kv, columns=["Показатель", "Значение"])}
    if not payout.empty:
        out["Дивиденды"] = payout
    return out


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
            st.plotly_chart(fig, use_container_width=True)
        with right:
            pp = payout.dropna(subset=["Payout, %"])
            if pp.empty:
                st.info("Payout не рассчитан: нет прибыли или числа акций.")
            else:
                fig = px.bar(pp.assign(Год=pp["Год"].astype(str)), x="Год", y="Payout, %", text_auto=".0f",
                             title="Payout: доля чистой прибыли на дивиденды, %", color_discrete_sequence=[color])
                fig.update_layout(xaxis_type="category", yaxis_title="%")
                st.plotly_chart(fig, use_container_width=True)
        st.dataframe(payout.set_index("Год"), use_container_width=True)
    else:
        st.info("Данных о дивидендах нет.")

    st.caption("Мультипликаторы текущие: последняя цена, число акций из файла и показатели последнего отчетного года. "
               "«Дивиденды всего» и payout считаются по текущему числу акций, поэтому для прошлых лет это приближение. "
               "При необычной прибыли (разовые статьи, убыток) P/E может быть неинформативным — смотрите определение прибыли в подсказке. "
               "Справочная аналитика, не рекомендация.")
    return market_tables(stats, payout)


def market_summary_row(label: str, stats: dict, payout: pd.DataFrame) -> dict:
    pr = payout["Payout, %"].dropna() if not payout.empty else pd.Series(dtype=float)
    r = lambda v, n=2: None if v is None else round(v, n)
    return {"Компания": label, "Валюта": stats["currency"], "Цена": r(stats["price"]), "Капитализация, млрд": r(stats["mcap"], 1),
            "P/S": r(stats["ps"]), "P/E": r(stats["pe"], 1),
            "Дивиденд на акцию": None if stats["dps_last"] is None else stats["dps_last"][1],
            "Дивдоходность, %": r(stats["div_yield"]), "Payout, %": None if pr.empty else float(pr.iloc[-1]),
            "Год выручки": stats["y_rev"], "Год прибыли": stats["y_ni"]}


# ============ Метрики цены ============

TRADING_DAYS = 252  # торговых дней в году для годовой волатильности


def add_price_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Сортирует цены и добавляет скользящие средние MA50 и MA200 (по торговым дням)."""
    p = prices[["begin", "close"]].sort_values("begin").reset_index(drop=True).copy()
    p["ma50"] = p["close"].rolling(50).mean()
    p["ma200"] = p["close"].rolling(200).mean()
    return p


def price_metrics(p: pd.DataFrame):
    """Метрики по выбранному периоду (таблица после add_price_indicators). None, если данных мало."""
    if len(p) < 2 or p["close"].iloc[0] <= 0:
        return None
    close = p["close"]
    start, end = close.iloc[0], close.iloc[-1]
    days = (p["begin"].iloc[-1] - p["begin"].iloc[0]).days
    rets = close.pct_change().dropna()
    dd = (close / close.cummax() - 1) * 100
    i_min = int(dd.values.argmin())
    i_peak = int(close.iloc[: i_min + 1].values.argmax())
    ma50, ma200 = p["ma50"].iloc[-1], p["ma200"].iloc[-1]
    return {
        "last": end,
        "total": (end / start - 1) * 100,
        "annual": ((end / start) ** (365 / days) - 1) * 100 if days >= 90 else None,
        "vol": rets.std() * (TRADING_DAYS ** 0.5) * 100 if len(rets) >= 20 else None,
        "max_dd": dd.min(), "dd_peak": p["begin"].iloc[i_peak], "dd_trough": p["begin"].iloc[i_min],
        "cur_dd": dd.iloc[-1],
        "ma50": None if pd.isna(ma50) else ma50,
        "ma200": None if pd.isna(ma200) else ma200,
    }


def trend_label(m: dict) -> str:
    if m["ma200"] is None:
        return "н/д"
    return "🟢 выше MA200" if m["last"] > m["ma200"] else "🔴 ниже MA200"


def price_metrics_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица метрик по нескольким бумагам (колонки prices: begin, label, close)."""
    rows = []
    for label, g in prices.groupby("label", sort=False):
        m = price_metrics(add_price_indicators(g))
        if m is None:
            continue
        rows.append({
            "Компания": label,
            "Доходность за период, %": round(m["total"], 1),
            "Годовая доходность, %": None if m["annual"] is None else round(m["annual"], 1),
            "Волатильность (год), %": None if m["vol"] is None else round(m["vol"], 1),
            "Макс. просадка, %": round(m["max_dd"], 1),
            "Текущая просадка, %": round(m["cur_dd"], 1),
            "Тренд": trend_label(m),
        })
    return pd.DataFrame(rows)


def render_price(prices: pd.DataFrame, ticker: str, currency: str, color: str):
    """Вкладка 'Цена акции': метрики риска и доходности, график со скользящими средними, график просадки."""
    if prices.empty:
        st.info("История цен недоступна для этого тикера.")
        return

    years = st.selectbox("Период:", [1, 2, 3], index=2, format_func=lambda y: f"{y} г.", key=f"price_years_{ticker}")
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
    st.plotly_chart(fig, use_container_width=True)

    p["drawdown"] = (p["close"] / p["close"].cummax() - 1) * 100
    dfig = px.area(p, x="begin", y="drawdown", title="Просадка от максимума, %", color_discrete_sequence=["#D64545"])
    dfig.update_layout(xaxis_title="Дата", yaxis_title="%")
    st.plotly_chart(dfig, use_container_width=True)

    st.caption("MA — среднее цен закрытия за 50 и 200 торговых дней. Просадка — падение от предыдущего максимума. "
               "Прошлые показатели не гарантируют будущих результатов; это справочная аналитика, а не рекомендация.")
    out = p.assign(drawdown=p["drawdown"].round(2)).rename(columns={
        "begin": "Дата", "close": f"Цена закрытия, {currency}", "ma50": "MA50", "ma200": "MA200", "drawdown": "Просадка, %"})
    st.download_button("⬇️ Скачать цены и метрики (CSV)", data=to_csv_bytes(out), mime="text/csv",
                       key=f"dl_price_{ticker}", file_name=f"{ticker}_prices.csv")


# ============ Сравнение компаний: расчеты ============

MAX_COMPANIES = 6


def parse_tickers(text: str) -> list:
    return list(dict.fromkeys(t.strip().upper() for t in re.split(r"[,\s;]+", text) if t.strip()))


def collect_revenue(ru_tickers, us_tickers, revenue_all, problems) -> pd.DataFrame:
    """Единая таблица выручки: label, year, revenue, unit, is_estimate, growth."""
    cols = ["label", "year", "revenue", "unit", "is_estimate"]
    frames = []
    for t in ru_tickers:
        rows = revenue_all[revenue_all["ticker"] == t]
        if rows.empty:
            problems.append(f"{t}: нет данных в {CSV_PATH.name}")
            continue
        frames.append(rows.assign(label=t)[cols])
    for t in us_tickers:
        try:
            data = get_us_data(t)
        except Exception as e:
            problems.append(f"{t}: ошибка загрузки ({e})")
            continue
        if data is None:
            problems.append(f"{t}: тикер не найден")
        elif data["df"] is None:
            problems.append(f"{t}: нет открытых данных по выручке")
        else:
            frames.append(data["df"].assign(label=t, unit="$", is_estimate=False)[cols])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return pd.concat([add_growth(g) for _, g in out.groupby("label", sort=False)], ignore_index=True)


def index_revenue(rev: pd.DataFrame, base_year=None) -> pd.DataFrame:
    """Выручка в индексах (базовый год = 100). base_year=None: у каждой компании свой первый год."""
    parts = []
    for _, g in rev.groupby("label", sort=False):
        g = g.sort_values("year")
        if base_year is not None:
            g = g[g["year"] >= base_year]
        if g.empty or g["revenue"].iloc[0] <= 0:
            continue
        if base_year is not None and g["year"].iloc[0] != base_year:
            continue
        parts.append(g.assign(index=g["revenue"] / g["revenue"].iloc[0] * 100))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summary_table(rev: pd.DataFrame, common_years: list) -> pd.DataFrame:
    c_from, c_to = (min(common_years), max(common_years)) if len(common_years) >= 2 else (None, None)
    rows = []
    for label, g in rev.groupby("label", sort=False):
        g = g.sort_values("year")
        last = g.iloc[-1]
        cagr = calc_cagr(g)
        cagr_common = None
        if c_from is not None:
            cagr_common = calc_cagr(g[(g["year"] >= c_from) & (g["year"] <= c_to)])
        rows.append({
            "Компания": label,
            "Период": f"{int(g['year'].iloc[0])}–{int(last['year'])}",
            "Рост к пред. году": "н/д" if pd.isna(last["growth"]) else f"{signal_icon(last['growth'])} {last['growth']:+.1f}%",
            "CAGR (весь ряд)": "н/д" if cagr is None else f"{signal_icon(cagr)} {cagr:.1f}%",
            f"CAGR ({c_from}–{c_to})" if c_from is not None else "CAGR (общий период)":
                "н/д" if cagr_common is None else f"{signal_icon(cagr_common)} {cagr_common:.1f}%",
            "Есть оценки": "да" if g["is_estimate"].any() else "нет",
        })
    return pd.DataFrame(rows).set_index("Компания")


def collect_profit(ru_tickers, us_tickers, ru_all) -> pd.DataFrame:
    """Единая таблица прибыльности: label, year, net_income, ebitda, roe, net_margin, ebitda_margin."""
    parts = []
    for t in ru_tickers:
        w, unit, mixed = make_wide(ru_all[ru_all["ticker"] == t] if not ru_all.empty else ru_all)
        if not w.empty:
            parts.append(add_margins(w, enabled=not mixed).assign(label=t, unit=unit))
    for t in us_tickers:
        try:
            data = get_us_data(t)
        except Exception:
            continue  # ошибка уже показана на вкладках выручки
        if data is not None and not data["wide"].empty:
            parts.append(add_margins(data["wide"]).assign(label=t, unit="$"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_prices(ru_tickers, us_tickers, years, problems) -> pd.DataFrame:
    """Цены закрытия, приведенные к 100 на общую дату старта. Колонки: begin, label, close, index."""
    series = {}
    for t in ru_tickers:
        try:
            df = get_moex_candles(t, years)
        except requests.RequestException as e:
            problems.append(f"{t}: ошибка MOEX ({e})")
            continue
        if df.empty:
            problems.append(f"{t}: нет истории цен")
        else:
            series[t] = df[["begin", "close"]]
    for t in us_tickers:
        try:
            df = get_us_prices(t, years)
        except Exception as e:
            problems.append(f"{t}: ошибка загрузки цен ({e})")
            continue
        if df.empty:
            problems.append(f"{t}: нет истории цен")
        else:
            series[t] = df[["begin", "close"]]
    if len(series) == 0:
        return pd.DataFrame()
    start = max(df["begin"].min() for df in series.values())
    parts = []
    for label, df in series.items():
        df = df[df["begin"] >= start].sort_values("begin")
        if df.empty or df["close"].iloc[0] <= 0:
            continue
        parts.append(df.assign(label=label, index=df["close"] / df["close"].iloc[0] * 100))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ============ Курс валют и инфляция ============

# единицы в долларах -> (рублевая единица, множитель при умножении на курс)
USD_UNITS = {"млн $": ("млрд ₽", 1 / 1000), "млрд $": ("млрд ₽", 1.0), "$": ("₽", 1.0)}


@st.cache_data(show_spinner=False)
def _read_macro(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in MACRO_COLS if c not in df.columns]
    if missing:
        raise ValueError("в файле нет колонок: " + ", ".join(missing))
    df["year"] = df["year"].astype(int)
    df["indicator"] = df["indicator"].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def load_macro() -> dict:
    """{'usd_rub': {год: курс}, 'cpi': {год: инфляция дек/дек, %}}. Пустые словари, если файла нет."""
    out = {"usd_rub": {}, "cpi": {}}
    if not MACRO_PATH.exists():
        return out
    try:
        df = _read_macro(str(MACRO_PATH), MACRO_PATH.stat().st_mtime)
    except ValueError as e:
        st.error(f"Ошибка в файле {MACRO_PATH.name}: {e}")
        return out
    for key in out:
        sub = df[df["indicator"] == key]
        out[key] = dict(zip(sub["year"].astype(int), sub["value"].astype(float)))
    return out


def price_levels(cpi: dict) -> dict:
    """Средний уровень цен в году (середина года) относительно конца года перед первым годом ряда.
    Берется среднее геометрическое уровней на конец предыдущего и текущего года; ряд должен быть непрерывным."""
    levels, prev, last_year = {}, 1.0, None
    for y in sorted(cpi):
        if last_year is not None and y != last_year + 1:
            break  # разрыв в годах: дальше цепочку считать нельзя
        end = prev * (1 + cpi[y] / 100)
        levels[y] = (prev * end) ** 0.5
        prev, last_year = end, y
    return levels


def adjust_long(rows: pd.DataFrame, to_rub: bool, real: bool, base_year, macro: dict):
    """Пересчет денежных показателей: $ -> ₽ по среднему курсу года и/или приведение к ценам base_year.
    ROE не меняется. Возвращает (таблица, список замечаний)."""
    notes = []
    if rows.empty or not (to_rub or real):
        return rows, notes
    levels = price_levels(macro["cpi"]) if real else {}
    if real and base_year not in levels:
        notes.append("Нет данных об инфляции для выбранного базового года: поправка не применена.")
        real = False
    df = rows.copy()
    missing_fx, missing_cpi, skipped_real = set(), set(), set()
    values, units, extra = [], [], []
    nan = float("nan")
    for _, r in df.iterrows():
        v, u, how = r["value"], r["unit"], []
        if r["indicator"] in MONEY_INDICATORS:
            if to_rub and u in USD_UNITS:
                rate = macro["usd_rub"].get(int(r["year"]))
                if rate is None:
                    missing_fx.add(int(r["year"]))
                    v = nan
                else:
                    new_u, k = USD_UNITS[u]
                    v, u = v * rate * k, new_u
                    how.append(f"{rate:.2f} ₽ за $")
            if real:
                if "$" in u:
                    skipped_real.add(r["ticker"])
                elif int(r["year"]) in levels:
                    k = levels[base_year] / levels[int(r["year"])]
                    v *= k
                    how.append(f"в ценах {base_year} г. (×{k:.3f})")
                    u = f"{u} (цены {base_year} г.)"
                else:
                    missing_cpi.add(int(r["year"]))
                    v = nan
        values.append(v)
        units.append(u)
        extra.append("; пересчет: " + ", ".join(how) if how else "")
    df["value"], df["unit"] = values, units
    df["note"] = [(n + e).lstrip("; ") for n, e in zip(df["note"], extra)]
    df = df.dropna(subset=["value"])
    if missing_fx:
        notes.append("Нет курса ЦБ за годы: " + ", ".join(map(str, sorted(missing_fx))) + " — эти значения исключены.")
    if missing_cpi:
        notes.append("Нет данных об инфляции за годы: " + ", ".join(map(str, sorted(missing_cpi))) + " — эти значения исключены.")
    if skipped_real:
        notes.append("Для значений в долларах (" + ", ".join(sorted(skipped_real)) + ") поправка на российскую инфляцию "
                     "не применяется. Включите пересчет в рубли.")
    return df.reset_index(drop=True), notes


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
        st.warning(f"Файл {MACRO_PATH.name} не найден или пуст: пересчет недоступен.")
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
        st.dataframe(pd.DataFrame(tbl).set_index("Год"), use_container_width=True)
        st.caption("Курс — простое среднее официальных курсов ЦБ за год. Уровень цен в середине года — среднее геометрическое "
                   "уровней на конец предыдущего и текущего года (приближение для годовых потоков). "
                   "Данные лежат в macro_ru.csv, их можно обновлять без изменения кода.")


# ============ Контроль качества данных ============

SEV_ERR, SEV_WARN, SEV_INFO = "🔴 Ошибка", "🟡 Предупреждение", "🔵 Информация"
SEV_ORDER = {SEV_ERR: 0, SEV_WARN: 1, SEV_INFO: 2}
KNOWN_INDICATORS = ("revenue", "net_income", "ebitda", "roe", "dps", "shares")
KNOWN_UNITS = {"млрд ₽", "млн ₽", "млрд $", "млн $", "$", "₽", "%", "₽ на акцию", "млн шт."}
BOOL_VALUES = {"true", "false", "1", "0", "yes", "no", "да", "нет"}
JUMP_LIMITS = {"revenue": (100, SEV_WARN), "net_income": (300, SEV_INFO), "ebitda": (150, SEV_INFO)}
ISSUE_COLS = ["Серьезность", "Проверка", "Тикер", "Год", "Показатель", "Строка в файле", "Описание"]


def read_raw_csv() -> pd.DataFrame:
    """Сырой файл без преобразований: все значения как текст."""
    return pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def _issues_df(issues: list) -> pd.DataFrame:
    df = pd.DataFrame(issues, columns=ISSUE_COLS)
    if df.empty:
        return df
    df["_o"] = df["Серьезность"].map(SEV_ORDER)
    df = df.sort_values(["_o", "Тикер", "Проверка"], kind="stable").drop(columns="_o")
    return df.reset_index(drop=True)


def validate_financials(raw: pd.DataFrame, today_year: int):
    """Проверяет сырой CSV. Возвращает (таблица проблем, таблица корректных строк с типами)."""
    issues = []

    def add(sev, check, msg, ticker="", year="", indicator="", line=None):
        issues.append({"Серьезность": sev, "Проверка": check, "Тикер": ticker, "Год": year,
                       "Показатель": INDICATOR_NAMES.get(indicator, indicator), "Строка в файле": line,
                       "Описание": msg})

    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        add(SEV_ERR, "Структура файла", "Нет колонок: " + ", ".join(missing) + ". Приложение не сможет прочитать файл.")
        return _issues_df(issues), pd.DataFrame()
    if raw.empty:
        add(SEV_ERR, "Структура файла", "Файл не содержит данных.")
        return _issues_df(issues), pd.DataFrame()

    df = raw.copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df["line"] = df.index + 2  # строка 1 — заголовок
    df["ticker"] = df["ticker"].str.upper()
    df["indicator"] = df["indicator"].str.lower()
    year = pd.to_numeric(df["year"], errors="coerce")
    value = pd.to_numeric(df["value"], errors="coerce")

    bad = pd.Series(False, index=df.index)
    for i, r in df.iterrows():
        ln, t, ind = int(r["line"]), r["ticker"], r["indicator"]
        if not t:
            add(SEV_ERR, "Пустой тикер", "Строка без тикера будет некорректно обработана.", "", r["year"], ind, ln)
            bad[i] = True
            continue
        if pd.isna(year[i]) or year[i] != int(year[i]):
            add(SEV_ERR, "Некорректный год", f"«{r['year']}» не похоже на год. Строка игнорируется.", t, r["year"], ind, ln)
            bad[i] = True
            continue
        if year[i] < 2000 or year[i] > today_year:
            add(SEV_WARN, "Год вне диапазона", f"Год {int(year[i])} вне диапазона 2000–{today_year}.", t, int(year[i]), ind, ln)
        if ind not in KNOWN_INDICATORS:
            add(SEV_WARN, "Неизвестный показатель",
                f"«{ind}» не распознан (допустимо: {', '.join(KNOWN_INDICATORS)}). Строка игнорируется.",
                t, int(year[i]), ind, ln)
            bad[i] = True
            continue
        if pd.isna(value[i]):
            add(SEV_ERR, "Значение не число",
                f"«{r['value']}» нельзя прочитать как число; строка игнорируется. "
                "Используйте точку как десятичный разделитель и не используйте пробелы.", t, int(year[i]), ind, ln)
            bad[i] = True
            continue
        if r["is_estimate"].lower() not in BOOL_VALUES:
            add(SEV_WARN, "Флаг оценки", f"«{r['is_estimate']}» не похоже на true/false; считается как false.",
                t, int(year[i]), ind, ln)
        if not r["unit"]:
            add(SEV_WARN, "Нет единиц", "Не указаны единицы измерения.", t, int(year[i]), ind, ln)
        if not r["source"]:
            add(SEV_WARN, "Нет источника", "Не указан источник данных.", t, int(year[i]), ind, ln)

    ok = df[~bad].copy()
    if ok.empty:
        return _issues_df(issues), ok
    ok["year"] = year[~bad].astype(int)
    ok["value"] = value[~bad]
    ok["is_estimate"] = ok["is_estimate"].str.lower().isin(["true", "1", "yes", "да"])

    # Дубликаты
    dups = ok[ok.duplicated(["ticker", "year", "indicator"], keep=False)]
    for (t, y, ind), g in dups.groupby(["ticker", "year", "indicator"]):
        add(SEV_ERR, "Дубликат", f"Строки {', '.join(map(str, g['line']))} описывают одно и то же; "
            "используется последняя.", t, y, ind)
    ok = ok.drop_duplicates(["ticker", "year", "indicator"], keep="last")

    # Единицы
    for t, g in ok[ok["indicator"].isin(MONEY_INDICATORS)].groupby("ticker"):
        units = [u for u in g["unit"].unique() if u]
        if len(units) > 1:
            add(SEV_ERR, "Разные единицы", f"{', '.join(units)}: маржа и сравнение будут некорректны.", t)
    for _, r in ok[(ok["indicator"] == "roe") & (ok["unit"] != "%") & (ok["unit"] != "")].iterrows():
        add(SEV_WARN, "Единицы ROE", f"ROE должен быть в «%», указано «{r['unit']}».",
            r["ticker"], r["year"], "roe", r["line"])
    for u in sorted(set(ok["unit"]) - KNOWN_UNITS - {""}):
        tick = ", ".join(sorted(ok.loc[ok["unit"] == u, "ticker"].unique()))
        add(SEV_INFO, "Незнакомые единицы", f"«{u}» ({tick}): убедитесь, что это не опечатка.", tick)

    # Пропуски, устаревшие данные, покрытие
    for (t, ind), g in ok.groupby(["ticker", "indicator"]):
        ys = set(g["year"])
        gaps = [y for y in range(min(ys), max(ys) + 1) if y not in ys]
        if gaps:
            add(SEV_WARN, "Пропущенные годы", f"Нет данных за {', '.join(map(str, gaps))}.", t, "", ind)
    for t, g in ok.groupby("ticker"):
        last = int(g["year"].max())
        if last < today_year - 1:
            add(SEV_WARN, "Устаревшие данные", f"Последний год в данных — {last}, ожидается {today_year - 1}.", t, last)
        inds = set(g["indicator"])
        if "shares" not in inds:
            add(SEV_INFO, "Нет числа акций", "Без indicator = shares не считаются капитализация и мультипликаторы.", t)
        elif "dps" not in inds:
            add(SEV_INFO, "Нет дивидендов", "Нет строк indicator = dps: дивиденды и дивидендная доходность не показываются.", t)
        if "revenue" not in inds:
            add(SEV_WARN, "Нет выручки", "Без выручки не считаются маржа и рост.", t)
        else:
            if "net_income" not in inds:
                add(SEV_INFO, "Нет чистой прибыли", "Вкладка «Прибыльность» будет неполной.", t)
            n_rev = g.loc[g["indicator"] == "revenue", "year"].nunique()
            if n_rev < 3:
                add(SEV_INFO, "Короткий ряд", f"Выручка только за {n_rev} г.: CAGR и сравнение будут ограничены.", t)

    # Расчетные оценки
    for _, r in ok[ok["is_estimate"] & (ok["note"] == "")].iterrows():
        add(SEV_WARN, "Оценка без пояснения", "Расчетное значение без комментария: опишите в note, как оно получено.",
            r["ticker"], r["year"], r["indicator"], r["line"])

    # Здравый смысл
    for _, r in ok[(ok["indicator"] == "shares") & (ok["value"] <= 0)].iterrows():
        add(SEV_ERR, "Число акций ≤ 0", f"Значение {r['value']} выглядит ошибочным.", r["ticker"], r["year"], "shares", r["line"])
    for _, r in ok[(ok["indicator"] == "dps") & (ok["value"] < 0)].iterrows():
        add(SEV_ERR, "Дивиденд < 0", f"Значение {r['value']} выглядит ошибочным.", r["ticker"], r["year"], "dps", r["line"])
    for _, r in ok[(ok["indicator"] == "revenue") & (ok["value"] <= 0)].iterrows():
        add(SEV_ERR, "Выручка ≤ 0", f"Значение {r['value']} выглядит ошибочным.", r["ticker"], r["year"], "revenue", r["line"])
    for _, r in ok[(ok["indicator"] == "roe") & ((ok["value"] < -100) | (ok["value"] > 100))].iterrows():
        add(SEV_WARN, "ROE вне диапазона", f"ROE {r['value']}%: проверьте, что значение в процентах.",
            r["ticker"], r["year"], "roe", r["line"])
    wide = ok.pivot_table(index=["ticker", "year"], columns="indicator", values="value", aggfunc="last").reset_index()
    for c in KNOWN_INDICATORS:
        if c not in wide.columns:
            wide[c] = float("nan")
    for _, r in wide.iterrows():
        if pd.notna(r["revenue"]) and r["revenue"] > 0 and pd.notna(r["net_income"]) \
                and r["net_income"] / r["revenue"] * 100 > 100:
            add(SEV_WARN, "Чистая маржа > 100%", "Чистая прибыль больше выручки: проверьте значения и определения.",
                r["ticker"], r["year"], "net_income")
        if pd.notna(r["ebitda"]) and pd.notna(r["net_income"]) and r["ebitda"] < r["net_income"]:
            add(SEV_INFO, "EBITDA < чистой прибыли", "Обычно EBITDA больше чистой прибыли: проверьте определения.",
                r["ticker"], r["year"], "ebitda")

    # Резкие изменения год к году
    for (t, ind), g in ok.sort_values("year").groupby(["ticker", "indicator"]):
        if ind not in JUMP_LIMITS:
            continue
        limit, sev = JUMP_LIMITS[ind]
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            prev, cur = g.loc[i - 1], g.loc[i]
            if cur["year"] - prev["year"] == 1 and prev["value"] > 0 and cur["value"] > 0:
                ch = (cur["value"] / prev["value"] - 1) * 100
                if abs(ch) > limit:
                    add(sev, "Резкое изменение", f"{ch:+.0f}% к {int(prev['year'])} г.: проверьте значение, единицы "
                        "и сопоставимость определений.", t, int(cur["year"]), ind, cur["line"])

    return _issues_df(issues), ok


def coverage_matrix(ok: pd.DataFrame) -> pd.DataFrame:
    """Матрица покрытия: ✓ — есть значение, ≈ — оценка, пусто — нет данных."""
    if ok.empty:
        return pd.DataFrame()
    years = list(range(int(ok["year"].min()), int(ok["year"].max()) + 1))
    order = {k: i for i, k in enumerate(KNOWN_INDICATORS)}
    rows = []
    for (t, ind), g in ok.groupby(["ticker", "indicator"]):
        rec = {"Тикер": t, "Показатель": INDICATOR_NAMES[ind], "Единицы": g.sort_values("year")["unit"].iloc[-1],
               "_o": order[ind]}
        for y in years:
            sub = g[g["year"] == y]
            rec[str(y)] = "" if sub.empty else ("≈" if sub["is_estimate"].any() else "✓")
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["Тикер", "_o"]).drop(columns="_o").reset_index(drop=True)


def quality_view():
    st.write(f"Проверка файла **{CSV_PATH.name}**: структура, форматы, дубликаты, пропуски, единицы измерения, "
             "расчетные оценки и подозрительные значения. Файл перечитывается при каждом обновлении страницы, "
             "поэтому после правки CSV достаточно обновить страницу.")
    if not CSV_PATH.exists():
        st.error(f"Файл {CSV_PATH.name} не найден. Он должен лежать рядом с приложением: {CSV_PATH.parent}")
        return
    try:
        raw = read_raw_csv()
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")
        return

    issues, ok = validate_financials(raw, date.today().year)
    n = {sev: int((issues["Серьезность"] == sev).sum()) for sev in SEV_ORDER}

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Строк в файле", len(raw))
    m2.metric("Компаний", ok["ticker"].nunique() if not ok.empty else 0)
    m3.metric("Годы", f"{ok['year'].min()}–{ok['year'].max()}" if not ok.empty else "н/д")
    m4.metric("Расчетных оценок", int(ok["is_estimate"].sum()) if not ok.empty else 0)
    m5.metric("Ошибок / предупр.", f"{n[SEV_ERR]} / {n[SEV_WARN]}")
    st.caption("Файл изменен: " + datetime.fromtimestamp(CSV_PATH.stat().st_mtime).strftime("%d.%m.%Y %H:%M"))

    if n[SEV_ERR]:
        st.error(f"Найдено ошибок: {n[SEV_ERR]}. Часть строк приложение игнорирует или обрабатывает неверно. Исправьте файл.")
    elif n[SEV_WARN]:
        st.warning(f"Критичных ошибок нет, но есть предупреждения: {n[SEV_WARN]}. Стоит проверить.")
    else:
        st.success("Ошибок и предупреждений нет. Информационные заметки (если есть) — ниже.")

    coverage = coverage_matrix(ok)
    est_tbl = pd.DataFrame()
    if not ok.empty and ok["is_estimate"].any():
        e = ok[ok["is_estimate"]].sort_values(["ticker", "indicator", "year"])
        est_tbl = pd.DataFrame({
            "Тикер": e["ticker"], "Год": e["year"], "Показатель": e["indicator"].map(INDICATOR_NAMES),
            "Значение": e["value"], "Единицы": e["unit"], "Источник": e["source"], "Комментарий": e["note"]})

    tab_issues, tab_cov, tab_est, tab_raw = st.tabs(["Проблемы", "Покрытие данных", "Расчетные оценки", "Сырые данные"])
    with tab_issues:
        if issues.empty:
            st.info("Проблем не найдено.")
        else:
            chosen = st.multiselect("Показывать:", list(SEV_ORDER), default=list(SEV_ORDER))
            view = issues[issues["Серьезность"].isin(chosen)]
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.caption("Номер строки считается по файлу: строка 1 — заголовок. Информационные заметки не означают ошибку: "
                       "например, резкий рост прибыли может быть реальным, но его стоит сверить с источником.")
    with tab_cov:
        if coverage.empty:
            st.info("Нет корректных данных для матрицы покрытия.")
        else:
            st.dataframe(coverage, use_container_width=True, hide_index=True)
            st.caption("✓ — значение есть, ≈ — расчетная оценка, пусто — данных нет.")
    with tab_est:
        if est_tbl.empty:
            st.info("Расчетных оценок в файле нет.")
        else:
            st.dataframe(est_tbl, use_container_width=True, hide_index=True)
    with tab_raw:
        st.dataframe(raw, use_container_width=True, hide_index=True)

    with st.expander(f"🌐 Макроданные ({MACRO_PATH.name})"):
        macro = load_macro()
        if not (macro["usd_rub"] or macro["cpi"]):
            st.warning(f"Файл {MACRO_PATH.name} не найден или пуст: пересчет в рубли и поправка на инфляцию недоступны.")
        else:
            levels = price_levels(macro["cpi"])
            fx_years, cpi_years = sorted(macro["usd_rub"]), sorted(macro["cpi"])
            st.write(f"Курс ЦБ: {fx_years[0]}–{fx_years[-1]} ({len(fx_years)} г.). "
                     f"Инфляция: {cpi_years[0]}–{cpi_years[-1]} ({len(cpi_years)} г.).")
            if len(levels) < len(cpi_years):
                st.warning("В ряду инфляции есть пропущенные годы: цепочка прерывается, поправка доступна не для всех лет.")
            data_years = set(ok["year"]) if not ok.empty else set()
            lacking_fx = sorted(data_years - set(macro["usd_rub"]))
            lacking_cpi = sorted(data_years - set(levels))
            if lacking_fx:
                st.info("Курса ЦБ нет за годы из данных: " + ", ".join(map(str, lacking_fx)))
            if lacking_cpi:
                st.info("Инфляции нет за годы из данных: " + ", ".join(map(str, lacking_cpi)))

    st.divider()
    render_export({"Проблемы": issues, "Покрытие": coverage, "Расчетные оценки": est_tbl}, "data_quality")


# ============ UI ============

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

        ru_orig = ru_all[ru_all["ticker"] == ticker]
        ru_long = ru_orig
        market_tables_ = {}
        if not ru_long.empty:
            to_rub, real, base_year, macro = adjustment_controls(list(ru_long["unit"].unique()), f"single_{ticker}")
            ru_long, adj_notes = adjust_long(ru_long, to_rub, real, base_year, macro)
            render_adjustment_info(macro, to_rub, real, base_year, adj_notes)
        rows = ru_long[ru_long["indicator"] == "revenue"].rename(columns={"value": "revenue"}).drop(columns=["indicator"])

        tab_rev, tab_prof, tab_mkt, tab_price = st.tabs(["Выручка (МСФО)", "Прибыльность", "Мультипликаторы и дивиденды", "Цена акции"])

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

        with tab_prof:
            ru_rows = ru_long
            if ru_rows.empty:
                st.warning(f"Для **{ticker}** нет данных о прибыли в {CSV_PATH.name}. "
                           "Добавьте строки (indicator: net_income, ebitda, roe) — код менять не нужно.")
            else:
                w, unit_p, mixed = make_wide(ru_rows)
                render_profitability(w, unit_p, GREEN, ru_rows, mixed)

        with tab_mkt:
            if ru_orig.empty or not (ru_orig["indicator"] == "shares").any():
                st.warning(f"Для **{ticker}** в {CSV_PATH.name} нет числа акций (indicator = shares). "
                           "Добавьте строку — мультипликаторы и дивиденды появятся без изменения кода.")
            else:
                try:
                    price_m, pdate_m = last_close(get_moex_candles(ticker))
                except requests.RequestException as e:
                    price_m, pdate_m = None, None
                    st.error(f"Не удалось загрузить цену: {e}")
                stats_m, payout_m, notes_m = market_stats_ru(ru_orig, load_macro(), price_m, pdate_m)
                market_tables_ = render_market_stats(stats_m, payout_m, notes_m, GREEN)

        with tab_price:
            try:
                render_price(get_moex_candles(ticker), ticker, "₽", GREEN)
            except requests.RequestException as e:
                st.error(f"Не удалось загрузить историю цен: {e}")

        ru_export_rows = ru_long
        if not ru_export_rows.empty:
            st.divider()
            w_e, unit_e, _ = make_wide(ru_export_rows)
            render_export({**export_tables(w_e, unit_e, ru_export_rows, ticker), **market_tables_}, f"{ticker}_financials")
    else:
        try:
            with st.spinner(f"Загружаем данные для {ticker}..."):
                data = get_us_data(ticker)
        except Exception as e:
            st.error(f"Не удалось загрузить данные: {e}")
            st.stop()

        if data is None:
            st.error(f"❌ Тикер **{ticker}** не найден.")
            return
        st.subheader(f"🏢 {data['name']}")
        site = data["website"]
        st.write(f"**Сектор:** {data['sector']} | **Сайт:** {f'[{site}]({site})' if site else '—'}")
        market_tables_ = {}
        tab_rev, tab_prof, tab_mkt, tab_price = st.tabs(["Выручка", "Прибыльность", "Мультипликаторы и дивиденды", "Цена акции"])

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

        with tab_mkt:
            try:
                price_m, pdate_m = last_close(get_us_prices(ticker))
            except Exception as e:
                price_m, pdate_m = None, None
                st.error(f"Не удалось загрузить цену: {e}")
            stats_m, payout_m, notes_m = market_stats_us(data, price_m, pdate_m)
            market_tables_ = render_market_stats(stats_m, payout_m, notes_m, BLUE)

        with tab_price:
            try:
                render_price(get_us_prices(ticker), ticker, data["currency"], BLUE)
            except Exception as e:
                st.error(f"Не удалось загрузить историю цен: {e}")

        if not data["wide"].empty:
            st.divider()
            render_export({**export_tables(data["wide"], "$", None, ticker), **market_tables_}, f"{ticker}_financials")


def comparison_view():
    st.write(f"Выберите от 2 до {MAX_COMPANIES} компаний. Российские берутся из {CSV_PATH.name}, "
             "зарубежные вводятся тикерами.")
    revenue_all = load_revenue_csv()
    ru_all = load_ru_all()
    ru_options = sorted(revenue_all["ticker"].unique()) if not revenue_all.empty else []

    c1, c2 = st.columns(2)
    with c1:
        default = [t for t in ("SBER", "GAZP", "YDEX") if t in ru_options]
        ru_sel = st.multiselect("Российские компании:", ru_options, default=default)
    with c2:
        us_text = st.text_input("Зарубежные тикеры через запятую (например: AAPL, MSFT):", value="")
    us_sel = parse_tickers(us_text)

    total = len(ru_sel) + len(us_sel)
    if total < 2:
        st.info("Выберите минимум две компании.")
        return
    if total > MAX_COMPANIES:
        st.warning(f"Выбрано {total} компаний, максимум — {MAX_COMPANIES}. Уберите лишние.")
        return

    ru_sel_rows = ru_all[ru_all["ticker"].isin(ru_sel)] if not ru_all.empty else ru_all
    to_rub, real, base_year, macro = adjustment_controls(list(ru_sel_rows["unit"].unique()), "cmp")
    ru_all_adj, adj_notes = adjust_long(ru_sel_rows, to_rub, real, base_year, macro)
    revenue_all_adj = (ru_all_adj[ru_all_adj["indicator"] == "revenue"]
                       .rename(columns={"value": "revenue"}).drop(columns=["indicator"]))
    render_adjustment_info(macro, to_rub, real, base_year, adj_notes)
    if (to_rub or real) and us_sel:
        st.caption("ℹ️ Зарубежные компании не пересчитываются: их значения остаются в долларах (российская инфляция к ним не применима).")

    problems = []
    with st.spinner("Загружаем данные..."):
        rev = collect_revenue(ru_sel, us_sel, revenue_all_adj, problems)
    if rev.empty:
        st.error("Не удалось получить данные о выручке ни по одной компании.")
        for p in problems:
            st.write(f"- {p}")
        return

    labels = list(rev["label"].unique())
    year_sets = [set(g["year"]) for _, g in rev.groupby("label")]
    common = sorted(set.intersection(*year_sets))

    prof = collect_profit(ru_sel, us_sel, ru_all_adj)

    tab_idx, tab_growth, tab_abs, tab_prof, tab_mkt, tab_price = st.tabs(
        ["Выручка (индекс)", "Рост и CAGR", "Выручка в единицах", "Прибыльность", "Мультипликаторы", "Цена акции (индекс)"])

    with tab_idx:
        st.caption("Индекс выручки: значение в базовом году = 100. Так можно сравнивать компании "
                   "с разными масштабами и валютами.")
        if common:
            base = st.selectbox("Базовый год (= 100):", common, index=0)
            idx = index_revenue(rev, base)
        else:
            st.warning("Нет года, по которому есть данные у всех компаний: каждая индексируется по "
                       "своему первому году, сравнивать осторожно.")
            idx = index_revenue(rev, None)
        if idx.empty:
            st.info("Недостаточно данных для индекса.")
        else:
            fig = px.line(idx, x="year", y="index", color="label", markers=True,
                          title="Динамика выручки (базовый год = 100)")
            fig.add_hline(y=100, line_dash="dot", line_color="gray")
            fig.update_layout(xaxis=dict(dtick=1), xaxis_title="Год", yaxis_title="Индекс", legend_title="")
            st.plotly_chart(fig, use_container_width=True)

    with tab_growth:
        st.dataframe(summary_table(rev, common), use_container_width=True)
        st.caption(f"Светофор: 🟢 от +{GROWTH_UP:.0f}%, 🟡 от {GROWTH_DOWN:.0f}% до +{GROWTH_UP:.0f}%, "
                   f"🔴 ниже {GROWTH_DOWN:.0f}%. Рост номинальный, в валюте отчетности каждой компании. "
                   "CAGR по общему периоду сопоставим между компаниями, CAGR по всему ряду — нет, "
                   "если длина рядов разная.")
        gdf = rev.dropna(subset=["growth"])
        if gdf.empty:
            st.info("Для годового роста нужны данные минимум за два соседних года.")
        else:
            fig = px.bar(gdf.assign(Год=gdf["year"].astype(str)), x="Год", y="growth", color="label",
                         barmode="group", text_auto=".1f", title="Рост выручки к предыдущему году, %")
            fig.update_layout(xaxis_type="category", yaxis_title="%", legend_title="")
            st.plotly_chart(fig, use_container_width=True)

    with tab_abs:
        st.caption("Абсолютные значения сравнимы только внутри одной единицы измерения, "
                   "поэтому компании сгруппированы по единицам.")
        for unit, g in rev.groupby("unit"):
            fig = px.bar(g.assign(Год=g["year"].astype(str)), x="Год", y="revenue", color="label",
                         barmode="group", text_auto=".4s", title=f"Выручка, {unit}")
            fig.update_layout(xaxis_type="category", yaxis_title=unit, legend_title="")
            st.plotly_chart(fig, use_container_width=True)
        if rev["is_estimate"].any():
            est = rev[rev["is_estimate"]].groupby("label")["year"].apply(lambda s: ", ".join(map(str, s)))
            st.caption("⚠️ Расчетные оценки: " + "; ".join(f"{k} ({v})" for k, v in est.items()))

    with tab_prof:
        if prof.empty:
            st.info("Нет данных о прибыльности по выбранным компаниям.")
        else:
            summary = []
            for label, g in prof.groupby("label", sort=False):
                rec = {"Компания": label}
                for col, name in (("net_margin", "Чистая маржа"), ("ebitda_margin", "EBITDA-маржа"), ("roe", "ROE")):
                    lv = latest_value(g, col)
                    rec[name] = "н/д" if lv is None else f"{lv[1]:.1f}% ({lv[0]})"
                summary.append(rec)
            st.dataframe(pd.DataFrame(summary).set_index("Компания"), use_container_width=True)

            metric = st.radio("Показатель на графике:", ["Чистая маржа", "EBITDA-маржа", "ROE"], horizontal=True)
            col = {"Чистая маржа": "net_margin", "EBITDA-маржа": "ebitda_margin", "ROE": "roe"}[metric]
            chart = prof.dropna(subset=[col])
            if chart.empty:
                st.info(f"Нет данных по показателю «{metric}» у выбранных компаний.")
            else:
                fig = px.bar(chart.assign(Год=chart["year"].astype(str)), x="Год", y=col, color="label",
                             barmode="group", text_auto=".1f", title=f"{metric}, %")
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.update_layout(xaxis_type="category", yaxis_title="%", legend_title="")
                st.plotly_chart(fig, use_container_width=True)
            st.caption("В скобках — последний год с данными. Маржа = показатель / выручка; для банков (Сбербанк, "
                       "Т-Технологии) «выручка» — операционный доход, EBITDA не применяется, вместо нее ROE. "
                       "Для Яндекса прибыль и EBITDA скорректированные; для Лукойла прибыль от продолжающейся деятельности. "
                       "Сравнивать уровни маржи между банками и нефинансовыми компаниями некорректно.")

    with tab_mkt:
        st.caption("Текущие мультипликаторы: последняя цена × число акций / показатели последнего отчетного года. "
                   "Российские компании — по данным файла (число акций и дивиденды), зарубежные — по Yahoo Finance. "
                   "P/S и P/E безразмерны, поэтому сравнимы между рынками; капитализация указана в валюте листинга.")
        macro_m = load_macro()
        rows_m, notes_all = [], []
        with st.spinner("Загружаем цены и считаем мультипликаторы..."):
            for t in ru_sel:
                try:
                    price_c, pdate_c = last_close(get_moex_candles(t))
                except requests.RequestException as e:
                    price_c, pdate_c = None, None
                    notes_all.append(f"{t}: ошибка MOEX ({e})")
                stats_c, payout_c, n_c = market_stats_ru(ru_all[ru_all["ticker"] == t], macro_m, price_c, pdate_c)
                notes_all += [f"{t}: {x}" for x in n_c]
                if stats_c is not None:
                    rows_m.append(market_summary_row(t, stats_c, payout_c))
            for t in us_sel:
                try:
                    data_c = get_us_data(t)
                    price_c, pdate_c = last_close(get_us_prices(t))
                except Exception as e:
                    notes_all.append(f"{t}: ошибка загрузки ({e})")
                    continue
                if data_c is None:
                    continue
                stats_c, payout_c, n_c = market_stats_us(data_c, price_c, pdate_c)
                notes_all += [f"{t}: {x}" for x in n_c]
                rows_m.append(market_summary_row(t, stats_c, payout_c))
        if not rows_m:
            st.info("Нет данных для расчета мультипликаторов.")
        else:
            mdf = pd.DataFrame(rows_m).set_index("Компания")
            st.dataframe(mdf, use_container_width=True)
            metric_m = st.radio("Показатель на графике:", ["P/S", "P/E", "Дивдоходность, %", "Payout, %"], horizontal=True,
                                key="mkt_metric")
            cm = mdf[metric_m].dropna()
            if cm.empty:
                st.info(f"Нет данных по показателю «{metric_m}».")
            else:
                fig = px.bar(cm.reset_index(), x="Компания", y=metric_m, text_auto=".2f", title=metric_m)
                fig.update_layout(yaxis_title=metric_m)
                st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Скачать таблицу (CSV)", key="dl_cmp_mkt", mime="text/csv",
                               data=to_csv_bytes(mdf.reset_index()), file_name="comparison_multiples.csv")
        for n in dict.fromkeys(notes_all):
            st.caption("⚠️ " + n)

    with tab_price:
        years = st.selectbox("Период:", [1, 2, 3], index=0, format_func=lambda y: f"{y} г.")
        with st.spinner("Загружаем цены..."):
            prices = collect_prices(ru_sel, us_sel, years, problems)
        if prices.empty:
            st.info("История цен недоступна.")
        else:
            start_date = prices["begin"].min().date()
            fig = px.line(prices, x="begin", y="index", color="label",
                          title=f"Цена закрытия (старт с {start_date} = 100)")
            fig.add_hline(y=100, line_dash="dot", line_color="gray")
            fig.update_layout(xaxis_title="Дата", yaxis_title="Индекс", legend_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Метрики доходности и риска")
            pm = price_metrics_table(prices)
            if not pm.empty:
                st.dataframe(pm.set_index("Компания"), use_container_width=True)
                st.caption("Считаются по ценам за выбранный период (с общей даты старта). Волатильность — "
                           "годовая; просадка — от максимума внутри периода. MA200 на коротком периоде может быть недоступна (н/д).")
                st.download_button("⬇️ Скачать метрики (CSV)", key="dl_cmp_pm", mime="text/csv",
                                   data=to_csv_bytes(pm), file_name="comparison_price_metrics.csv")
            st.download_button("⬇️ Скачать индекс цен (CSV)", key="dl_cmp_prices", mime="text/csv",
                               data=to_csv_bytes(prices.rename(columns={"begin": "Дата", "label": "Компания",
                                                                        "close": "Цена закрытия", "index": "Индекс"})),
                               file_name="comparison_prices.csv")
            st.caption("Цены каждой бумаги в ее валюте (₽ или $), поэтому сравниваются изменения в процентах, "
                       "а не сами цены. Старт — общая дата, с которой есть данные по всем выбранным бумагам.")

    st.divider()
    render_export(comparison_export_tables(rev, common, prof), "comparison")

    if problems:
        with st.expander("⚠️ Что не удалось загрузить"):
            for p in dict.fromkeys(problems):
                st.write(f"- {p}")


st.title("📊 Анализатор продаж компаний (РФ и Мир)")
mode = st.radio("Режим:", ["Одна компания", "Сравнение компаний", "Контроль данных"], horizontal=True)

if mode != "Контроль данных" and CSV_PATH.exists():  # быстрая проверка файла данных на каждом экране
    try:
        _issues, _ = validate_financials(read_raw_csv(), date.today().year)
        _n_err = int((_issues["Серьезность"] == SEV_ERR).sum())
        if _n_err:
            st.warning(f"⚠️ В файле {CSV_PATH.name} найдено ошибок: {_n_err}. Откройте режим «Контроль данных».")
    except Exception:
        pass

if mode == "Сравнение компаний":
    comparison_view()
elif mode == "Контроль данных":
    quality_view()
else:
    single_view()
