"""Подготовка таблиц и файлов для выгрузки (CSV, Excel)."""

from datetime import date
from io import BytesIO

import pandas as pd

from salesanalyzer.config import INDICATOR_NAMES
from salesanalyzer.core.comparison import summary_numeric
from salesanalyzer.core.growth import add_growth, calc_cagr, signal_name
from salesanalyzer.core.profitability import add_margins


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
