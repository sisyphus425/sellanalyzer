"""Данные пользователя: своя компания из файла CSV или Excel.

Формат тот же, что у financials_ru.csv (колонки ticker, year, indicator, value, unit, metric, is_estimate, source, note).
Файл проверяется так же, как в режиме «Контроль данных»; корректные строки хранятся только в сессии браузера
(пока открыта вкладка) и на сервер не записываются. Если тикер совпадает с компанией из financials_ru.csv,
загруженные данные заменяют данные файла для этого тикера.
"""

import io
from datetime import date

import pandas as pd
import streamlit as st

from salesanalyzer.config import REQUIRED_COLS, SEV_ERR
from salesanalyzer.core.validation import validate_financials

SESSION_ROWS, SESSION_NAME = "user_financials_rows", "user_financials_name"
MAX_UPLOAD_ROWS = 5000


def _strip_year(value: str) -> str:
    """«2025.0» из Excel -> «2025»."""
    v = str(value).strip()
    return v[:-2] if v.endswith(".0") and v[:-2].isdigit() else v


def read_upload(data: bytes, filename: str) -> pd.DataFrame:
    """Файл пользователя -> таблица строк (все значения как текст), как read_raw_csv.
    CSV: кодировка UTF-8 (в том числе с BOM) или Windows-1251, разделитель запятая или точка с запятой
    (при точке с запятой десятичная запятая в value заменяется точкой). Excel (.xlsx): первый лист.
    Бросает ValueError с понятным текстом, если файл не прочитать."""
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(io.BytesIO(data), dtype=str, keep_default_na=False)
        else:
            text = None
            for enc in ("utf-8-sig", "cp1251"):
                try:
                    text = data.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                raise ValueError("не удалось определить кодировку (нужна UTF-8 или Windows-1251)")
            header = text.splitlines()[0] if text else ""
            sep = ";" if header.count(";") > header.count(",") else ","
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
            if sep == ";" and "value" in df.columns:
                df["value"] = df["value"].str.replace(" ", "").str.replace(",", ".")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"файл не читается как таблица ({type(e).__name__}: {e})") from e
    df.columns = [str(c).strip().lower() for c in df.columns]
    if len(df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"слишком много строк ({len(df)}), максимум {MAX_UPLOAD_ROWS}")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("нет колонок: " + ", ".join(missing) + ". Скачайте шаблон и заполните его.")
    df = df[REQUIRED_COLS].astype(str).apply(lambda s: s.str.strip())
    df["year"] = df["year"].map(_strip_year)
    return df[(df != "").any(axis=1)].reset_index(drop=True)


def prepare_rows(raw: pd.DataFrame, today_year=None):
    """Проверка строк пользователя. Возвращает (строки в формате load_ru_all, таблица проблем, число ошибок).
    В строки попадает только то, что приложение может корректно прочитать."""
    today_year = date.today().year if today_year is None else today_year
    issues, ok = validate_financials(raw, today_year)
    n_err = int((issues["Серьезность"] == SEV_ERR).sum()) if not issues.empty else 0
    if ok.empty:
        return pd.DataFrame(columns=REQUIRED_COLS), issues, n_err
    rows = ok[REQUIRED_COLS].copy()
    for col in ("unit", "metric", "source", "note"):
        rows[col] = rows[col].fillna("").astype(str)
    rows["value"] = rows["value"].astype(float)
    return rows.reset_index(drop=True), issues, n_err


def template_frame() -> pd.DataFrame:
    """Пример заполнения для своей компании (тикер MYCO — условный; для компании с акциями на Мосбирже укажите ее тикер)."""
    src = "Собственные данные (управленческая отчетность)"
    rows = []
    for year, rev, ni, ebitda in ((2023, 120.0, 9.5, 18.0), (2024, 138.0, 11.2, 21.5), (2025, 151.0, 12.8, 24.0)):
        rows += [("MYCO", year, "revenue", rev, "млн ₽", "Выручка", "false", src, ""),
                 ("MYCO", year, "net_income", ni, "млн ₽", "Чистая прибыль", "false", src, ""),
                 ("MYCO", year, "ebitda", ebitda, "млн ₽", "EBITDA", "false", src, "")]
    rows.append(("MYCO", 2026, "revenue", 165.0, "млн ₽", "Выручка", "true", src, "Оценка по бюджету на 2026 г."))
    rows.append(("MYCO", 2025, "shares", 10.0, "млн шт.", "Число акций", "false", src,
                 "Необязательно: нужно для мультипликаторов, если у компании есть акции на бирже"))
    return pd.DataFrame(rows, columns=REQUIRED_COLS)


TEMPLATE_HELP = pd.DataFrame({
    "Колонка": REQUIRED_COLS,
    "Что писать": [
        "Тикер Мосбиржи, если акции торгуются (тогда подтянутся цены), или любое короткое имя латиницей для частной компании",
        "Год (2023, 2024, …)",
        "revenue — выручка, net_income — чистая прибыль, ebitda — EBITDA, roe — ROE (%), dps — дивиденд на акцию, "
        "shares — число акций (млн шт.), payout_target — целевой payout (%)",
        "Число с точкой, без пробелов: 1234.5",
        "Одинаковые для денежных показателей: млрд ₽, млн ₽, млрд $ или млн $; для roe и payout_target — %, для dps — ₽ на акцию, для shares — млн шт.",
        "Что именно считается показателем (например, «Выручка по РСБУ»)",
        "true для оценки (бюджет, прогноз), false для факта",
        "Откуда цифры",
        "Комментарий (обязателен для оценок)",
    ]})


def _session():
    state = getattr(st, "session_state", None)
    return state if hasattr(state, "get") and hasattr(state, "__setitem__") else None


def user_rows() -> pd.DataFrame:
    """Загруженные пользователем строки (пустая таблица, если ничего не загружено)."""
    state = _session()
    rows = state.get(SESSION_ROWS) if state is not None else None
    return rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(columns=REQUIRED_COLS)


def user_tickers() -> list:
    rows = user_rows()
    return sorted(rows["ticker"].unique()) if not rows.empty else []


def user_file_name():
    state = _session()
    return state.get(SESSION_NAME) if state is not None else None


def set_user_rows(rows: pd.DataFrame, filename: str):
    state = _session()
    if state is not None:
        state[SESSION_ROWS] = rows
        state[SESSION_NAME] = filename


def clear_user_rows():
    state = _session()
    if state is not None:
        for key in (SESSION_ROWS, SESSION_NAME):
            if key in state:
                del state[key]


def merge_with_file(file_rows: pd.DataFrame, uploaded: pd.DataFrame) -> pd.DataFrame:
    """Строки файла + загруженные: для тикеров из загрузки данные файла заменяются загруженными."""
    if uploaded is None or uploaded.empty:
        return file_rows
    if file_rows is None or file_rows.empty:
        return uploaded.reset_index(drop=True)
    keep = file_rows[~file_rows["ticker"].isin(set(uploaded["ticker"]))]
    return pd.concat([keep, uploaded[keep.columns.intersection(uploaded.columns)]], ignore_index=True)
