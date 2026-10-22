"""Мосбиржа (MOEX ISS API): название бумаги и история цен."""

from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

from salesanalyzer import config
from salesanalyzer.config import CACHE_TTL, PRICE_HISTORY_YEARS
from salesanalyzer.data_sources.errors import DataSourceError
from salesanalyzer.data_sources.retry import call_with_retries


class _TransientHTTP(Exception):
    """Временная ошибка сервера (например, 503): запрос стоит повторить."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def _is_transient(error) -> bool:
    return isinstance(error, (requests.ConnectionError, requests.Timeout, _TransientHTTP))


def _describe(error) -> str:
    """Понятное пользователю описание ошибки обращения к Мосбирже."""
    if isinstance(error, requests.Timeout):
        return f"Мосбиржа не ответила за {config.REQUEST_TIMEOUT} с. Повторите попытку позже."
    if isinstance(error, requests.ConnectionError):
        return "Нет соединения с Мосбиржей. Проверьте интернет-подключение."
    if isinstance(error, _TransientHTTP):
        return f"Мосбиржа временно недоступна (HTTP {error.status}). Повторите попытку позже."
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        return f"Мосбиржа отклонила запрос (HTTP {status})."
    if isinstance(error, (ValueError, TypeError, KeyError, AttributeError)):
        return "Мосбиржа вернула неожиданный ответ: не удалось разобрать данные."
    return f"Ошибка при обращении к Мосбирже: {error}"


def iss_get(path: str, **params) -> dict:
    """Запрос к ISS с повторными попытками. Возвращает словарь блоков-таблиц.
    При постоянном сбое бросает DataSourceError с понятным текстом."""
    url = f"{config.ISS}/{path}.json"

    def once():
        r = requests.get(url, params={"iss.meta": "off", **params}, timeout=config.REQUEST_TIMEOUT)
        if r.status_code in config.RETRY_STATUSES:
            raise _TransientHTTP(r.status_code)
        r.raise_for_status()
        return r.json()

    try:
        payload = call_with_retries(once, is_transient=_is_transient)
        return {k: pd.DataFrame(v["data"], columns=v["columns"])
                for k, v in payload.items() if isinstance(v, dict) and "columns" in v}
    except Exception as error:
        raise DataSourceError(_describe(error), "MOEX") from error


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_moex_name(ticker: str):
    """Полное название бумаги на Мосбирже или None, если тикер неизвестен."""
    desc = iss_get(f"securities/{ticker}", **{"iss.only": "description"}).get("description")
    if desc is None or desc.empty:
        return None
    return dict(zip(desc["name"], desc["value"])).get("NAME", ticker)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_moex_candles(ticker: str, years: int = PRICE_HISTORY_YEARS) -> pd.DataFrame:
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    frames, cursor = [], 0
    while True:  # ISS отдает ограниченное число свечей за запрос, поэтому грузим страницами
        df = iss_get(f"engines/stock/markets/shares/securities/{ticker}/candles",
                     interval=24, **{"from": start, "start": cursor}).get("candles")
        if df is None or df.empty:
            break
        frames.append(df)
        cursor += len(df)
        if len(df) < config.ISS_PAGE_SIZE:
            break
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["begin"] = pd.to_datetime(out["begin"])
    return out
