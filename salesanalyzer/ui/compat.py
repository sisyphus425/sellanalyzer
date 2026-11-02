"""Совместимость с разными версиями Streamlit."""

import inspect
from functools import lru_cache

import streamlit as st


@lru_cache(maxsize=None)
def _stretch_items(func):
    try:
        param = inspect.signature(func).parameters.get("width")
    except (TypeError, ValueError):
        param = None
    if param is not None and param.default == "stretch":
        return (("width", "stretch"),)  # новые версии: параметр use_container_width объявлен устаревшим
    return (("use_container_width", True),)  # старые версии


def stretch(func) -> dict:
    """Аргументы растягивания элемента на всю ширину для данной функции Streamlit (под установленную версию)."""
    return dict(_stretch_items(func))


def show_chart(fig):
    """График Plotly на всю ширину."""
    return st.plotly_chart(fig, **stretch(st.plotly_chart))


def show_table(data, **kwargs):
    """Таблица на всю ширину."""
    return st.dataframe(data, **stretch(st.dataframe), **kwargs)


def rerun():
    """Перезапуск скрипта (в старых версиях Streamlit метод назывался experimental_rerun)."""
    return (getattr(st, "rerun", None) or st.experimental_rerun)()
