"""Общие настройки тестов.

Тесты проверяют расчетную часть приложения (всё в app_final.py до блока интерфейса «# ============ UI ============»).
Интерфейс и сетевые запросы не запускаются: streamlit подменяется заглушкой, а данные берутся из файлов
financials_ru.csv и macro_ru.csv, лежащих рядом с app_final.py.
"""
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
APP_NAME = "app_final.py"
UI_MARKER = "# ============ UI ============"
_SKIP_DIRS = {"tests", "venv", "env", "site-packages", "node_modules", "__pycache__"}


def _candidate_bases():
    bases = []
    for base in (HERE, HERE.parent, Path.cwd()):
        if base not in bases:
            bases.append(base)
    return bases


def find_app_file():
    """Ищет app_final.py. Порядок: переменная APP_FINAL_PATH, папка с этим файлом, ее родитель,
    текущая папка, затем их подпапки (на один уровень). Возвращает путь или None."""
    env = os.environ.get("APP_FINAL_PATH")
    if env:
        path = Path(env)
        return path.resolve() if path.is_file() else None
    bases = _candidate_bases()
    for base in bases:
        if (base / APP_NAME).is_file():
            return (base / APP_NAME).resolve()
    for base in bases:
        try:
            for sub in sorted(base.iterdir()):
                if (sub.is_dir() and not sub.name.startswith(".") and sub.name not in _SKIP_DIRS
                        and (sub / APP_NAME).is_file()):
                    return (sub / APP_NAME).resolve()
        except OSError:
            continue
    return None


APP_FILE = find_app_file()


def pytest_report_header(config):
    return f"Тестируемый файл: {APP_FILE}"


def pytest_sessionstart(session):
    """Если app_final.py не найден, останавливаем запуск с понятным сообщением (вместо сотен одинаковых ошибок)."""
    if APP_FILE is None:
        searched = "\n".join(f"  - {b}" for b in _candidate_bases())
        pytest.exit(
            f"Не найден файл {APP_NAME}.\nЯ искал в папках (и в их подпапках):\n{searched}\n"
            f"Положите {APP_NAME} рядом с тестами или задайте путь к нему в переменной окружения APP_FINAL_PATH, например:\n"
            f'  PowerShell:  $env:APP_FINAL_PATH = "C:\\путь\\к\\app_final.py"\n'
            f"  bash:        export APP_FINAL_PATH=/путь/к/app_final.py",
            returncode=2)


class FakeStreamlit(types.ModuleType):
    """Заглушка streamlit: кэш-декораторы пропускают функцию как есть, остальные вызовы записываются в messages."""

    def __init__(self):
        super().__init__("streamlit")
        self.messages = []

    def cache_data(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def recorder(*args, **kwargs):
            self.messages.append((name, args[:1]))

        return recorder


@pytest.fixture(scope="session")
def fake_st():
    return FakeStreamlit()


@pytest.fixture(scope="session")
def app(fake_st):
    """Модуль приложения без блока интерфейса: все расчетные функции доступны как app.имя_функции."""
    sys.modules["streamlit"] = fake_st
    for name in ("yfinance", "plotly", "plotly.express", "requests"):
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
    source = APP_FILE.read_text(encoding="utf-8")
    code = source[: source.index(UI_MARKER)]
    module = types.ModuleType("app_final_under_test")
    module.__file__ = str(APP_FILE)
    exec(compile(code, str(APP_FILE), "exec"), module.__dict__)
    return module


@pytest.fixture(scope="session")
def ru_all(app):
    """Все показатели РФ из financials_ru.csv (длинный формат)."""
    return app.load_ru_all()


@pytest.fixture(scope="session")
def macro(app):
    """Курс ЦБ и инфляция из macro_ru.csv."""
    return app.load_macro()


@pytest.fixture(scope="session")
def raw_csv(app):
    return app.read_raw_csv()


@pytest.fixture
def mk_rows():
    """Фабрика длинных таблиц показателей: mk_rows(("SBER", 2024, "revenue", 100.0, "млрд ₽"), ...)."""

    def build(*items, estimate=False, note="", metric="тест", source="тест"):
        records = []
        for it in items:
            ticker, year, indicator, value, unit = it[:5]
            est = it[5] if len(it) > 5 else estimate
            records.append({"ticker": ticker, "year": year, "indicator": indicator, "value": value, "unit": unit,
                            "metric": metric, "is_estimate": est, "source": source, "note": note})
        return pd.DataFrame(records)

    return build


@pytest.fixture
def mk_raw():
    """Фабрика «сырых» таблиц (всё как текст), как при чтении CSV без преобразований."""
    columns = ["ticker", "year", "indicator", "value", "unit", "metric", "is_estimate", "source", "note"]

    def build(*rows):
        data = [[str(x) for x in row] for row in rows]
        return pd.DataFrame(data, columns=columns)

    return build
