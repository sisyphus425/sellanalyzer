"""Общие настройки тестов.

Тесты проверяют расчетную часть приложения (пакет salesanalyzer: core, data_sources, services) и сами файлы данных
financials_ru.csv и macro_ru.csv. Интерфейс (salesanalyzer/ui), браузер и сетевые запросы не запускаются:
streamlit подменяется заглушкой, yfinance, requests и plotly при отсутствии тоже.
"""
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
PACKAGE = "salesanalyzer"
_SKIP_DIRS = {"tests", "venv", "env", "site-packages", "node_modules", "__pycache__"}


def _candidate_bases():
    bases = []
    for base in (HERE, HERE.parent, Path.cwd()):
        if base not in bases:
            bases.append(base)
    return bases


def find_project_root():
    """Ищет папку, в которой лежит пакет salesanalyzer. Порядок: переменная SALESANALYZER_ROOT, папка с этим файлом,
    ее родитель, текущая папка, затем их подпапки (на один уровень). Возвращает путь или None."""
    env = os.environ.get("SALESANALYZER_ROOT")
    if env:
        root = Path(env)
        return root.resolve() if (root / PACKAGE / "__init__.py").is_file() else None
    bases = _candidate_bases()
    for base in bases:
        if (base / PACKAGE / "__init__.py").is_file():
            return base.resolve()
    for base in bases:
        try:
            for sub in sorted(base.iterdir()):
                if (sub.is_dir() and not sub.name.startswith(".") and sub.name not in _SKIP_DIRS
                        and (sub / PACKAGE / "__init__.py").is_file()):
                    return sub.resolve()
        except OSError:
            continue
    return None


PROJECT_ROOT = find_project_root()
if PROJECT_ROOT is not None and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_report_header(config):
    return f"Корень проекта (там, где пакет {PACKAGE}): {PROJECT_ROOT}"


def pytest_sessionstart(session):
    """Если пакет не найден, останавливаем запуск с понятным сообщением (вместо сотен одинаковых ошибок)."""
    if PROJECT_ROOT is None:
        searched = "\n".join(f"  - {b}" for b in _candidate_bases())
        pytest.exit(
            f"Не найден пакет {PACKAGE} (папка {PACKAGE}/ с файлом __init__.py).\nЯ искал в папках (и в их подпапках):\n{searched}\n"
            f"Запускайте pytest из папки проекта или задайте путь к ней в переменной окружения SALESANALYZER_ROOT, например:\n"
            f'  PowerShell:  $env:SALESANALYZER_ROOT = "C:\\путь\\к\\проекту"\n'
            f"  bash:        export SALESANALYZER_ROOT=/путь/к/проекту",
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


# модули, чьи функции тесты вызывают как app.имя_функции
APP_MODULES = [
    "config", "data_sources.csv_store", "data_sources.moex", "data_sources.yahoo",
    "core.formatting", "core.growth", "core.profitability", "core.adjustments", "core.prices", "core.market",
    "core.comparison", "core.validation", "core.exports", "services",
    "data_sources.errors", "data_sources.retry", "ui.compat", "core.forecast",
]


@pytest.fixture(scope="session")
def fake_st():
    return FakeStreamlit()


@pytest.fixture(scope="session")
def cfg(fake_st):
    """Модуль настроек (пути к файлам данных и константы). Через него тесты подменяют пути."""
    _install_stubs(fake_st)
    import importlib
    return importlib.import_module(f"{PACKAGE}.config")


def _install_stubs(fake_st):
    sys.modules["streamlit"] = fake_st
    for name in ("yfinance", "plotly", "plotly.express", "requests"):
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)


@pytest.fixture(scope="session")
def app(fake_st):
    """Единый доступ ко всем расчетным функциям: app.add_growth, app.market_stats_ru и т.д.
    Собирается из модулей пакета; функции берутся из тех модулей, где они определены."""
    import importlib
    _install_stubs(fake_st)
    facade = types.SimpleNamespace()
    for mod_name in APP_MODULES:
        module = importlib.import_module(f"{PACKAGE}.{mod_name}")
        for name, obj in vars(module).items():
            if name.startswith("__") or isinstance(obj, types.ModuleType):
                continue
            if getattr(obj, "__module__", module.__name__) == module.__name__ or not callable(obj):
                setattr(facade, name, obj)
    return facade


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
