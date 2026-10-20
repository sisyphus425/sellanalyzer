"""Настройки приложения: пути к файлам данных, пороги «светофора», цвета, справочники показателей и единиц."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # папка, где лежат streamlit_app.py и файлы данных

# --- Пути и общие настройки ---
ISS = "https://iss.moex.com/iss"
CACHE_TTL = 60 * 60
CSV_PATH = PROJECT_ROOT / "financials_ru.csv"
REQUIRED_COLS = ["ticker", "year", "indicator", "value", "unit", "metric", "is_estimate", "source", "note"]
MACRO_PATH = PROJECT_ROOT / "macro_ru.csv"
MACRO_COLS = ["year", "indicator", "value", "unit", "source", "note"]

# --- Цвета и «светофор» роста ---
GREEN, GREEN_LIGHT, BLUE = "#00A86B", "#9AD8BF", "#1f77b4"
GROWTH_UP, GROWTH_DOWN = 5.0, -5.0  # пороги "светофора", % (меняйте под себя)
SIGNAL_COLORS = {"Рост": "#00A86B", "Стагнация": "#F2B01E", "Падение": "#D64545"}

# --- Показатели, единицы и расчетные константы ---
INDICATOR_NAMES = {"revenue": "Выручка", "net_income": "Чистая прибыль", "ebitda": "EBITDA", "roe": "ROE",
                   "dps": "Дивиденд на акцию", "shares": "Число акций"}
MONEY_INDICATORS = ("revenue", "net_income", "ebitda")  # денежные показатели одной единицы измерения
# единицы в долларах -> (рублевая единица, множитель при умножении на курс)
USD_UNITS = {"млн $": ("млрд ₽", 1 / 1000), "млрд $": ("млрд ₽", 1.0), "$": ("₽", 1.0)}
UNIT_TO_BN = {"млрд ₽": 1.0, "млн ₽": 1 / 1000, "₽": 1e-9, "млрд $": 1.0, "млн $": 1 / 1000, "$": 1e-9}
PAYOUT_COLS = ["Год", "Дивиденд на акцию", "Дивиденды всего, млрд ≈", "Чистая прибыль, млрд", "Payout, %", "Оценка"]
TRADING_DAYS = 252  # торговых дней в году для годовой волатильности
MAX_COMPANIES = 6

# --- Контроль качества данных ---
SEV_ERR, SEV_WARN, SEV_INFO = "🔴 Ошибка", "🟡 Предупреждение", "🔵 Информация"
SEV_ORDER = {SEV_ERR: 0, SEV_WARN: 1, SEV_INFO: 2}
KNOWN_INDICATORS = ("revenue", "net_income", "ebitda", "roe", "dps", "shares")
KNOWN_UNITS = {"млрд ₽", "млн ₽", "млрд $", "млн $", "$", "₽", "%", "₽ на акцию", "млн шт."}
BOOL_VALUES = {"true", "false", "1", "0", "yes", "no", "да", "нет"}
JUMP_LIMITS = {"revenue": (100, SEV_WARN), "net_income": (300, SEV_INFO), "ebitda": (150, SEV_INFO)}
ISSUE_COLS = ["Серьезность", "Проверка", "Тикер", "Год", "Показатель", "Строка в файле", "Описание"]
