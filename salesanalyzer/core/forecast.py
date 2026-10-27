"""Простые прогнозы показателей на 1–2 года по малому числу годовых значений.

Принципы (данных мало, поэтому прогноз только простой и прозрачный):
- три несложных метода: линейный тренд, экстраполяция CAGR, затухающий рост;
- диапазон = разброс методов, расширенный на типичную ошибку методов на прошлом;
- проверка на прошлом: каждый метод «прогнозирует» уже известные годы по более ранним данным;
- оценка надежности с причинами; при меньше чем FORECAST_MIN_POINTS значений прогноз не строится.
"""

import numpy as np
import pandas as pd

from salesanalyzer.config import (FORECAST_DAMPING, FORECAST_ERROR_WARN, FORECAST_MIN_POINTS,
                                  FORECAST_VOLATILITY_WARN)

METHODS = ("Линейный тренд", "CAGR", "Затухающий рост")
BACKTEST_COLS = ["Год прогноза", "Данные до", "Метод", "Прогноз", "Факт", "Ошибка, %"]
MAX_BAND = 0.6  # доля от базового значения, больше которой расширение диапазона не растет


def clean_series(years, values):
    """Годы и значения без пропусков и дубликатов, по возрастанию лет."""
    s = pd.DataFrame({"year": list(years), "value": list(values)}).dropna()
    s = s.astype({"year": int, "value": float}).drop_duplicates("year", keep="last").sort_values("year")
    return s["year"].to_numpy(), s["value"].to_numpy()


def linear_trend(years, values, targets):
    """Прямая по методу наименьших квадратов (по годам); нужно минимум 2 значения."""
    if len(years) < 2:
        return None
    x = np.asarray(years, dtype=float) - years[-1]  # центрируем годы для устойчивости расчета
    slope, intercept = np.polyfit(x, np.asarray(values, dtype=float), 1)
    return [float(slope * (t - years[-1]) + intercept) for t in targets]


def cagr_rate(years, values):
    """Среднегодовой темп между первым и последним значением (доля) или None, если в ряду есть значения ≤ 0."""
    if len(years) < 2 or years[-1] == years[0] or (np.asarray(values) <= 0).any():
        return None  # темп роста имеет смысл только для ряда из положительных значений
    return float((values[-1] / values[0]) ** (1 / (years[-1] - years[0])) - 1)


def cagr_extrapolation(years, values, targets):
    """Продолжение среднегодового темпа CAGR от последнего значения."""
    g = cagr_rate(years, values)
    if g is None:
        return None
    return [float(values[-1] * (1 + g) ** (t - years[-1])) for t in targets]


def mean_growth(years, values):
    """Среднее арифметическое годовых темпов по соседним годам с положительными значениями (доля) или None."""
    rates = [values[i] / values[i - 1] - 1 for i in range(1, len(values))
             if years[i] - years[i - 1] == 1 and values[i - 1] > 0 and values[i] > 0]
    return float(np.mean(rates)) if rates else None


def damped_growth(years, values, targets, damping=None):
    """Средний темп роста затухает с каждым годом: в первом году g·φ, во втором g·φ² и т.д. (φ = damping)."""
    phi = FORECAST_DAMPING if damping is None else damping
    g = mean_growth(years, values)
    if g is None or (np.asarray(values) <= 0).any():
        return None
    levels, level, rate = {}, float(values[-1]), g
    for year in range(int(years[-1]) + 1, int(max(targets)) + 1):
        rate *= phi
        level *= 1 + rate
        levels[year] = level
    return [levels[t] for t in targets]


def run_methods(years, values, targets, damping=None):
    """Результаты всех методов: {название: [значение на каждый год] или None, если метод неприменим}."""
    return {"Линейный тренд": linear_trend(years, values, targets),
            "CAGR": cagr_extrapolation(years, values, targets),
            "Затухающий рост": damped_growth(years, values, targets, damping)}


def holdout(years, values, min_train=2, damping=None):
    """Проверка на прошлом: для каждого известного года (начиная с min_train-го) прогноз по более ранним данным."""
    rows = []
    for i in range(min_train, len(years)):
        results = run_methods(years[:i], values[:i], [int(years[i])], damping)
        for name, res in results.items():
            if res is None:
                continue
            actual = float(values[i])
            rows.append({"Год прогноза": int(years[i]), "Данные до": int(years[i - 1]), "Метод": name,
                         "Прогноз": res[0], "Факт": actual,
                         "Ошибка, %": (res[0] / actual - 1) * 100 if actual != 0 else float("nan")})
    return pd.DataFrame(rows, columns=BACKTEST_COLS)


def backtest_summary(bt):
    """По каждому методу: число проверок, средняя абсолютная ошибка и ошибка последней проверки, %."""
    if bt.empty:
        return pd.DataFrame(columns=["Метод", "Проверок", "Средняя ошибка, %", "Ошибка на последнем году, %"])
    rows = []
    for name in METHODS:
        sub = bt[bt["Метод"] == name].dropna(subset=["Ошибка, %"]).sort_values("Год прогноза")
        if sub.empty:
            continue
        rows.append({"Метод": name, "Проверок": len(sub), "Средняя ошибка, %": float(sub["Ошибка, %"].abs().mean()),
                     "Ошибка на последнем году, %": float(sub["Ошибка, %"].iloc[-1])})
    return pd.DataFrame(rows)


def typical_error(bt):
    """Типичная ошибка методов на прошлом (средняя абсолютная, %) или None."""
    if bt.empty:
        return None
    err = bt["Ошибка, %"].abs().dropna()
    return float(err.mean()) if len(err) else None


def growth_spread(years, values):
    """Стандартное отклонение годовых темпов роста, п.п. (None, если темпов меньше двух)."""
    rates = [(values[i] / values[i - 1] - 1) * 100 for i in range(1, len(values))
             if years[i] - years[i - 1] == 1 and values[i - 1] > 0 and values[i] > 0]
    return float(np.std(rates, ddof=1)) if len(rates) >= 2 else None


def assess_reliability(years, values, bt, estimate_years=()):
    """Оценка надежности прогноза: {'label': ..., 'reasons': [...]}. Выше «средней» не бывает: данных слишком мало."""
    n = len(years)
    if n < FORECAST_MIN_POINTS:
        return {"label": "недостаточно данных",
                "reasons": [f"в ряду {n} знач.; для прогноза нужно минимум {FORECAST_MIN_POINTS} годовых значений"]}
    label, reasons = "средняя", [f"прогноз строится по {n} годовым значениям, поэтому он очень приблизительный"]
    if n == FORECAST_MIN_POINTS:
        label = "низкая"
        reasons.append("минимальное число значений: тренд определяется почти случайно")
    err = typical_error(bt)
    if err is not None and err > FORECAST_ERROR_WARN:
        label = "низкая"
        reasons.append(f"на прошлом методы ошибались в среднем на {err:.0f}% (порог {FORECAST_ERROR_WARN:.0f}%)")
    spread = growth_spread(years, values)
    if spread is not None and spread > FORECAST_VOLATILITY_WARN:
        label = "низкая"
        reasons.append(f"темпы роста по годам сильно различаются (разброс {spread:.0f} п.п.)")
    if (np.asarray(values) <= 0).any():
        label = "низкая"
        reasons.append("в ряду есть нулевые или отрицательные значения: часть методов неприменима")
    if int(years[-1]) in set(int(y) for y in estimate_years):
        label = "низкая"
        reasons.append("последнее значение расчётное (оценка), а прогноз строится от него")
    return {"label": label, "reasons": reasons}


def scenarios(methods, targets, typical_err, nonnegative=False):
    """Низкий, базовый и высокий сценарии по каждому году: база = медиана методов, границы = разброс методов,
    расширенный на типичную ошибку на прошлом (растет с горизонтом как √k)."""
    rows = []
    for k, year in enumerate(targets, start=1):
        vals = [res[k - 1] for res in methods.values() if res is not None]
        if not vals:
            continue
        base = float(np.median(vals))
        width = min((typical_err or 0.0) / 100 * np.sqrt(k), MAX_BAND) * abs(base)
        low, high = min(min(vals), base - width), max(max(vals), base + width)
        if nonnegative:
            low = max(low, 0.0)
        rows.append({"Год": int(year), "Низкий": float(low), "База": base, "Высокий": float(high), "Методов": len(vals)})
    return pd.DataFrame(rows, columns=["Год", "Низкий", "База", "Высокий", "Методов"])


def forecast_series(years, values, horizon=1, estimate_years=(), nonnegative=False, damping=None):
    """Полный прогноз ряда. Возвращает словарь:
    years, values — очищенный ряд; targets — прогнозные годы; methods — результаты методов; scenarios — таблица
    низкий/база/высокий; backtest, backtest_summary — проверка на прошлом; reliability — оценка надежности."""
    yrs, vals = clean_series(years, values)
    result = {"years": yrs, "values": vals, "targets": [], "methods": {}, "scenarios": scenarios({}, [], None),
              "backtest": pd.DataFrame(columns=BACKTEST_COLS), "backtest_summary": backtest_summary(pd.DataFrame(columns=BACKTEST_COLS)),
              "reliability": assess_reliability(yrs, vals, pd.DataFrame(columns=BACKTEST_COLS), estimate_years)}
    if len(yrs) < FORECAST_MIN_POINTS:
        return result
    targets = [int(yrs[-1]) + k for k in range(1, int(horizon) + 1)]
    methods = run_methods(yrs, vals, targets, damping)
    bt = holdout(yrs, vals, damping=damping)
    result.update(targets=targets, methods=methods, backtest=bt, backtest_summary=backtest_summary(bt),
                  scenarios=scenarios(methods, targets, typical_error(bt), nonnegative),
                  reliability=assess_reliability(yrs, vals, bt, estimate_years))
    return result


def methods_table(result):
    """Таблица результатов методов по годам с их средней ошибкой на прошлом."""
    if not result["targets"]:
        return pd.DataFrame()
    summary = result["backtest_summary"].set_index("Метод") if not result["backtest_summary"].empty else None
    rows = []
    for name in METHODS:
        res = result["methods"].get(name)
        row = {"Метод": name}
        for i, year in enumerate(result["targets"]):
            row[str(year)] = None if res is None else res[i]
        row["Средняя ошибка на прошлом, %"] = (None if summary is None or name not in summary.index
                                               else float(summary.loc[name, "Средняя ошибка, %"]))
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_overview(series_by_label, horizon=1, nonnegative=False):
    """Сводка для нескольких компаний: {название: (годы, значения, годы-оценки)} -> таблица по первому прогнозному году."""
    rows = []
    for label, (years, values, estimate_years) in series_by_label.items():
        res = forecast_series(years, values, horizon, estimate_years, nonnegative)
        rel = res["reliability"]["label"]
        row = {"Компания": label, "Последний год": int(res["years"][-1]) if len(res["years"]) else None,
               "Последнее значение": float(res["values"][-1]) if len(res["values"]) else None, "Надежность": rel}
        if res["scenarios"].empty:
            rows.append({**row, "Год прогноза": None, "Низкий": None, "База": None, "Высокий": None,
                         "Рост базы к последнему году, %": None, "Типичная ошибка на прошлом, %": None})
            continue
        last = res["scenarios"].iloc[-1]  # самый дальний год горизонта
        last_val = float(res["values"][-1])
        err = typical_error(res["backtest"])
        rows.append({**row, "Год прогноза": int(last["Год"]), "Низкий": float(last["Низкий"]), "База": float(last["База"]),
                     "Высокий": float(last["Высокий"]),
                     "Рост базы к последнему году, %": ((last["База"] / last_val - 1) * 100) if last_val > 0 else None,
                     "Типичная ошибка на прошлом, %": err})
    return pd.DataFrame(rows)
