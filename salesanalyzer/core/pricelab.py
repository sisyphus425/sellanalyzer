"""Исследовательская лаборатория: простые модели цены (дрейф, авторегрессия, ARIMA(1,1,1), гребневая регрессия) против наивного прогноза.

Наивный прогноз: «цена через h дней равна сегодняшней» (ожидаемая лог-доходность 0). Модель имеет смысл, только если она ощутимо
и устойчиво точнее наивного на проверке «шаг за шагом» по прошлому. Каждая модель на каждой дате проверки обучается ТОЛЬКО на данных
до этой даты. Даты проверки идут с шагом в h дней, чтобы ошибки не перекрывались. Только numpy, без дополнительных библиотек.
"""

import numpy as np
import pandas as pd

from salesanalyzer.config import (LAB_AR_LAGS, LAB_ARMA_LONG_LAGS, LAB_MAX_ORIGINS, LAB_MIN_ORIGINS, LAB_MIN_TRAIN, LAB_RIDGE_ALPHA,
                                  LAB_SIGNIFICANCE)

NAIVE, DRIFT, AR, ARIMA, RIDGE = "Наивный (цена не меняется)", "Средний дрейф", "Авторегрессия AR(5)", "ARIMA(1,1,1)", "Гребневая регрессия (ML)"
MODELS = (NAIVE, DRIFT, AR, ARIMA, RIDGE)
VERDICT_BETTER, VERDICT_WORSE, VERDICT_SAME, VERDICT_FEW, VERDICT_BASE = (
    "лучше наивного", "хуже наивного", "не отличается от наивного", "мало проверок", "—")
SUMMARY_COLS = ["Модель", "Горизонт, дней", "Проверок", "RMSE доходности, %", "RMSE к наивному", "Средняя ошибка, п.п.",
                "Знак угадан, %", "Доля роста в выборке, %", "t-статистика", "Вывод"]
ORIGIN_COLS = ["Дата", "Горизонт, дней", "Модель", "Прогноз доходности, %", "Факт доходности, %"]
MIN_PRICES = LAB_MIN_TRAIN + 80


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _lag_matrix(r, p):
    """Строки [1, r[t-1], ..., r[t-p]] и цели r[t] для t = p..len-1."""
    n = len(r)
    X = np.column_stack([np.ones(n - p)] + [r[p - k:n - k] for k in range(1, p + 1)])
    return X, r[p:]


def forecast_drift(lp, h):
    r = np.diff(lp)
    return float(r.mean() * h) if len(r) else 0.0


def forecast_ar(lp, h, p=None):
    """AR(p) по дневным доходностям, прогноз на h шагов суммируется."""
    p = LAB_AR_LAGS if p is None else p
    r = np.diff(lp)
    if len(r) < p * 5:
        return 0.0
    X, y = _lag_matrix(r, p)
    beta = _ols(X, y)
    buf, total = list(r[-p:][::-1]), 0.0           # buf[0] — последняя доходность
    for _ in range(h):
        nxt = float(beta[0] + np.dot(beta[1:], buf[:p]))
        total += nxt
        buf.insert(0, nxt)
    return total


def forecast_arima(lp, h, m=None):
    """ARIMA(1,1,1) (ARMA(1,1) по доходностям), оценка по Ханнану–Риссанену: остатки берутся из длинной авторегрессии."""
    m = LAB_ARMA_LONG_LAGS if m is None else m
    r = np.diff(lp)
    if len(r) < m * 5:
        return 0.0
    X, y = _lag_matrix(r, m)
    e = np.concatenate([np.zeros(m), y - X @ _ols(X, y)])      # остатки (первые m нулевые)
    Z = np.column_stack([np.ones(len(r) - m - 1), r[m:-1], e[m:-1]])
    c, phi, theta = _ols(Z, r[m + 1:])
    prev = c + phi * r[-1] + theta * e[-1]
    total = prev
    for _ in range(h - 1):
        prev = c + phi * prev
        total += prev
    return float(total)


def _features(lp, i):
    r_last = lp[i] - lp[i - 1]
    window = np.diff(lp[i - 21:i + 1])
    return np.array([r_last, lp[i] - lp[i - 5], lp[i] - lp[i - 21], window.std(), lp[i] - lp[i - 49:i + 1].mean()])


def feature_matrix(lp):
    """Признаки для каждой даты i >= 49 (строки до этого заполнены NaN). Признак на дату i использует только цены до i включительно."""
    F = np.full((len(lp), 5), np.nan)
    for i in range(49, len(lp)):
        F[i] = _features(lp, i)
    return F


def forecast_ridge(lp, h, alpha=None, feats=None):
    """Гребневая регрессия: будущая h-дневная доходность по пяти признакам (импульс, волатильность, отклонение от средней).
    feats: заранее посчитанная feature_matrix (для ускорения проверки на прошлом; результат тот же)."""
    alpha = LAB_RIDGE_ALPHA if alpha is None else alpha
    T = len(lp) - 1
    F = feature_matrix(lp) if feats is None else feats[:T + 1]
    idx = np.arange(49, T - h + 1)
    if len(idx) < 60:
        return 0.0
    X, y = F[idx], lp[idx + h] - lp[idx]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    ym = y.mean()
    beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ (y - ym))
    return float(ym + ((F[T] - mu) / sd) @ beta)


FORECASTERS = {NAIVE: lambda lp, h: 0.0, DRIFT: forecast_drift, AR: forecast_ar, ARIMA: forecast_arima, RIDGE: forecast_ridge}


def walk_forward(prices, h, models=MODELS, min_train=None, max_origins=None):
    """Проверка на прошлом: на каждой дате проверки модель видит только прошлое и прогнозирует лог-доходность за h дней.
    Возвращает таблицу ORIGIN_COLS (доходности в %)."""
    min_train = LAB_MIN_TRAIN if min_train is None else min_train
    max_origins = LAB_MAX_ORIGINS if max_origins is None else max_origins
    df = prices.reset_index(drop=True)
    lp = np.log(df["close"].to_numpy(dtype=float))
    first, last = min_train, len(lp) - 1 - h
    if last < first:
        return pd.DataFrame(columns=ORIGIN_COLS)
    origins = list(range(last, first - 1, -h))[:max_origins][::-1]      # от самой свежей даты назад
    feats = feature_matrix(lp) if RIDGE in models else None
    rows = []
    for T in origins:
        train, actual = lp[:T + 1], (lp[T + h] - lp[T]) * 100
        for name in models:
            pred = forecast_ridge(train, h, feats=feats) if name == RIDGE else FORECASTERS[name](train, h)
            rows.append({"Дата": df["begin"].iloc[T], "Горизонт, дней": h, "Модель": name,
                         "Прогноз доходности, %": pred * 100, "Факт доходности, %": actual})
    return pd.DataFrame(rows, columns=ORIGIN_COLS)


def verdict(ratio, t, n):
    if n < LAB_MIN_ORIGINS:
        return VERDICT_FEW
    if ratio < 1 and t >= LAB_SIGNIFICANCE:
        return VERDICT_BETTER
    if ratio > 1 and t <= -LAB_SIGNIFICANCE:
        return VERDICT_WORSE
    return VERDICT_SAME


def summarize(origins, h):
    """Сводка по моделям: точность против наивного, угадывание знака, t-статистика разницы квадратов ошибок."""
    sub = origins[origins["Горизонт, дней"] == h]
    if sub.empty:
        return pd.DataFrame(columns=SUMMARY_COLS)
    naive = sub[sub["Модель"] == NAIVE].set_index("Дата")
    out = []
    for name in [m for m in MODELS if m in set(sub["Модель"])]:
        s = sub[sub["Модель"] == name].set_index("Дата")
        err = s["Прогноз доходности, %"] - s["Факт доходности, %"]
        nerr = naive["Прогноз доходности, %"] - naive["Факт доходности, %"]
        n = len(s)
        rmse, rmse_n = float(np.sqrt((err ** 2).mean())), float(np.sqrt((nerr ** 2).mean()))
        ratio = rmse / rmse_n if rmse_n > 0 else np.nan
        diff = (nerr.reindex(err.index) ** 2 - err ** 2)         # >0: модель точнее наивного
        sd = diff.std(ddof=1) if n > 1 else 0.0
        t = float(diff.mean() / (sd / np.sqrt(n))) if sd and sd > 0 else 0.0
        actual = s["Факт доходности, %"]
        mask = actual != 0
        hit = float((np.sign(s["Прогноз доходности, %"][mask]) == np.sign(actual[mask])).mean() * 100) if (mask.any() and name != NAIVE) else np.nan
        mae = float(err.abs().mean())
        out.append({"Модель": name, "Горизонт, дней": h, "Проверок": n, "RMSE доходности, %": rmse,
                    "RMSE к наивному": ratio, "Средняя ошибка, п.п.": mae, "Знак угадан, %": hit,
                    "Доля роста в выборке, %": float((actual > 0).mean() * 100), "t-статистика": t,
                    "Вывод": VERDICT_BASE if name == NAIVE else verdict(ratio, t, n)})
    return pd.DataFrame(out, columns=SUMMARY_COLS)


def overall_conclusion(summary):
    """Честный итог по горизонту: есть ли модель, устойчиво лучшая наивного."""
    if summary.empty:
        return "Недостаточно данных для проверки."
    best = summary[summary["Вывод"] == VERDICT_BETTER]
    if summary["Проверок"].max() < LAB_MIN_ORIGINS:
        return "Слишком мало проверок на прошлом: выводов о моделях не делается."
    if best.empty:
        return "Ни одна модель не оказалась заметно точнее наивного прогноза «цена не изменится». Прогнозировать цену этими моделями нельзя."
    names = ", ".join(best["Модель"])
    return (f"Заметно точнее наивного оказалась: {names}. Это результат на конкретном отрезке истории и одной бумаге: "
            "при проверке многих моделей какая-то выигрывает и случайно, поэтому не считайте это прогнозом.")


def latest_forecasts(prices, h, models=MODELS):
    """Что каждая модель сегодня «говорит» про цену через h дней (для справки): {модель: (доходность %, цена)}."""
    df = prices.reset_index(drop=True)
    lp = np.log(df["close"].to_numpy(dtype=float))
    p0 = float(df["close"].iloc[-1])
    return {m: (FORECASTERS[m](lp, h) * 100, p0 * float(np.exp(FORECASTERS[m](lp, h)))) for m in models}
