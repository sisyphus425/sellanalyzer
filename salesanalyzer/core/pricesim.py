"""Диапазон будущей цены: моделирование Монте-Карло и проверка калибровки на прошлом.

Это НЕ прогноз цены, а набор сценариев: «какие цены были бы возможны при историческом разбросе доходностей».
Модели (на выбор):
- GBM: нормальные дневные доходности с заданной волатильностью (свежей EWMA или за весь период);
- бутстрап: дневные доходности случайно выбираются из истории (сохраняются «тяжелые хвосты»).
Дрейф (ожидаемый рост) по умолчанию нулевой: ожидаемая цена равна текущей, прошлый рост не продолжается.
Проверка на прошлом показывает, как часто фактическая цена попадала в построенные коридоры.

Дивиденды. Если переданы прошлые выплаты, доходности считаются по ряду полной доходности: дивидендный гэп в день отсечки
не считается падением цены и не завышает волатильность. Ожидаемые выплаты внутри горизонта вычитаются из цены в день отсечки
(после гэпа траектория продолжается от уменьшенной цены). Проверка на прошлом при этом идет по полной доходности.
"""

from statistics import NormalDist

import numpy as np
import pandas as pd

from salesanalyzer.core.benchmark import add_total_return

from salesanalyzer.config import (MC_CALIBRATION_MIN_TRAIN, MC_CALIBRATION_SIMULATIONS, MC_CALIBRATION_STEP,
                                  MC_EWMA_LAMBDA, MC_MIN_RETURNS, MC_SEED, MC_SIMULATIONS, TRADING_DAYS)

MODEL_GBM, MODEL_BOOTSTRAP = "gbm", "bootstrap"
VOL_EWMA, VOL_FULL = "ewma", "full"
DRIFT_ZERO, DRIFT_HIST = "zero", "hist"
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QUANTILE_NAMES = {q: f"p{int(round(q * 100))}" for q in QUANTILES}
CALIBRATION_COLS = ["Дата прогноза", "Цена на дату", "Факт через горизонт", "p10", "p25", "p50", "p75", "p90",
                    "В 80%-коридоре", "В 50%-коридоре"]


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Цены без пропусков и неположительных значений, по возрастанию даты (колонки begin, close)."""
    df = prices[["begin", "close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[df["close"] > 0].dropna().sort_values("begin").reset_index(drop=True)


def log_returns(close) -> np.ndarray:
    """Дневные логарифмические доходности ряда цен."""
    return np.diff(np.log(np.asarray(close, dtype=float)))


def ewma_variance(returns, lam=None) -> float:
    """Экспоненциально взвешенная дисперсия (RiskMetrics): свежие дни весят больше."""
    lam = MC_EWMA_LAMBDA if lam is None else lam
    r = np.asarray(returns, dtype=float)
    w = lam ** np.arange(len(r))[::-1]
    w = w / w.sum()
    return float(np.sum(w * r ** 2))


def estimate_params(returns, vol_mode=VOL_EWMA, drift_mode=DRIFT_ZERO) -> dict:
    """Дневные волатильность sigma и логарифмический дрейф mu.
    Нулевой дрейф: mu = −σ²/2, тогда ожидаемая цена равна текущей. Исторический: средняя дневная доходность."""
    r = np.asarray(returns, dtype=float)
    sigma = float(np.sqrt(ewma_variance(r))) if vol_mode == VOL_EWMA else float(np.std(r, ddof=1))
    mu = float(-0.5 * sigma ** 2) if drift_mode == DRIFT_ZERO else float(np.mean(r))
    return {"sigma": sigma, "mu": mu, "sigma_annual": sigma * float(np.sqrt(TRADING_DAYS))}


def _steps(returns, n_sim, horizon, model, vol_mode, drift_mode, rng):
    """Матрица дневных логарифмических приращений (n_sim × horizon)."""
    r = np.asarray(returns, dtype=float)
    if model == MODEL_GBM:
        p = estimate_params(r, vol_mode, drift_mode)
        return p["mu"] + p["sigma"] * rng.standard_normal((n_sim, horizon))
    sample = r[rng.integers(0, len(r), size=(n_sim, horizon))]
    if drift_mode == DRIFT_ZERO:  # центрируем так, чтобы ожидаемая цена равнялась текущей
        sample = sample - r.mean() - 0.5 * r.var(ddof=1)
    return sample


def risk_metrics(p0, terminal, path_min) -> dict:
    """Вероятности и потери на конец горизонта по смоделированным траекториям."""
    terminal, path_min = np.asarray(terminal), np.asarray(path_min)
    q05 = float(np.quantile(terminal, 0.05))
    tail = terminal[terminal <= q05]
    return {
        "median": float(np.median(terminal)), "p10": float(np.quantile(terminal, 0.10)), "p90": float(np.quantile(terminal, 0.90)),
        "prob_below_current": float(np.mean(terminal < p0)),
        "prob_end_drop_10": float(np.mean(terminal <= 0.9 * p0)), "prob_end_drop_20": float(np.mean(terminal <= 0.8 * p0)),
        "prob_touch_drop_10": float(np.mean(path_min <= 0.9 * p0)), "prob_touch_drop_20": float(np.mean(path_min <= 0.8 * p0)),
        "var95_pct": float((p0 - q05) / p0 * 100), "cvar95_pct": float((p0 - tail.mean()) / p0 * 100),
    }


DIVIDEND_COLS = ["ex_date", "dps"]
MIN_PRICE_AFTER_GAP = 0.01  # доля цены, ниже которой гэп не опускает траекторию (дивиденд больше цены на практике не бывает)


def model_series(df: pd.DataFrame, dividends=None) -> np.ndarray:
    """Ряд, по которому оцениваются доходности: цены или, если есть прошлые выплаты, индекс полной доходности
    (дивидендный гэп компенсирован дивидендом)."""
    if dividends is None or len(dividends) == 0:
        return df["close"].to_numpy(dtype=float)
    return add_total_return(df, dividends)["tr"].to_numpy(dtype=float)


def project_dividends(events, last_date, horizon_end, lookback_days: int = 365) -> pd.DataFrame:
    """Ожидаемые выплаты в (last_date, horizon_end]: объявленные (с датой отсечки в будущем) и повтор выплат последних
    lookback_days дней через год с той же суммой. Повтор пропускается, если в пределах 60 дней от него уже есть объявленная выплата.
    Возвращает таблицу ex_date, dps, source («объявлена» или «как год назад»)."""
    cols = DIVIDEND_COLS + ["source"]
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=cols)
    ev = events[DIVIDEND_COLS].dropna().copy()
    ev["ex_date"] = pd.to_datetime(ev["ex_date"]).dt.normalize()
    ev = ev[ev["dps"] > 0]
    last_date, horizon_end = pd.Timestamp(last_date).normalize(), pd.Timestamp(horizon_end).normalize()
    announced = ev[(ev["ex_date"] > last_date) & (ev["ex_date"] <= horizon_end)].assign(source="объявлена")
    recent = ev[(ev["ex_date"] > last_date - pd.Timedelta(days=lookback_days)) & (ev["ex_date"] <= last_date)]
    repeated = recent.assign(ex_date=recent["ex_date"] + pd.DateOffset(years=1), source="как год назад")
    repeated = repeated[(repeated["ex_date"] > last_date) & (repeated["ex_date"] <= horizon_end)]
    if not announced.empty and not repeated.empty:
        near = repeated["ex_date"].apply(lambda d: (abs(announced["ex_date"] - d) <= pd.Timedelta(days=60)).any())
        repeated = repeated[~near]
    out = pd.concat([announced, repeated], ignore_index=True)
    return out.sort_values("ex_date").reset_index(drop=True)[cols]


def apply_dividend_gaps(paths: np.ndarray, dates, future_dividends) -> tuple:
    """Вычитает дивиденды из траекторий в первый день горизонта не раньше даты отсечки: цена в этот день уменьшается
    на сумму дивиденда, дальше траектория продолжается от уменьшенной цены. Возвращает (траектории, таблица учтенных выплат
    с колонками ex_date, dps, day — номер дня горизонта, начиная с 1)."""
    booked = pd.DataFrame(columns=DIVIDEND_COLS + ["day"])
    if future_dividends is None or len(future_dividends) == 0:
        return paths, booked
    days = pd.DatetimeIndex(dates).normalize()
    rows = []
    paths = paths.copy()
    ev = future_dividends.copy()
    ev["ex_date"] = pd.to_datetime(ev["ex_date"]).dt.normalize()
    for _, r in ev.sort_values("ex_date").iterrows():
        k = int(np.searchsorted(days.values, np.datetime64(r["ex_date"]), side="left"))
        if k >= paths.shape[1] or r["dps"] <= 0:
            continue
        factor = np.maximum(1 - float(r["dps"]) / paths[:, k], MIN_PRICE_AFTER_GAP)
        paths[:, k:] *= factor[:, None]
        rows.append({"ex_date": r["ex_date"], "dps": float(r["dps"]), "day": k + 1})
    if rows:
        booked = pd.DataFrame(rows, columns=DIVIDEND_COLS + ["day"])
    return paths, booked


def price_range(prices, horizon_days, model=MODEL_GBM, vol_mode=VOL_EWMA, drift_mode=DRIFT_ZERO, n_sim=None, seed=None,
                dividends=None, future_dividends=None):
    """Диапазон цены на horizon_days торговых дней вперед. Возвращает словарь или None, если данных мало:
    p0, last_date, params, dates, quantiles (по дням), terminal (цены на конец горизонта), metrics, dividends (учтенные выплаты),
    dividends_sum. dividends: прошлые выплаты (ex_date, dps) — доходности оцениваются без дивидендных гэпов;
    future_dividends: ожидаемые выплаты внутри горизонта — вычитаются из цены в день отсечки."""
    df = clean_prices(prices)
    series = model_series(df, dividends) if len(df) > 1 else np.array([])
    returns = log_returns(series) if len(series) > 1 else np.array([])
    if len(returns) < MC_MIN_RETURNS:
        return None
    n_sim = MC_SIMULATIONS if n_sim is None else n_sim
    rng = np.random.default_rng(MC_SEED if seed is None else seed)
    p0, last_date = float(df["close"].iloc[-1]), df["begin"].iloc[-1]
    steps = _steps(returns, n_sim, int(horizon_days), model, vol_mode, drift_mode, rng)
    paths = p0 * np.exp(np.cumsum(steps, axis=1))
    dates = pd.bdate_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=int(horizon_days))
    paths, booked = apply_dividend_gaps(paths, dates, future_dividends)
    q = np.quantile(paths, QUANTILES, axis=0)
    quantiles = pd.DataFrame({QUANTILE_NAMES[qq]: q[i] for i, qq in enumerate(QUANTILES)})
    quantiles.insert(0, "День", np.arange(1, int(horizon_days) + 1))
    terminal = paths[:, -1]
    div_sum = float(booked["dps"].sum()) if not booked.empty else 0.0
    metrics = risk_metrics(p0, terminal, paths.min(axis=1))
    metrics["median_with_dividends"] = float(np.median(terminal + div_sum))
    metrics["prob_loss_with_dividends"] = float(np.mean(terminal + div_sum < p0))
    return {"p0": p0, "last_date": last_date, "params": estimate_params(returns, vol_mode, drift_mode), "dates": dates,
            "quantiles": quantiles, "terminal": terminal, "metrics": metrics,
            "horizon_days": int(horizon_days), "n_returns": int(len(returns)), "dividends": booked, "dividends_sum": div_sum,
            "gaps_removed": dividends is not None and len(dividends) > 0}


def _terminal_quantiles(train_returns, horizon, model, vol_mode, drift_mode, n_sim, rng, qs):
    """Квантили логарифма отношения цены к текущей через horizon дней: аналитически для GBM, моделированием для бутстрапа."""
    if model == MODEL_GBM:
        p = estimate_params(train_returns, vol_mode, drift_mode)
        nd = NormalDist()
        return {q: p["mu"] * horizon + p["sigma"] * np.sqrt(horizon) * nd.inv_cdf(q) for q in qs}
    r = np.asarray(train_returns, dtype=float)
    total = r[rng.integers(0, len(r), size=(n_sim, horizon))].sum(axis=1)
    if drift_mode == DRIFT_ZERO:
        total = total - horizon * (r.mean() + 0.5 * r.var(ddof=1))
    return {q: float(np.quantile(total, q)) for q in qs}


def calibrate(prices, horizon_days, model=MODEL_GBM, vol_mode=VOL_EWMA, drift_mode=DRIFT_ZERO, step=None, min_train=None,
              n_sim=None, seed=None, dividends=None):
    """Проверка на прошлом: на каждой дате модель обучается только на более ранних данных и строит коридоры на horizon_days вперед;
    затем сверяем с фактической ценой. Возвращает (таблица по датам, сводка). Окна перекрываются, поэтому независимых проверок меньше (n_eff).
    С прошлыми выплатами (dividends) проверка идет по индексу полной доходности, приведенному к цене на дату прогноза:
    так дивидендный гэп внутри окна не считается промахом модели."""
    step = MC_CALIBRATION_STEP if step is None else step
    min_train = MC_CALIBRATION_MIN_TRAIN if min_train is None else min_train
    n_sim = MC_CALIBRATION_SIMULATIONS if n_sim is None else n_sim
    horizon = int(horizon_days)
    df = clean_prices(prices)
    close = model_series(df, dividends) if len(df) > 1 else df["close"].to_numpy()
    returns = log_returns(close) if len(close) > 1 else np.array([])
    rng = np.random.default_rng(MC_SEED if seed is None else seed)
    qs = (0.10, 0.25, 0.50, 0.75, 0.90)
    rows = []
    for i in range(max(min_train, MC_MIN_RETURNS), len(close) - horizon, step):
        # на цене с индексом i известны доходности returns[:i]
        tq = _terminal_quantiles(returns[:i], horizon, model, vol_mode, drift_mode, n_sim, rng, qs)
        band = {q: close[i] * float(np.exp(v)) for q, v in tq.items()}
        actual = float(close[i + horizon])
        rows.append({"Дата прогноза": df["begin"].iloc[i], "Цена на дату": float(close[i]), "Факт через горизонт": actual,
                     "p10": band[0.10], "p25": band[0.25], "p50": band[0.50], "p75": band[0.75], "p90": band[0.90],
                     "В 80%-коридоре": band[0.10] <= actual <= band[0.90], "В 50%-коридоре": band[0.25] <= actual <= band[0.75]})
    table = pd.DataFrame(rows, columns=CALIBRATION_COLS)
    if table.empty:
        return table, {"n": 0, "n_eff": 0.0, "coverage80": None, "coverage50": None, "below_median": None, "width80_pct": None,
                       "margin80": None, "margin50": None}
    n = int(len(table))
    n_eff = max(1.0, n * step / horizon)  # окна перекрываются: независимых проверок примерно в horizon/step раз меньше
    width = ((table["p90"] - table["p10"]) / table["Цена на дату"] * 100).mean()
    return table, {"n": n, "n_eff": float(n_eff), "coverage80": float(table["В 80%-коридоре"].mean()),
                   "coverage50": float(table["В 50%-коридоре"].mean()),
                   "below_median": float((table["Факт через горизонт"] < table["p50"]).mean()), "width80_pct": float(width),
                   "margin80": float(2 * np.sqrt(0.8 * 0.2 / n_eff)), "margin50": float(2 * np.sqrt(0.5 * 0.5 / n_eff))}


MIN_EFFECTIVE_CHECKS = 5  # меньше независимых проверок: делать вывод о качестве модели нельзя


def calibration_verdict(summary) -> str:
    """Вывод о калибровке с учетом случайного разброса (около 2 стандартных ошибок при числе независимых проверок n_eff):
    'мало проверок', 'отклоняется' (покрытие вне случайного разброса) или 'в пределах разброса'."""
    if not summary["n"] or summary.get("n_eff", 0) < MIN_EFFECTIVE_CHECKS:
        return "мало проверок"
    n_eff = summary["n_eff"]
    margin80, margin50 = 2 * np.sqrt(0.8 * 0.2 / n_eff), 2 * np.sqrt(0.5 * 0.5 / n_eff)
    if abs(summary["coverage80"] - 0.80) > margin80 or abs(summary["coverage50"] - 0.50) > margin50:
        return "отклоняется"
    return "в пределах разброса"
