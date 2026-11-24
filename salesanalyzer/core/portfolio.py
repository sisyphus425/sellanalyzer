"""Портфель: стоимость в рублях, доходность (по цене и с дивидендами), риск, сравнение с индексом, вклады бумаг, дивидендный поток.

Состав задается количеством бумаг (или долями и суммой: доли переводятся в количество по текущим ценам, можно дробное).
История — «если бы этот состав держали весь период»: стоимость_t = Σ количество_i × цена_i,t в рублях.
Цены и дивиденды в валюте пересчитываются в рубли по курсу на ту же дату (последний известный курс на эту дату).
С дивидендами: каждая бумага берется индексом полной доходности (дивиденды реинвестируются в ту же бумагу в день отсечки).
Чистые функции pandas/numpy: без Streamlit и сети.
"""

import io

import numpy as np
import pandas as pd

from salesanalyzer.config import TRADING_DAYS
from salesanalyzer.core.benchmark import add_total_return
from salesanalyzer.core.pricesim import project_dividends

POSITION_COLS = ["ticker", "market", "qty", "buy_price"]
HOLDING_COLS = ["Тикер", "Рынок", "Валюта", "Количество", "Цена", "Курс к ₽", "Стоимость, ₽", "Доля, %",
                "Цена покупки", "Прибыль/убыток, ₽", "Прибыль/убыток, %"]
RUB = "RUB"


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Чистит ввод: тикер в верхнем регистре без пробелов, рынок ru/us, количество > 0; одинаковые тикеры одного рынка складываются
    (цена покупки — средневзвешенная). Строки без тикера или с нулевым количеством убираются."""
    if df is None or df.empty:
        return pd.DataFrame(columns=POSITION_COLS)
    d = df.copy()
    for col in POSITION_COLS:
        if col not in d.columns:
            d[col] = np.nan
    d["ticker"] = d["ticker"].fillna("").astype(str).str.strip().str.upper()
    d["market"] = d["market"].fillna("ru").astype(str).str.strip().str.lower().map(
        lambda m: "us" if m in ("us", "сша", "usa", "сша / международный") else "ru")
    d["qty"] = pd.to_numeric(d["qty"], errors="coerce")
    d["buy_price"] = pd.to_numeric(d["buy_price"], errors="coerce")
    d = d[(d["ticker"] != "") & (d["qty"] > 0)]
    if d.empty:
        return pd.DataFrame(columns=POSITION_COLS)
    d["cost"] = d["qty"] * d["buy_price"]
    g = d.groupby(["ticker", "market"], sort=False).agg(qty=("qty", "sum"), cost=("cost", "sum"),
                                                         known=("buy_price", lambda s: s.notna().all()))
    g["buy_price"] = np.where(g["known"], g["cost"] / g["qty"], np.nan)
    return g.reset_index()[POSITION_COLS]


def weights_to_qty(weights: pd.Series, prices_rub: pd.Series, amount: float) -> pd.Series:
    """Доли (в процентах, сумма может быть не 100 — нормируются) -> количество бумаг на сумму amount по текущим ценам в рублях."""
    w = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
    if w.sum() <= 0:
        return w * 0
    w = w / w.sum()
    return w * amount / prices_rub


def rate_on(dates, fx: pd.DataFrame) -> np.ndarray:
    """Курс на каждую дату: последний известный на эту дату (если дата раньше начала ряда — первый курс ряда)."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if fx is None or fx.empty:
        return np.full(len(dates), np.nan)
    f = fx[["begin", "close"]].dropna().sort_values("begin")
    days, vals = f["begin"].dt.normalize().values, f["close"].to_numpy(dtype=float)
    pos = np.searchsorted(days, dates.values, side="right") - 1
    return vals[np.clip(pos, 0, len(vals) - 1)]


def to_rub(prices: pd.DataFrame, dividends, currency: str, fx: pd.DataFrame):
    """Цены и дивиденды бумаги в рублях. Возвращает (цены begin, close; выплаты ex_date, dps) или (None, None), если нужен курс, а его нет."""
    p = prices[["begin", "close"]].dropna().sort_values("begin").reset_index(drop=True).copy()
    p["begin"] = pd.to_datetime(p["begin"])
    ev = pd.DataFrame(columns=["ex_date", "dps"]) if dividends is None or len(dividends) == 0 else dividends[["ex_date", "dps"]].copy()
    if currency == RUB:
        return p, ev
    if fx is None or fx.empty:
        return None, None
    p["close"] = p["close"] * rate_on(p["begin"], fx)
    if not ev.empty:
        ev["dps"] = ev["dps"].astype(float) * rate_on(ev["ex_date"], fx)
    return p, ev


def build_panel(assets: dict, years=None):
    """Общая таблица цен и полной доходности в рублях. assets: {тикер: {"prices": цены в ₽, "dividends": выплаты в ₽}}.
    Календарь — все даты торгов любой бумаги, начиная с даты, когда есть цены всех бумаг (пропуски заполняются последней ценой).
    years: оставить последние years лет. Возвращает (цены, полная доходность) — таблицы дата × тикер; полная доходность
    на первую дату равна цене."""
    if not assets:
        return pd.DataFrame(), pd.DataFrame()
    closes = {t: a["prices"].assign(begin=lambda d: pd.to_datetime(d["begin"]).dt.normalize())
              .drop_duplicates("begin", keep="last").set_index("begin")["close"] for t, a in assets.items()}
    start = max(s.index.min() for s in closes.values())
    end = max(s.index.max() for s in closes.values())
    if years is not None:
        start = max(start, end - pd.DateOffset(years=years))
    index = sorted(set().union(*[s.index for s in closes.values()]))
    index = pd.DatetimeIndex([d for d in index if start <= d <= end])
    prices = pd.DataFrame({t: s.reindex(s.index.union(index)).sort_index().ffill().reindex(index) for t, s in closes.items()})
    prices = prices.dropna()
    if prices.empty:
        return prices, prices
    tr = {}
    for t in prices.columns:
        frame = pd.DataFrame({"begin": prices.index, "close": prices[t].to_numpy()})
        tr[t] = add_total_return(frame, assets[t].get("dividends"))["tr"].to_numpy()
    return prices, pd.DataFrame(tr, index=prices.index)


def value_series(panel: pd.DataFrame, qty: pd.Series) -> pd.DataFrame:
    """Стоимость портфеля по дням: колонки begin, close (Σ количество × цена)."""
    q = qty.reindex(panel.columns).fillna(0)
    return pd.DataFrame({"begin": panel.index, "close": (panel * q).sum(axis=1).to_numpy()})


def holdings_table(positions: pd.DataFrame, last_price: dict, last_rate: dict, currency: dict) -> pd.DataFrame:
    """Состав на сегодня: стоимость в рублях, доля, прибыль/убыток к цене покупки (в рублях по текущему курсу)."""
    rows = []
    for _, r in positions.iterrows():
        t = r["ticker"]
        if t not in last_price:
            continue
        price, rate, cur = last_price[t], last_rate.get(t, 1.0), currency.get(t, RUB)
        value = r["qty"] * price * rate
        bp = r.get("buy_price")
        pnl = (price - bp) * r["qty"] * rate if bp is not None and not pd.isna(bp) and bp > 0 else None
        rows.append({"Тикер": t, "Рынок": "РФ" if r["market"] == "ru" else "США", "Валюта": cur, "Количество": r["qty"],
                     "Цена": price, "Курс к ₽": rate, "Стоимость, ₽": value, "Цена покупки": bp if pnl is not None else None,
                     "Прибыль/убыток, ₽": pnl, "Прибыль/убыток, %": (price / bp - 1) * 100 if pnl is not None else None})
    out = pd.DataFrame(rows, columns=HOLDING_COLS)
    total = out["Стоимость, ₽"].sum()
    out["Доля, %"] = out["Стоимость, ₽"] / total * 100 if total > 0 else np.nan
    return out


def contributions(prices: pd.DataFrame, tr: pd.DataFrame, qty: pd.Series) -> pd.DataFrame:
    """Вклад бумаг за период: в доходность (по цене и с дивидендами, п.п. от стоимости портфеля на старте) и в риск
    (доля в дисперсии портфеля по текущим долям: w_i × (Σw)_i / wᵀΣw, по дневным доходностям полной доходности)."""
    q = qty.reindex(prices.columns).fillna(0)
    v0 = float((prices.iloc[0] * q).sum())
    if v0 <= 0 or len(prices) < 2:
        return pd.DataFrame()
    price_c = (prices.iloc[-1] - prices.iloc[0]) * q / v0 * 100
    tr_c = (tr.iloc[-1] - tr.iloc[0]) * q / v0 * 100
    w = prices.iloc[-1] * q
    w = w / w.sum()
    rets = tr.pct_change().dropna()
    cov = rets.cov().to_numpy()
    wv = w.to_numpy()
    var = float(wv @ cov @ wv)
    risk = wv * (cov @ wv) / var * 100 if var > 0 else np.full(len(wv), np.nan)
    vol = rets.std().to_numpy() * np.sqrt(TRADING_DAYS) * 100
    out = pd.DataFrame({"Тикер": prices.columns, "Доля сейчас, %": wv * 100,
                        "Доходность бумаги по цене, %": ((prices.iloc[-1] / prices.iloc[0] - 1) * 100).to_numpy(),
                        "Вклад в доходность по цене, п.п.": price_c.to_numpy(),
                        "Вклад в доходность с дивидендами, п.п.": tr_c.to_numpy(),
                        "Волатильность бумаги (год), %": vol, "Вклад в риск, %": risk})
    return out


def correlation_matrix(tr: pd.DataFrame) -> pd.DataFrame:
    """Корреляции дневных доходностей бумаг (по полной доходности, чтобы дивидендные гэпы не искажали)."""
    if tr.shape[1] < 2:
        return pd.DataFrame()
    return tr.pct_change().dropna().corr()


def portfolio_dividend_events(dividends_rub: dict, qty: pd.Series) -> pd.DataFrame:
    """Прошлые выплаты портфеля в рублях: ex_date, dps = Σ количество × дивиденд (по всем бумагам на эту дату)."""
    parts = []
    for t, ev in dividends_rub.items():
        if ev is None or len(ev) == 0 or t not in qty.index:
            continue
        parts.append(pd.DataFrame({"ex_date": pd.to_datetime(ev["ex_date"]).dt.normalize(), "dps": ev["dps"].astype(float) * qty[t]}))
    if not parts:
        return pd.DataFrame(columns=["ex_date", "dps"])
    all_ev = pd.concat(parts, ignore_index=True)
    return all_ev.groupby("ex_date", as_index=False)["dps"].sum()


def dividend_calendar(dividends: dict, qty: pd.Series, last_rate: dict, last_date, months: int = 12) -> pd.DataFrame:
    """Ожидаемые выплаты на months месяцев вперед: объявленные и повтор выплат последних 12 месяцев через год с той же суммой.
    dividends: {тикер: выплаты в валюте бумаги}. Сумма в рублях — по текущему курсу. Колонки: Тикер, Дата отсечки, Дивиденд на акцию,
    Количество, Сумма, ₽, Откуда."""
    cols = ["Тикер", "Дата отсечки", "Дивиденд на акцию", "Количество", "Сумма, ₽", "Откуда"]
    end = pd.Timestamp(last_date) + pd.DateOffset(months=months)
    rows = []
    for t, ev in dividends.items():
        if t not in qty.index or ev is None or len(ev) == 0:
            continue
        for _, r in project_dividends(ev, last_date, end).iterrows():
            rows.append({"Тикер": t, "Дата отсечки": r["ex_date"], "Дивиденд на акцию": float(r["dps"]), "Количество": float(qty[t]),
                         "Сумма, ₽": float(r["dps"]) * float(qty[t]) * last_rate.get(t, 1.0), "Откуда": r["source"]})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("Дата отсечки").reset_index(drop=True) if not out.empty else out


def monthly_dividends(calendar: pd.DataFrame) -> pd.DataFrame:
    """Сумма ожидаемых выплат по месяцам (колонки Месяц, Сумма, ₽)."""
    if calendar.empty:
        return pd.DataFrame(columns=["Месяц", "Сумма, ₽"])
    m = calendar.assign(Месяц=pd.to_datetime(calendar["Дата отсечки"]).dt.to_period("M").dt.to_timestamp())
    return m.groupby("Месяц", as_index=False)["Сумма, ₽"].sum()


PORTFOLIO_FILE_COLS = ["ticker", "market", "qty", "weight", "buy_price"]


def read_portfolio_csv(data: bytes) -> pd.DataFrame:
    """Файл портфеля (CSV: ticker, market, qty, weight, buy_price; разделитель запятая или точка с запятой, десятичная запятая
    допускается) -> таблица с колонками PORTFOLIO_FILE_COLS (market: ru/us, числа — float или NaN). ValueError, если нет колонки ticker."""
    text = data.decode("utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise ValueError("файл пустой")
    sep = ";" if lines[0].count(";") > lines[0].count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("нет колонки ticker")
    out = pd.DataFrame({c: df[c] if c in df.columns else "" for c in PORTFOLIO_FILE_COLS})
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["market"] = out["market"].astype(str).str.strip().str.lower().map(lambda m: "us" if m in ("us", "сша", "usa") else "ru")
    for c in ("qty", "weight", "buy_price"):
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(" ", "").str.replace(",", "."), errors="coerce")
    return out[out["ticker"] != ""].reset_index(drop=True)
