"""Сбор данных для сравнения компаний: объединяет CSV, Мосбиржу и Yahoo Finance."""

import pandas as pd

from salesanalyzer import config
from salesanalyzer.core.growth import add_growth
from salesanalyzer.core.profitability import add_margins, make_wide
from salesanalyzer.core.benchmark import add_total_return, compare_with_benchmark
from salesanalyzer.data_sources.csv_store import load_dividends_ru
from salesanalyzer.data_sources.moex import get_moex_candles, get_moex_index_candles
from salesanalyzer.data_sources.yahoo import get_us_data, get_us_dividend_events, get_us_prices
from salesanalyzer.data_sources.errors import DataSourceError


def collect_revenue(ru_tickers, us_tickers, revenue_all, problems) -> pd.DataFrame:
    """Единая таблица выручки: label, year, revenue, unit, is_estimate, growth."""
    cols = ["label", "year", "revenue", "unit", "is_estimate"]
    frames = []
    for t in ru_tickers:
        rows = revenue_all[revenue_all["ticker"] == t]
        if rows.empty:
            problems.append(f"{t}: нет данных в {config.CSV_PATH.name}")
            continue
        frames.append(rows.assign(label=t)[cols])
    for t in us_tickers:
        try:
            data = get_us_data(t)
        except Exception as e:
            problems.append(f"{t}: ошибка загрузки ({e})")
            continue
        if data is None:
            problems.append(f"{t}: тикер не найден")
        elif data["df"] is None:
            problems.append(f"{t}: нет открытых данных по выручке")
        else:
            frames.append(data["df"].assign(label=t, unit="$", is_estimate=False)[cols])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return pd.concat([add_growth(g) for _, g in out.groupby("label", sort=False)], ignore_index=True)


def collect_profit(ru_tickers, us_tickers, ru_all) -> pd.DataFrame:
    """Единая таблица прибыльности: label, year, net_income, ebitda, roe, net_margin, ebitda_margin."""
    parts = []
    for t in ru_tickers:
        w, unit, mixed = make_wide(ru_all[ru_all["ticker"] == t] if not ru_all.empty else ru_all)
        if not w.empty:
            parts.append(add_margins(w, enabled=not mixed).assign(label=t, unit=unit))
    for t in us_tickers:
        try:
            data = get_us_data(t)
        except Exception:
            continue  # ошибка уже показана на вкладках выручки
        if data is not None and not data["wide"].empty:
            parts.append(add_margins(data["wide"]).assign(label=t, unit="$"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _series_for(label, loader, years, problems, dividends_loader=None, total=False):
    """Цены одной бумаги (begin, close). При total=True close заменяется индексом полной доходности, если есть даты дивидендов."""
    try:
        df = loader(label, years)
    except Exception as e:
        problems.append(f"{label}: {e}" if isinstance(e, DataSourceError) else f"{label}: ошибка загрузки цен ({e})")
        return None
    if df is None or df.empty:
        problems.append(f"{label}: нет истории цен")
        return None
    df = df[["begin", "close"]]
    if total and dividends_loader is not None:
        try:
            events = dividends_loader(label)
        except Exception:
            events = None
        if events is None or len(events) == 0:
            problems.append(f"{label}: нет дат выплаты дивидендов, показана только цена")
        else:
            tr = add_total_return(df, events)
            df = tr[["begin", "tr"]].rename(columns={"tr": "close"})
    return df


def collect_prices(ru_tickers, us_tickers, years, problems, with_index=False, total=False) -> pd.DataFrame:
    """Цены закрытия, приведенные к 100 на общую дату старта. Колонки: begin, label, close, index, is_benchmark.
    with_index: добавить индексы рынков выбранных бумаг (IMOEX для РФ, S&P 500 для США).
    total: полная доходность (цена + реинвестированные дивиденды); индексы тогда берутся тоже полной доходности."""
    series = {}
    for t in ru_tickers:
        df = _series_for(t, get_moex_candles, years, problems, load_dividends_ru, total)
        if df is not None:
            series[t] = df
    for t in us_tickers:
        df = _series_for(t, get_us_prices, years, problems, lambda x: get_us_dividend_events(x, years), total)
        if df is not None:
            series[t] = df
    if len(series) == 0:
        return pd.DataFrame()
    benchmarks = set()
    if with_index:
        markets = [("ru", get_moex_index_candles)] * bool(ru_tickers) + [("us", get_us_prices)] * bool(us_tickers)
        for market, loader in markets:
            secid, name = config.BENCHMARKS[market]["total" if total else "price"]
            df = _series_for(secid, loader, years, problems)
            if df is not None:
                series[name] = df
                benchmarks.add(name)
    start = max(df["begin"].min() for df in series.values())
    parts = []
    for label, df in series.items():
        df = df[df["begin"] >= start].sort_values("begin")
        if df.empty or df["close"].iloc[0] <= 0:
            continue
        parts.append(df.assign(label=label, index=df["close"] / df["close"].iloc[0] * 100, is_benchmark=label in benchmarks))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def market_of(label, ru_tickers) -> str:
    return "ru" if label in ru_tickers else "us"


def benchmark_table(prices: pd.DataFrame, ru_tickers, total=False) -> pd.DataFrame:
    """Бета, корреляция и разница доходностей каждой бумаги с индексом ее рынка (по таблице collect_prices с индексами)."""
    if prices.empty or "is_benchmark" not in prices.columns:
        return pd.DataFrame()
    rows = []
    for label, g in prices[~prices["is_benchmark"]].groupby("label", sort=False):
        name = config.BENCHMARKS[market_of(label, ru_tickers)]["total" if total else "price"][1]
        bench = prices[prices["label"] == name]
        if bench.empty:
            continue
        m, _ = compare_with_benchmark(g[["begin", "close"]], bench[["begin", "close"]])
        if m is None:
            continue
        rows.append({"Компания": label, "Индекс": name, "Разница с индексом, п.п.": round(m["excess"], 1),
                     "Бета": None if m["beta"] is None else round(m["beta"], 2),
                     "Корреляция": None if m["corr"] is None else round(m["corr"], 2)})
    return pd.DataFrame(rows)
