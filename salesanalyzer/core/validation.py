"""Контроль качества файла данных: поиск ошибок, пропусков и подозрительных значений."""

import pandas as pd

from salesanalyzer.config import BOOL_VALUES, INDICATOR_NAMES, ISSUE_COLS, JUMP_LIMITS, KNOWN_INDICATORS, KNOWN_UNITS, MONEY_INDICATORS, REQUIRED_COLS, SEV_ERR, SEV_INFO, SEV_ORDER, SEV_WARN


def _issues_df(issues: list) -> pd.DataFrame:
    df = pd.DataFrame(issues, columns=ISSUE_COLS)
    if df.empty:
        return df
    df["_o"] = df["Серьезность"].map(SEV_ORDER)
    df = df.sort_values(["_o", "Тикер", "Проверка"], kind="stable").drop(columns="_o")
    return df.reset_index(drop=True)


def validate_financials(raw: pd.DataFrame, today_year: int):
    """Проверяет сырой CSV. Возвращает (таблица проблем, таблица корректных строк с типами)."""
    issues = []

    def add(sev, check, msg, ticker="", year="", indicator="", line=None):
        issues.append({"Серьезность": sev, "Проверка": check, "Тикер": ticker, "Год": year,
                       "Показатель": INDICATOR_NAMES.get(indicator, indicator), "Строка в файле": line,
                       "Описание": msg})

    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        add(SEV_ERR, "Структура файла", "Нет колонок: " + ", ".join(missing) + ". Приложение не сможет прочитать файл.")
        return _issues_df(issues), pd.DataFrame()
    if raw.empty:
        add(SEV_ERR, "Структура файла", "Файл не содержит данных.")
        return _issues_df(issues), pd.DataFrame()

    df = raw.copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df["line"] = df.index + 2  # строка 1 — заголовок
    df["ticker"] = df["ticker"].str.upper()
    df["indicator"] = df["indicator"].str.lower()
    year = pd.to_numeric(df["year"], errors="coerce")
    value = pd.to_numeric(df["value"], errors="coerce")

    bad = pd.Series(False, index=df.index)
    for i, r in df.iterrows():
        ln, t, ind = int(r["line"]), r["ticker"], r["indicator"]
        if not t:
            add(SEV_ERR, "Пустой тикер", "Строка без тикера будет некорректно обработана.", "", r["year"], ind, ln)
            bad[i] = True
            continue
        if pd.isna(year[i]) or year[i] != int(year[i]):
            add(SEV_ERR, "Некорректный год", f"«{r['year']}» не похоже на год. Строка игнорируется.", t, r["year"], ind, ln)
            bad[i] = True
            continue
        if year[i] < 2000 or year[i] > today_year:
            add(SEV_WARN, "Год вне диапазона", f"Год {int(year[i])} вне диапазона 2000–{today_year}.", t, int(year[i]), ind, ln)
        if ind not in KNOWN_INDICATORS:
            add(SEV_WARN, "Неизвестный показатель",
                f"«{ind}» не распознан (допустимо: {', '.join(KNOWN_INDICATORS)}). Строка игнорируется.",
                t, int(year[i]), ind, ln)
            bad[i] = True
            continue
        if pd.isna(value[i]):
            add(SEV_ERR, "Значение не число",
                f"«{r['value']}» нельзя прочитать как число; строка игнорируется. "
                "Используйте точку как десятичный разделитель и не используйте пробелы.", t, int(year[i]), ind, ln)
            bad[i] = True
            continue
        if r["is_estimate"].lower() not in BOOL_VALUES:
            add(SEV_WARN, "Флаг оценки", f"«{r['is_estimate']}» не похоже на true/false; считается как false.",
                t, int(year[i]), ind, ln)
        if not r["unit"]:
            add(SEV_WARN, "Нет единиц", "Не указаны единицы измерения.", t, int(year[i]), ind, ln)
        if not r["source"]:
            add(SEV_WARN, "Нет источника", "Не указан источник данных.", t, int(year[i]), ind, ln)

    ok = df[~bad].copy()
    if ok.empty:
        return _issues_df(issues), ok
    ok["year"] = year[~bad].astype(int)
    ok["value"] = value[~bad]
    ok["is_estimate"] = ok["is_estimate"].str.lower().isin(["true", "1", "yes", "да"])

    # Дубликаты
    dups = ok[ok.duplicated(["ticker", "year", "indicator"], keep=False)]
    for (t, y, ind), g in dups.groupby(["ticker", "year", "indicator"]):
        add(SEV_ERR, "Дубликат", f"Строки {', '.join(map(str, g['line']))} описывают одно и то же; "
            "используется последняя.", t, y, ind)
    ok = ok.drop_duplicates(["ticker", "year", "indicator"], keep="last")

    # Единицы
    for t, g in ok[ok["indicator"].isin(MONEY_INDICATORS)].groupby("ticker"):
        units = [u for u in g["unit"].unique() if u]
        if len(units) > 1:
            add(SEV_ERR, "Разные единицы", f"{', '.join(units)}: маржа и сравнение будут некорректны.", t)
    for _, r in ok[(ok["indicator"] == "roe") & (ok["unit"] != "%") & (ok["unit"] != "")].iterrows():
        add(SEV_WARN, "Единицы ROE", f"ROE должен быть в «%», указано «{r['unit']}».",
            r["ticker"], r["year"], "roe", r["line"])
    for u in sorted(set(ok["unit"]) - KNOWN_UNITS - {""}):
        tick = ", ".join(sorted(ok.loc[ok["unit"] == u, "ticker"].unique()))
        add(SEV_INFO, "Незнакомые единицы", f"«{u}» ({tick}): убедитесь, что это не опечатка.", tick)

    # Пропуски, устаревшие данные, покрытие
    for (t, ind), g in ok.groupby(["ticker", "indicator"]):
        ys = set(g["year"])
        gaps = [y for y in range(min(ys), max(ys) + 1) if y not in ys]
        if gaps:
            add(SEV_WARN, "Пропущенные годы", f"Нет данных за {', '.join(map(str, gaps))}.", t, "", ind)
    for t, g in ok.groupby("ticker"):
        last = int(g["year"].max())
        if last < today_year - 1:
            add(SEV_WARN, "Устаревшие данные", f"Последний год в данных — {last}, ожидается {today_year - 1}.", t, last)
        inds = set(g["indicator"])
        if "shares" not in inds:
            add(SEV_INFO, "Нет числа акций", "Без indicator = shares не считаются капитализация и мультипликаторы.", t)
        elif "dps" not in inds:
            add(SEV_INFO, "Нет дивидендов", "Нет строк indicator = dps: дивиденды и дивидендная доходность не показываются.", t)
        if "revenue" not in inds:
            add(SEV_WARN, "Нет выручки", "Без выручки не считаются маржа и рост.", t)
        else:
            if "net_income" not in inds:
                add(SEV_INFO, "Нет чистой прибыли", "Вкладка «Прибыльность» будет неполной.", t)
            n_rev = g.loc[g["indicator"] == "revenue", "year"].nunique()
            if n_rev < 3:
                add(SEV_INFO, "Короткий ряд", f"Выручка только за {n_rev} г.: CAGR и сравнение будут ограничены.", t)

    # Расчетные оценки
    for _, r in ok[ok["is_estimate"] & (ok["note"] == "")].iterrows():
        add(SEV_WARN, "Оценка без пояснения", "Расчетное значение без комментария: опишите в note, как оно получено.",
            r["ticker"], r["year"], r["indicator"], r["line"])

    # Здравый смысл
    for _, r in ok[(ok["indicator"] == "shares") & (ok["value"] <= 0)].iterrows():
        add(SEV_ERR, "Число акций ≤ 0", f"Значение {r['value']} выглядит ошибочным.", r["ticker"], r["year"], "shares", r["line"])
    for _, r in ok[(ok["indicator"] == "dps") & (ok["value"] < 0)].iterrows():
        add(SEV_ERR, "Дивиденд < 0", f"Значение {r['value']} выглядит ошибочным.", r["ticker"], r["year"], "dps", r["line"])
    for _, r in ok[(ok["indicator"] == "revenue") & (ok["value"] <= 0)].iterrows():
        add(SEV_ERR, "Выручка ≤ 0", f"Значение {r['value']} выглядит ошибочным.", r["ticker"], r["year"], "revenue", r["line"])
    for _, r in ok[(ok["indicator"] == "roe") & ((ok["value"] < -100) | (ok["value"] > 100))].iterrows():
        add(SEV_WARN, "ROE вне диапазона", f"ROE {r['value']}%: проверьте, что значение в процентах.",
            r["ticker"], r["year"], "roe", r["line"])
    wide = ok.pivot_table(index=["ticker", "year"], columns="indicator", values="value", aggfunc="last").reset_index()
    for c in KNOWN_INDICATORS:
        if c not in wide.columns:
            wide[c] = float("nan")
    for _, r in wide.iterrows():
        if pd.notna(r["revenue"]) and r["revenue"] > 0 and pd.notna(r["net_income"]) \
                and r["net_income"] / r["revenue"] * 100 > 100:
            add(SEV_WARN, "Чистая маржа > 100%", "Чистая прибыль больше выручки: проверьте значения и определения.",
                r["ticker"], r["year"], "net_income")
        if pd.notna(r["ebitda"]) and pd.notna(r["net_income"]) and r["ebitda"] < r["net_income"]:
            add(SEV_INFO, "EBITDA < чистой прибыли", "Обычно EBITDA больше чистой прибыли: проверьте определения.",
                r["ticker"], r["year"], "ebitda")

    # Резкие изменения год к году
    for (t, ind), g in ok.sort_values("year").groupby(["ticker", "indicator"]):
        if ind not in JUMP_LIMITS:
            continue
        limit, sev = JUMP_LIMITS[ind]
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            prev, cur = g.loc[i - 1], g.loc[i]
            if cur["year"] - prev["year"] == 1 and prev["value"] > 0 and cur["value"] > 0:
                ch = (cur["value"] / prev["value"] - 1) * 100
                if abs(ch) > limit:
                    add(sev, "Резкое изменение", f"{ch:+.0f}% к {int(prev['year'])} г.: проверьте значение, единицы "
                        "и сопоставимость определений.", t, int(cur["year"]), ind, cur["line"])

    return _issues_df(issues), ok


def coverage_matrix(ok: pd.DataFrame) -> pd.DataFrame:
    """Матрица покрытия: ✓ — есть значение, ≈ — оценка, пусто — нет данных."""
    if ok.empty:
        return pd.DataFrame()
    years = list(range(int(ok["year"].min()), int(ok["year"].max()) + 1))
    order = {k: i for i, k in enumerate(KNOWN_INDICATORS)}
    rows = []
    for (t, ind), g in ok.groupby(["ticker", "indicator"]):
        rec = {"Тикер": t, "Показатель": INDICATOR_NAMES[ind], "Единицы": g.sort_values("year")["unit"].iloc[-1],
               "_o": order[ind]}
        for y in years:
            sub = g[g["year"] == y]
            rec[str(y)] = "" if sub.empty else ("≈" if sub["is_estimate"].any() else "✓")
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["Тикер", "_o"]).drop(columns="_o").reset_index(drop=True)
