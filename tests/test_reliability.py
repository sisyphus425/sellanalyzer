"""Надежность: повторные попытки, понятные сообщения об ошибках Мосбиржи и Yahoo, совместимость с версиями Streamlit."""
import importlib

import pandas as pd
import pytest


# ---------- общий механизм повторных попыток ----------

def flaky(failures, error=ConnectionError("сбой"), result="ok"):
    """Функция, которая падает `failures` раз, а потом возвращает result."""
    state = {"calls": 0}

    def func():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise error
        return result

    return func, state


def test_retry_succeeds_after_transient_failures_and_waits_longer_each_time(app):
    func, state = flaky(2)
    pauses = []
    assert app.call_with_retries(func, attempts=3, delay=1.0, sleep=pauses.append) == "ok"
    assert state["calls"] == 3 and pauses == [1.0, 2.0]


def test_retry_gives_up_after_all_attempts_and_raises_the_last_error(app):
    func, state = flaky(10, error=ConnectionError("нет сети"))
    with pytest.raises(ConnectionError):
        app.call_with_retries(func, attempts=4, delay=0, sleep=lambda s: None)
    assert state["calls"] == 4


def test_retry_does_not_repeat_permanent_errors(app):
    func, state = flaky(10, error=KeyError("постоянная"))
    with pytest.raises(KeyError):
        app.call_with_retries(func, attempts=5, delay=0, is_transient=lambda e: isinstance(e, ConnectionError),
                              sleep=lambda s: None)
    assert state["calls"] == 1


def test_retry_with_one_attempt_calls_once_and_without_pause(app):
    func, state = flaky(10)
    pauses = []
    with pytest.raises(ConnectionError):
        app.call_with_retries(func, attempts=1, sleep=pauses.append)
    assert state["calls"] == 1 and pauses == []


def test_retry_uses_settings_from_config_by_default(app, cfg, monkeypatch):
    monkeypatch.setattr(cfg, "RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(cfg, "RETRY_DELAY", 0.25)
    func, state = flaky(10)
    pauses = []
    with pytest.raises(ConnectionError):
        app.call_with_retries(func, sleep=pauses.append)
    assert state["calls"] == 2 and pauses == [0.25]


def test_network_settings_are_sane(cfg):
    assert cfg.RETRY_ATTEMPTS >= 1 and cfg.RETRY_DELAY >= 0 and cfg.REQUEST_TIMEOUT > 0
    assert 429 in cfg.RETRY_STATUSES and 503 in cfg.RETRY_STATUSES and 404 not in cfg.RETRY_STATUSES
    assert cfg.PRICE_HISTORY_YEARS in cfg.PRICE_PERIOD_OPTIONS and cfg.ISS_PAGE_SIZE > 0


# ---------- Мосбиржа ----------

class FakeRequests:
    """Заглушка модуля requests: возвращает заранее заданные ответы или бросает заданные ошибки по очереди."""

    class RequestException(Exception):
        pass

    class ConnectionError(RequestException):
        pass

    class Timeout(RequestException):
        pass

    class HTTPError(RequestException):
        def __init__(self, response):
            super().__init__(f"HTTP {response.status_code}")
            self.response = response

    def __init__(self, *script):
        self.script, self.calls = list(script), 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Resp:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code, self.payload, self.bad_json = status, payload, bad_json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise FakeRequests.HTTPError(self)

    def json(self):
        if self.bad_json:
            raise ValueError("не JSON")
        return self.payload


def block(columns, data):
    return {"columns": columns, "data": data}


@pytest.fixture
def moex(app, cfg, monkeypatch):
    monkeypatch.setattr(cfg, "RETRY_DELAY", 0)  # в тестах не ждем между попытками
    return importlib.import_module("salesanalyzer.data_sources.moex")


def use(monkeypatch, moex, *script):
    fake = FakeRequests(*script)
    monkeypatch.setattr(moex, "requests", fake)
    return fake


def test_iss_get_parses_blocks(app, moex, monkeypatch):
    payload = {"description": block(["name", "value"], [["NAME", "Сбербанк"]]), "other": "не блок"}
    fake = use(monkeypatch, moex, Resp(payload=payload))
    out = moex.iss_get("securities/SBER")
    assert list(out) == ["description"] and out["description"]["value"].tolist() == ["Сбербанк"] and fake.calls == 1


def test_iss_get_retries_temporary_server_errors(app, moex, monkeypatch):
    fake = use(monkeypatch, moex, Resp(503), Resp(502), Resp(payload={"x": block(["a"], [[1]])}))
    assert moex.iss_get("x")["x"]["a"].tolist() == [1] and fake.calls == 3


def test_iss_get_retries_connection_problems(app, moex, monkeypatch):
    fake = use(monkeypatch, moex, FakeRequests.ConnectionError("сеть"), Resp(payload={}))
    assert moex.iss_get("x") == {} and fake.calls == 2


def test_iss_get_reports_persistent_server_error_in_plain_language(app, cfg, moex, monkeypatch):
    fake = use(monkeypatch, moex, *[Resp(503)] * cfg.RETRY_ATTEMPTS)
    with pytest.raises(app.DataSourceError) as error:
        moex.iss_get("x")
    assert "временно недоступна" in str(error.value) and "503" in str(error.value)
    assert fake.calls == cfg.RETRY_ATTEMPTS and error.value.source == "MOEX"


def test_iss_get_does_not_retry_client_errors(app, moex, monkeypatch):
    fake = use(monkeypatch, moex, Resp(404))
    with pytest.raises(app.DataSourceError) as error:
        moex.iss_get("x")
    assert "404" in str(error.value) and fake.calls == 1


def test_iss_get_messages_for_no_connection_and_timeout(app, cfg, moex, monkeypatch):
    use(monkeypatch, moex, *[FakeRequests.ConnectionError("x")] * cfg.RETRY_ATTEMPTS)
    with pytest.raises(app.DataSourceError) as error:
        moex.iss_get("x")
    assert "Нет соединения" in str(error.value)
    use(monkeypatch, moex, *[FakeRequests.Timeout("x")] * cfg.RETRY_ATTEMPTS)
    with pytest.raises(app.DataSourceError) as error:
        moex.iss_get("x")
    assert "не ответила" in str(error.value) and str(cfg.REQUEST_TIMEOUT) in str(error.value)


def test_iss_get_reports_unreadable_answer(app, moex, monkeypatch):
    use(monkeypatch, moex, Resp(bad_json=True))
    with pytest.raises(app.DataSourceError) as error:
        moex.iss_get("x")
    assert "неожиданный ответ" in str(error.value)


def test_moex_name_and_unknown_ticker(app, moex, monkeypatch):
    use(monkeypatch, moex, Resp(payload={"description": block(["name", "title", "value"], [["NAME", "Название", "ПАО Сбербанк"]])}))
    assert moex.get_moex_name("SBER") == "ПАО Сбербанк"
    use(monkeypatch, moex, Resp(payload={"description": block(["name", "title", "value"], [])}))
    assert moex.get_moex_name("ZZZZ") is None


def test_candles_are_loaded_page_by_page(app, cfg, moex, monkeypatch):
    monkeypatch.setattr(cfg, "ISS_PAGE_SIZE", 2)
    pages = {0: [["2025-01-02 00:00:00", 10.0], ["2025-01-03 00:00:00", 11.0]], 2: [["2025-01-04 00:00:00", 12.0]]}
    starts = []

    def fake_iss_get(path, **params):
        starts.append(params["start"])
        return {"candles": pd.DataFrame(pages[params["start"]], columns=["begin", "close"])}

    monkeypatch.setattr(moex, "iss_get", fake_iss_get)
    out = moex.get_moex_candles("SBER", 1)
    assert starts == [0, 2] and out["close"].tolist() == [10.0, 11.0, 12.0]
    assert pd.api.types.is_datetime64_any_dtype(out["begin"])


def test_candles_empty_answer_gives_empty_table(app, moex, monkeypatch):
    monkeypatch.setattr(moex, "iss_get", lambda path, **p: {"candles": pd.DataFrame(columns=["begin", "close"])})
    assert moex.get_moex_candles("ZZZZ", 1).empty


# ---------- Yahoo Finance ----------

class FakeTicker:
    """Заглушка yf.Ticker: каждое свойство либо возвращает значение, либо бросает ошибку; считаются обращения."""

    def __init__(self, info=None, financials=None, dividends=None, history=None):
        self._spec = {"info": info, "financials": financials, "dividends": dividends}
        self._history = history
        self.counts = {"info": 0, "financials": 0, "dividends": 0, "history": 0}

    def _get(self, name):
        self.counts[name] += 1
        value = self._spec[name]
        if isinstance(value, Exception):
            raise value
        return value

    info = property(lambda self: self._get("info"))
    financials = property(lambda self: self._get("financials"))
    dividends = property(lambda self: self._get("dividends"))

    def history(self, period=None, interval=None):
        self.counts["history"] += 1
        if isinstance(self._history, Exception):
            raise self._history
        return self._history


@pytest.fixture
def yahoo(app, cfg, monkeypatch):
    monkeypatch.setattr(cfg, "RETRY_DELAY", 0)
    return importlib.import_module("salesanalyzer.data_sources.yahoo")


def install(monkeypatch, yahoo, ticker):
    monkeypatch.setattr(yahoo, "yf", type("YF", (), {"Ticker": staticmethod(lambda t: ticker)}))
    return ticker


def fin_table():
    cols = pd.to_datetime(["2025-09-30", "2024-09-30"])
    return pd.DataFrame({cols[0]: [416.0, 112.0], cols[1]: [391.0, 93.7]}, index=["Total Revenue", "Net Income"])


def dividend_series():
    now = pd.Timestamp.now(tz="America/New_York")
    return pd.Series([0.25, 0.25, 0.20], index=[now - pd.Timedelta(days=30), now - pd.Timedelta(days=120), now - pd.Timedelta(days=500)])


def good_ticker(**overrides):
    spec = {"info": {"longName": "Apple Inc.", "sector": "Technology", "currency": "USD", "financialCurrency": "USD",
                     "marketCap": 3.8e12, "sharesOutstanding": 14.8e9, "website": "https://apple.com"},
            "financials": fin_table(), "dividends": dividend_series()}
    spec.update(overrides)
    return FakeTicker(**spec)


def test_us_data_for_a_normal_ticker(app, yahoo, monkeypatch):
    install(monkeypatch, yahoo, good_ticker())
    data = yahoo.get_us_data("AAPL")
    assert data["name"] == "Apple Inc." and data["currency"] == "USD" and data["market_cap"] == 3.8e12
    assert data["wide"]["year"].tolist() == [2024, 2025] and data["df"]["revenue"].tolist() == [391.0, 416.0]
    assert data["div_ttm"] == pytest.approx(0.5) and data["dividends"]["dps"].sum() == pytest.approx(0.7)


def test_us_data_unknown_ticker_gives_none_without_retries(app, yahoo, monkeypatch):
    ticker = install(monkeypatch, yahoo, good_ticker(info=Exception("404 Client Error: Not Found for url")))
    assert yahoo.get_us_data("ZZZZ") is None and ticker.counts["info"] == 1
    install(monkeypatch, yahoo, good_ticker(info={}))
    assert yahoo.get_us_data("ZZZZ") is None


def test_us_data_network_failure_is_retried_and_reported_in_plain_language(app, cfg, yahoo, monkeypatch):
    ticker = install(monkeypatch, yahoo, good_ticker(info=ConnectionError("Failed to resolve 'query1.finance.yahoo.com'")))
    with pytest.raises(app.DataSourceError) as error:
        yahoo.get_us_data("AAPL")
    assert "Нет соединения" in str(error.value) and ticker.counts["info"] == cfg.RETRY_ATTEMPTS
    assert error.value.source == "Yahoo"


def test_us_data_rate_limit_message(app, yahoo, monkeypatch):
    install(monkeypatch, yahoo, good_ticker(info=Exception("Too Many Requests. Rate limited. Try after a while.")))
    with pytest.raises(app.DataSourceError) as error:
        yahoo.get_us_data("AAPL")
    assert "ограничил число запросов" in str(error.value)


def test_us_data_failure_of_financials_is_an_error(app, yahoo, monkeypatch):
    install(monkeypatch, yahoo, good_ticker(financials=RuntimeError("что-то сломалось")))
    with pytest.raises(app.DataSourceError) as error:
        yahoo.get_us_data("AAPL")
    assert "не вернул данные" in str(error.value)


def test_us_data_dividends_are_optional(app, yahoo, monkeypatch):
    install(monkeypatch, yahoo, good_ticker(dividends=RuntimeError("нет дивидендов")))
    data = yahoo.get_us_data("AAPL")
    assert data["name"] == "Apple Inc." and data["dividends"].empty and data["div_ttm"] is None


def test_us_prices(app, yahoo, monkeypatch):
    hist = pd.DataFrame({"Close": [10.0, 11.0]}, index=pd.to_datetime(["2025-01-02", "2025-01-03"]).tz_localize("America/New_York"))
    install(monkeypatch, yahoo, good_ticker().__class__(history=hist))
    out = yahoo.get_us_prices("AAPL", 1)
    assert out.columns.tolist() == ["begin", "close"] and out["close"].tolist() == [10.0, 11.0]
    assert out["begin"].dt.tz is None


def test_us_prices_empty_and_failure(app, yahoo, monkeypatch):
    install(monkeypatch, yahoo, FakeTicker(history=pd.DataFrame()))
    assert yahoo.get_us_prices("ZZZZ", 1).empty
    ticker = install(monkeypatch, yahoo, FakeTicker(history=TimeoutError("timed out")))
    with pytest.raises(app.DataSourceError) as error:
        yahoo.get_us_prices("AAPL", 1)
    assert "Нет соединения" in str(error.value) and ticker.counts["history"] > 1


# ---------- совместимость с версиями Streamlit ----------

def test_stretch_uses_width_when_it_defaults_to_stretch(app):
    def new_api(data, width="stretch"):
        pass

    assert app.stretch(new_api) == {"width": "stretch"}


@pytest.mark.parametrize("kind", ["old", "opaque", "other_default"])
def test_stretch_falls_back_to_use_container_width(app, kind):
    def old_api(data, width=None, use_container_width=False):
        pass

    def opaque(*args, **kwargs):
        pass

    def other_default(data, width="content"):
        pass

    func = {"old": old_api, "opaque": opaque, "other_default": other_default}[kind]
    assert app.stretch(func) == {"use_container_width": True}


def test_show_helpers_pass_the_data_to_streamlit(app, fake_st):
    fake_st.messages.clear()
    app.show_chart("график")
    app.show_table(pd.DataFrame({"a": [1]}), hide_index=True)
    names = [name for name, _ in fake_st.messages]
    assert names == ["plotly_chart", "dataframe"] and fake_st.messages[0][1] == ("график",)


def test_data_source_error_keeps_the_message_and_source(app):
    error = app.DataSourceError("Нет соединения", "MOEX")
    assert str(error) == "Нет соединения" and error.source == "MOEX" and isinstance(error, Exception)
