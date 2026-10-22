"""Повторные попытки для нестабильных сетевых вызовов."""

import time

from salesanalyzer import config


def call_with_retries(func, *, attempts=None, delay=None, is_transient=None, sleep=None):
    """Вызывает func() до `attempts` раз, между попытками пауза delay, 2*delay, 4*delay и т.д.
    Повторяет только если is_transient(ошибка) истинно (без is_transient повторяются любые ошибки).
    Последняя ошибка пробрасывается как есть. Значения по умолчанию берутся из config."""
    attempts = max(1, config.RETRY_ATTEMPTS if attempts is None else attempts)
    delay = config.RETRY_DELAY if delay is None else delay
    sleep = time.sleep if sleep is None else sleep
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as error:
            if attempt == attempts or (is_transient is not None and not is_transient(error)):
                raise
            sleep(delay * 2 ** (attempt - 1))
