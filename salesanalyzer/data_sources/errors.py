"""Общая ошибка источников данных."""


class DataSourceError(Exception):
    """Источник данных недоступен или вернул неожиданный ответ.
    Текст исключения написан для пользователя и может показываться в интерфейсе как есть."""

    def __init__(self, message: str, source: str = ""):
        super().__init__(message)
        self.source = source
