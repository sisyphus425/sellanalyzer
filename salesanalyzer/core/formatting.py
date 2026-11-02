"""Форматирование значений для отображения."""


def fmt_value(v: float, unit: str) -> str:
    return f"{v:,.1f} {unit}".strip() if abs(v) < 100 else f"{v:,.0f} {unit}".strip()
