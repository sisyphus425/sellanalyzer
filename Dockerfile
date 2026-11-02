# Образ приложения «Анализатор продаж компаний».
# Сборка: docker compose build     Запуск: docker compose up -d
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SALESANALYZER_DATA_DIR=/data

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

COPY streamlit_app.py .
COPY salesanalyzer ./salesanalyzer
COPY .streamlit ./.streamlit
# Данные по умолчанию (если папка /data не подключена как том). На сервере файлы подключаются из ./data, см. docker-compose.yml
COPY financials_ru.csv macro_ru.csv /data/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app /data
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "streamlit_app.py"]
