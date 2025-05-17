# ─── Stage 1: Builder ─────────────────────────────────────────────────────────

FROM python:3.10-slim AS builder

# чтобы логи сразу писались в stdout
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Копируем только requirements, чтобы агрессивно закешировать pip install
COPY requirements.txt requirements-app.txt ./

# Устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      -r requirements.txt \
      -r requirements-app.txt

# ─── Stage 2: Production ───────────────────────────────────────────────────────

FROM python:3.10-slim AS prod

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Копируем уже установленные зависимости из билдера
# путь может отличаться, проверьте, где pip складывает пакеты в вашем образе:
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Копируем весь код приложения
COPY . .

# Открываем порт Streamlit
EXPOSE 8501

# Команда по-умолчанию — запуск Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.headless=true"]
