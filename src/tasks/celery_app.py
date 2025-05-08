import logging
import os

from dotenv import load_dotenv

# 1) Сначала подгружаем .env
load_dotenv()

# 2) Теперь настраиваем логирование — но если в globals уже есть setup_logging
#    (например, тест его замокал), то мы его не перезапишем
if "setup_logging" not in globals():
    from src.logging_config import setup_logging  # noqa: F401

# Вызываем setup_logging с уровнем из окружения
setup_logging(os.getenv("LOG_LEVEL", "INFO"))

# 3) Инициализация Sentry (опционально)
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

sentry_logging = LoggingIntegration(
    level=logging.INFO,  # INFO+ как breadcrumbs
    event_level=logging.ERROR,  # ERROR+ в Sentry
)

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENV", "dev"),
    integrations=[sentry_logging],
    traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", 0.1)),
)

# 4) Создаём Celery-приложение
from celery import Celery

celery_app = Celery(
    "tasks",
    broker=os.getenv("BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("RESULT_BACKEND", None),
)

# 5) Автопоиск тасков
celery_app.autodiscover_tasks(["src.tasks"])
