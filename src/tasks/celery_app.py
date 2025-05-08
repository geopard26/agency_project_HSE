import logging
import os

from dotenv import load_dotenv

# 1) Сначала загружаем переменные окружения
load_dotenv()  # теперь os.getenv("SENTRY_DSN") и другие будут доступны

# 2) Настройка логирования до всего остального
from src.logging_config import setup_logging

setup_logging(os.getenv("LOG_LEVEL", "INFO"))

# 3) Инициализация Sentry (опционально)
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

sentry_logging = LoggingIntegration(
    level=logging.INFO,  # захват любых INFO+ в breadcrumbs
    event_level=logging.ERROR,  # события ERROR+ в Sentry
)

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),  # DSN из .env
    environment=os.getenv("ENV", "dev"),
    integrations=[sentry_logging],
    traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", 0.1)),
)

# 4) Создаём экземпляр Celery
from celery import Celery

celery_app = Celery(
    "tasks",
    broker=os.getenv("BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("RESULT_BACKEND", None),  # можно оставить None
)

# 5) Автопоиск задач в вашем пакете
celery_app.autodiscover_tasks(["src.tasks"])

# 6) (Опционально) логгер для задач
logger = logging.getLogger(__name__)
logger.info("Celery app initialized with broker %s", celery_app.conf.broker_url)
