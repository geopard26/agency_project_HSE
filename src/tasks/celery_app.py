import logging
import os

from dotenv import load_dotenv

load_dotenv()

from src.logging_config import setup_logging

setup_logging(os.getenv("LOG_LEVEL", "INFO"))

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

sentry_logging = LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENV", "dev"),
    integrations=[sentry_logging],
    traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", 0.1)),
)

from celery import Celery

celery_app = Celery(
    "tasks",
    broker=os.getenv("BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("RESULT_BACKEND", None),
)

celery_app.autodiscover_tasks(["src.tasks"])
