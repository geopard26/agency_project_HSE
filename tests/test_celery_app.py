import importlib
import logging  # noqa: F401
import os  # noqa: F401

import pytest  # noqa: F401


def test_celery_app_configuration(monkeypatch):
    """
    Проверяем, что при заданных переменных окружения:
    - setup_logging() вызывается с LOG_LEVEL
    - Sentry инициализируется с нужными параметрами
    - Celery строится с правильным broker/backend
    - autodiscover_tasks вызывается с ["src.tasks"]
    """
    # 1) Подставляем окружение
    monkeypatch.setenv("BROKER_URL", "redis://testbroker:1234/1")
    monkeypatch.setenv("RESULT_BACKEND", "rpc://")
    monkeypatch.setenv("SENTRY_DSN", "https://example@sentry.io/42")
    monkeypatch.setenv("ENV", "testenv")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.42")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    # 2) Подменяем load_dotenv, чтобы он ничего не делал
    monkeypatch.setattr("src.tasks.celery_app.load_dotenv", lambda: None)

    # 3) Ловим setup_logging
    setup_called = {}

    def fake_setup_logging(level=None):
        setup_called["level"] = level

    monkeypatch.setattr("src.tasks.celery_app.setup_logging", fake_setup_logging)

    # 4) Подменяем LoggingIntegration, чтобы запомнить параметры
    class DummyIntegration:
        def __init__(self, level, event_level):
            self.level = level
            self.event_level = event_level

    monkeypatch.setattr("src.tasks.celery_app.LoggingIntegration", DummyIntegration)

    # 5) Подменяем sentry_sdk.init
    init_args = {}

    def fake_sentry_init(*, dsn, environment, integrations, traces_sample_rate):
        init_args["dsn"] = dsn
        init_args["environment"] = environment
        init_args["integrations"] = integrations
        init_args["traces_sample_rate"] = traces_sample_rate

    monkeypatch.setattr(
        "src.tasks.celery_app.sentry_sdk",
        type("SS", (), {"init": staticmethod(fake_sentry_init)}),
    )

    # 6) Подменяем Celery, чтобы запоминать создание и autodiscover
    celery_calls = {}

    class FakeCelery:
        def __init__(self, name, broker, backend):
            celery_calls["name"] = name
            celery_calls["broker"] = broker
            celery_calls["backend"] = backend
            # make conf.broker_url available
            self.conf = type("C", (), {"broker_url": broker})

        def autodiscover_tasks(self, args):
            celery_calls["autodiscover"] = args

    monkeypatch.setattr("src.tasks.celery_app.Celery", FakeCelery)

    # 7) Перезагружаем модуль
    import sentry_sdk.integrations.logging as spl

    import src.tasks.celery_app as mod

    monkeypatch.setattr(spl, "LoggingIntegration", DummyIntegration)

    importlib.reload(mod)

    # 8) Проверяем вызовы
    assert setup_called["level"] == "DEBUG"

    # должно создаться именно поле sentry_logging типа DummyIntegration
    assert isinstance(mod.sentry_logging, DummyIntegration)
    assert init_args == {
        "dsn": "https://example@sentry.io/42",
        "environment": "testenv",
        "integrations": [mod.sentry_logging],
        "traces_sample_rate": 0.42,
    }

    assert celery_calls["name"] == "tasks"
    assert celery_calls["broker"] == "redis://testbroker:1234/1"
    assert celery_calls["backend"] == "rpc://"
    assert celery_calls["autodiscover"] == ["src.tasks"]
