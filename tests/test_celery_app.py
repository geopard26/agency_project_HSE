import importlib

import celery


def test_celery_app_configuration(monkeypatch):
    # 1) Задаём окружение
    monkeypatch.setenv("BROKER_URL", "redis://testbroker:1234/1")
    monkeypatch.setenv("RESULT_BACKEND", "rpc://")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SENTRY_DSN", "https://example@sentry.io/42")
    monkeypatch.setenv("ENV", "testenv")
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.42")

    # 2) Блокируем load_dotenv, чтобы он не стирал наши переменные
    monkeypatch.setattr("src.tasks.celery_app.load_dotenv", lambda: None)

    # 3) Мокаем setup_logging в модуле перед reload
    setup_called = {}

    def fake_setup_logging(level=None):
        setup_called["level"] = level

    monkeypatch.setattr("src.tasks.celery_app.setup_logging", fake_setup_logging)

    # 4) Создаём DummyIntegration с нужными атрибутами
    class DummyIntegration:
        identifier = "dummy_integration"

        def __init__(self, level, event_level):
            self.level = level
            self.event_level = event_level

        @staticmethod
        def setup_once():
            pass

    # Патчим реальный модуль, откуда будет импортироваться интеграция
    import sentry_sdk.integrations.logging as spl

    monkeypatch.setattr(spl, "LoggingIntegration", DummyIntegration)

    # 5) Патчим реальный sentry_sdk.init, а не локальную переменную в celery_app
    init_args = {}

    def fake_sentry_init(*, dsn, environment, integrations, traces_sample_rate):
        init_args["dsn"] = dsn
        init_args["environment"] = environment
        init_args["integrations"] = integrations
        init_args["traces_sample_rate"] = traces_sample_rate

    import sentry_sdk

    monkeypatch.setattr(sentry_sdk, "init", fake_sentry_init)

    # 6) Мокаем Celery в пакете celery
    celery_calls = {}

    class FakeCelery:
        def __init__(self, name, broker, backend):
            celery_calls["name"] = name
            celery_calls["broker"] = broker
            celery_calls["backend"] = backend
            self.conf = type("C", (), {"broker_url": broker})

        def autodiscover_tasks(self, args):
            celery_calls["autodiscover"] = args

    monkeypatch.setattr(celery, "Celery", FakeCelery, raising=True)

    # 7) Перезагружаем модуль, чтобы все подмены вступили в силу
    import src.tasks.celery_app as mod

    importlib.reload(mod)

    # 8) Проверяем, что setup_logging получил нужный уровень
    assert setup_called["level"] == "DEBUG"

    # 9) Проверяем, что sentry_logging в модуле — наш DummyIntegration
    assert isinstance(mod.sentry_logging, DummyIntegration)

    # 10) Проверяем, что init_args заполнились ожидаемо
    assert init_args == {
        "dsn": "https://example@sentry.io/42",
        "environment": "testenv",
        "integrations": [mod.sentry_logging],
        "traces_sample_rate": 0.42,
    }

    # 11) Проверяем Celery
    assert celery_calls["name"] == "tasks"
    assert celery_calls["broker"] == "redis://testbroker:1234/1"
    assert celery_calls["backend"] == "rpc://"
    assert celery_calls["autodiscover"] == ["src.tasks"]
