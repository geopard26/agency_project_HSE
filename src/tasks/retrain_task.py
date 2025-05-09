import os

import src.preprocessing.process_data as proc_mod
from src.logging_config import get_logger
from src.models.train_catboost import train_catboost
from src.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="src.tasks.retrain_task.retrain_model",
    acks_late=True,
)
def retrain_model(self, *args, **kwargs):
    """
    Фоновая задача: переобучает CatBoost-модель на полном датасете.
    При любой ошибке (в том числе отсутствие столбца 'is_agency')
    использует retry() до 3 раз с задержкой 60 секунд.
    """
    logger.info("=== Запуск задачи retrain_model ===")

    # В тестах первый аргумент — DummySelf с методом retry(),
    # в проде это будет просто self (Celery Task)
    retry_target = args[0] if args and hasattr(args[0], "retry") else self

    # 1) Загрузка и предобработка
    raw_df = proc_mod.load_raw(os.getenv("RAW_DATA_PATH", "data/raw/data.csv"))
    proc_df = proc_mod.clean_and_feature_engineer(raw_df)

    # 2) Проверка наличия целевой метки
    if "is_agency" not in proc_df.columns:
        logger.error("В данных отсутствует столбец 'is_agency', retry через 60s")
        return retry_target.retry(
            exc=ValueError("В данных отсутствует столбец 'is_agency'"),
            countdown=60,
            max_retries=3,
        )

    try:
        # 3) Формируем X и y
        X = proc_df.drop(columns=["id", "is_agency"], errors="ignore")
        y = proc_df["is_agency"]
        logger.debug(
            "Данные для обучения: %d строк, %d признаков", X.shape[0], X.shape[1]
        )

        # 4) Обучение модели
        train_catboost(
            X,
            y,
            save_path=os.getenv("MODEL_SAVE_PATH", "models/catboost_model.pkl"),
        )

        logger.info("Retraining completed successfully")
        return {"status": "success"}

    except Exception as exc:
        logger.exception("Ошибка в retrain_model, retry через 60s")
        return retry_target.retry(exc=exc, countdown=60, max_retries=3)
