import logging  # noqa: F401
import os

from celery import Celery  # noqa: F401

import src.preprocessing.process_data as proc_mod
from src.logging_config import get_logger
from src.models.train_catboost import train_catboost
from src.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="src.tasks.retrain_task.retrain_model", acks_late=True)
def retrain_model(self, *args, **kwargs):
    """
    Фоновая задача Celery: перезапускает обучение CatBoost-модели на полном датасете.
    Если обучение падает, автоматически попробует повторить до 3 раз с паузой 60 секунд.
    """
    logger.info("=== Запуск задачи retrain_model ===")
    try:
        # 1) Загрузка и предобработка
        raw_df = proc_mod.load_raw(os.getenv("RAW_DATA_PATH", "data/raw/data.csv"))
        proc_df = proc_mod.clean_and_feature_engineer(raw_df)

        # 2) Сплит признаков и целей
        if "is_agency" not in proc_df.columns:
            raise ValueError("В данных отсутствует столбец 'is_agency'")
        X = proc_df.drop(columns=["id", "is_agency"], errors="ignore")
        y = proc_df["is_agency"]

        logger.debug(
            "Данные для обучения: %d строк, %d признаков", X.shape[0], X.shape[1]
        )

        # 3) Обучение модели
        train_catboost(
            X, y, save_path=os.getenv("MODEL_SAVE_PATH", "models/catboost_model.pkl")
        )
        logger.info("Retraining completed successfully")
        return {"status": "success"}

        # 4) (Опционально) можно сохранить модель по особому пути здесь,
        #    если train_catboost этого не делает сам:
        # model.save_model(os.getenv("MODEL_SAVE_PATH", "models/catboost_model.pkl"))

        return {"status": "success"}

    except Exception as exc:
        logger.exception("Ошибка в retrain_model, будет повтор через 60 секунд")
        # повторяем до 3 раз
        raise self.retry(exc=exc, countdown=60, max_retries=3)
