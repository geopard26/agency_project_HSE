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
    Фоновая задача Celery: переобучает CatBoost-модель на полном датасете.
    При любой ошибке (включая отсутствие столбца 'is_agency')
    выполняет retry() до 3 раз.
    """
    logger.info("=== Запуск задачи retrain_model ===")
    try:
        # 1) Загрузка и предобработка
        raw_df = proc_mod.load_raw(os.getenv("RAW_DATA_PATH", "data/raw/data.csv"))
        proc_df = proc_mod.clean_and_feature_engineer(raw_df)

        # 2) Проверяем наличие целевой метки
        if "is_agency" not in proc_df.columns:
            raise ValueError("В данных отсутствует столбец 'is_agency'")

        # 3) Формируем X и y
        X = proc_df.drop(columns=["id", "is_agency"], errors="ignore")
        y = proc_df["is_agency"]
        logger.debug(
            "Данные для обучения: %d строк, %d признаков", X.shape[0], X.shape[1]
        )

        # 4) Запускаем обучение
        train_catboost(
            X,
            y,
            save_path=os.getenv("MODEL_SAVE_PATH", "models/catboost_model.pkl"),
        )
        logger.info("Retraining completed successfully")
        return {"status": "success"}

    except Exception as exc:
        # Любая ошибка «упадёт» сюда, включая наш ValueError
        logger.exception("Ошибка в retrain_model, будет повтор через 60 секунд")
        # вызываем retry — DummySelf.retry поднимет RuntimeError("RETRY"),
        # который ждёт тест
        return self.retry(exc=exc, countdown=60, max_retries=3)
