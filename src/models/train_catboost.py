#!/usr/bin/env python3
import multiprocessing as mp  # noqa: F401
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np  # noqa: F401
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)

from src.logging_config import get_logger, setup_logging

# Настраиваем логирование
setup_logging()
logger = get_logger(__name__)

# Инициализируем MLflow-эксперимент
mlflow.set_experiment("agency_classifier_catboost")


def train_catboost(
    X,
    y,
    save_path: str = "models/catboost_model.pkl",
    param_search: bool = True,
    random_state: int = 42,
    n_iter: int = 20,
) -> CatBoostClassifier:
    """
    Тренирует CatBoostClassifier на X, y.
    Если param_search=True — делает RandomizedSearchCV, иначе просто fit().
    Логирует параметры, метрики и артефакты в MLflow, сохраняет модель в save_path.
    """
    with mlflow.start_run():
        # Параметры
        mlflow.log_param("param_search", param_search)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_iter", n_iter)

        # Сплит
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        mlflow.log_param("train_size", len(y_train))
        mlflow.log_param("test_size", len(y_test))

        # Классовые веса
        counts = y_train.value_counts().to_dict()
        weight_ratio = counts.get(0, 1) / (counts.get(1, 1) or 1)
        class_weights = [1, weight_ratio]
        mlflow.log_param("class_weight_ratio", weight_ratio)

        # Базовая модель
        base_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            class_weights=class_weights,
            od_type="Iter",
            od_wait=50,
            verbose=False,
        )

        if param_search:
            # Гиперпараметры
            param_distributions = {
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "depth": [4, 6, 8],
                "l2_leaf_reg": [1, 3, 5, 7],
                "iterations": [200, 500, 1000],
                "bagging_temperature": [0, 1, 2],
            }
            cv = RepeatedStratifiedKFold(
                n_splits=5, n_repeats=2, random_state=random_state
            )
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring="roc_auc",
                cv=cv,
                random_state=random_state,
                n_jobs=1,
                verbose=0,
            )
            logger.info("Start hyperparameter search for CatBoost...")
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_auc = search.best_score_
            best_params = search.best_params_  # noqa: F841
            logger.info("Best ROC-AUC: %.4f", best_auc)

            # Логируем результаты поиска
        try:
            mlflow.log_artifact(save_path, artifact_path="models")
        except Exception as e:
            logger.warning(f"Skipping mlflow.log_artifact due to error: {e}")

        # Оценка на тесте
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        mlflow.log_metrics(
            {
                "test_roc_auc": roc_auc_score(y_test, proba),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1": f1_score(y_test, y_pred),
            }
        )

        # Сохраняем модель на диск
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        logger.info("Model saved to %s", save_path)

        # Логируем артефакт модели в MLflow
        mlflow.log_artifact(save_path, artifact_path="models")

        return model


if __name__ == "__main__":
    # 1) Проверяем, что данные есть
    proc_path = "data/processed/data.csv"
    if not os.path.exists(proc_path):
        raise RuntimeError(
            f"Processed data not found: {proc_path}. "
            "Сначала выполните: python -m src.preprocessing.process_data"
        )

    # 2) Загружаем X и y
    df = pd.read_csv(proc_path, encoding="utf-8-sig")
    X = df.drop(columns=["id", "is_agency"])
    y = df["is_agency"]

    # 3) Запускаем обучение (внутри MLflow)
    train_catboost(
        X,
        y,
        save_path="models/catboost_model.pkl",
        param_search=True,
        n_iter=20,
        random_state=42,
    )

    print("✅ Обучение завершено. Проверьте папку mlruns/agency_classifier_catboost/")
