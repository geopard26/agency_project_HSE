#!/usr/bin/env python3
import multiprocessing as mp

import mlflow
import mlflow.sklearn

from src.logging_config import get_logger, setup_logging

setup_logging()  # уровень по умолчанию INFO
logger = get_logger(__name__)

mlflow.set_experiment("agency_classifier_catboost")

# Устанавливаем метод старта "fork" для избежания ошибок resource_tracker на macOS
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Оценивает модель на X_test, y_test с учётом заданного порога:
      - threshold: значение от 0 до 1, при котором proba>=threshold даёт метку 1.
    Выводит:
      - ROC-AUC (на вероятностях)
      - accuracy, precision, recall, f1 (на дискретных предсказаниях)
      - classification report
    """
    # вместо y_pred = model.predict(X_test):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print(f"=== Метрики при threshold = {threshold: .3f} ===")
    for name, val in metrics.items():
        print(f" {name: >10s} : {val: .4f}")


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
    Если param_search=True — делает RandomizedSearchCV, иначе просто fit.
    Сохраняет модель в save_path и возвращает её.
    """
    # Запускаем MLflow-run
    with mlflow.start_run():
        # Логируем входные параметры
        mlflow.log_param("param_search", param_search)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_iter", n_iter)
    # 1) Сплит для поиска гиперпараметров
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    mlflow.log_param("train_size", len(y_train))
    mlflow.log_param("test_size", len(y_test))

    # 2) Считаем веса классов
    counts = y_train.value_counts().to_dict()
    weight_ratio = counts.get(0, 1) / (counts.get(1, 1) or 1)
    class_weights = [1, weight_ratio]
    mlflow.log_param("class_weight_ratio", weight_ratio)

    # 3) Базовая модель
    base_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        class_weights=class_weights,
        od_type="Iter",
        od_wait=50,
        verbose=False,
    )

    # 4) Опциональный RandomizedSearchCV
    if param_search:
        param_distributions = {
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5, 7],
            "iterations": [200, 500, 1000],
            "bagging_temperature": [0, 1, 2],
        }
        mlflow.log_params({k: None for k in param_distributions})
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_state)
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
        logger.info("Start hyperparameter search for CatBoost...")
        search.fit(X_train, y_train)

        model = search.best_estimator_
        logger.info("Best ROC-AUC: %.4f", search.best_score_)
        best_auc = search.best_score_
        best_params = search.best_params_
        logger.info("Best ROC-AUC: %.4f", best_auc)

        # Логируем результаты поиска
        mlflow.log_metric("best_cv_roc_auc", best_auc)
        mlflow.log_params(best_params)
    else:
        logger.info("Training CatBoost without hyperparameter search...")
        model = base_model.fit(X_train, y_train)

    # Оцениваем на тесте и логируем метрики
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metrics(
        {
            "test_roc_auc": auc,
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
        }
    )

    # 5) Сохраняем
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    logger.info("CatBoost model saved to %s", save_path)
    # Логируем модель как артефакт
    mlflow.log_artifact(save_path, artifact_path="models")

    return model


if __name__ == "__main__":
    # 1) Загрузка обработанных данных
    proc_path = "data/processed/data.csv"
    if not os.path.exists(proc_path):
        raise RuntimeError(
            f"Processed data not found at {proc_path}. "
            "Сначала выполните: python -m src.preprocessing.process_data"
        )
    df = pd.read_csv(proc_path, encoding="utf-8-sig")

    X = df.drop(columns=["id", "is_agency"])
    y = df["is_agency"]

    # 2) Разбиение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3) Вычисление весов классов для борьбы с дисбалансом
    counts = y_train.value_counts().to_dict()
    weight_ratio = counts[0] / counts[1]
    class_weights = [1, weight_ratio]
    print(f"Class weights:  {{0: 1, 1: {weight_ratio: .2f}}}")

    # 4) Инициализация модели CatBoost
    base_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        class_weights=class_weights,
        od_type="Iter",
        od_wait=50,
        verbose=False,
    )

    # 5) Подбор гиперпараметров через RandomizedSearchCV
    param_distributions = {
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5, 7],
        "iterations": [200, 500, 1000],
        "bagging_temperature": [0, 1, 2],
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=1,  # один процесс для избежания ошибок resource_tracker
    )

    print("Start hyperparameter search for CatBoost...")
    search.fit(X_train, y_train)
    print(f"Best ROC-AUC: {search.best_score_: .4f}")
    print("Best parameters:", search.best_params_)

    # 6) Оценка лучшей модели на стандартном пороге 0.5
    best_model = search.best_estimator_
    evaluate_model(best_model, X_test, y_test)

    # 7) Подбор порога для precision
    probs = best_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # 7.1) Порог, максимизирующий F1
    best_idx = np.nanargmax(f1_scores[:-1])
    best_thresh = thresholds[best_idx]
    print(
        f"\nOptimal threshold for max F1: {best_thresh: .3f}",
        f"F1={f1_scores[best_idx]: .3f}",
    )

    # 7.2) Порог для минимизации ложных срабатываний (precision >= target_precision)
    target_precision = 0.80
    mask = precisions >= target_precision
    valid_thresholds = thresholds[mask[:-1]]
    if valid_thresholds.size > 0:
        chosen_thresh = valid_thresholds.min()
        print(f"threshold precision >= {target_precision: .2f}: {chosen_thresh: .3f}")
    else:
        chosen_thresh = best_thresh
        print(
            f"No threshold achieves precision >= {target_precision: .2f}",
            f"using best_thresh {chosen_thresh: .3f}",
        )

    # 7.3) Оценка метрик при различных порогах
    def eval_at_threshold(thresh):
        y_pred = (probs >= thresh).astype(int)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }

    for name, t in [
        ("0.5 (default)", 0.5),
        ("max F1", best_thresh),
        (f"precision>= {target_precision: .2f}", chosen_thresh),
    ]:
        metrics = eval_at_threshold(t)
        print(f"\nThreshold = {t: .3f} ({name}): ")
        for m, v in metrics.items():
            print(f"  {m: >9s}: {v: .4f}")

    # 8) Сохранение модели
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "catboost_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Модель сохранена в {model_path}")
    logger.info("=== Основные метрики ===")
