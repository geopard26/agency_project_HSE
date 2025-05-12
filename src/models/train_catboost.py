import os

import joblib
import mlflow
import mlflow.sklearn
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


def train_catboost(
    X, y, save_path: str = "models/catboost_model.pkl", param_search: bool = True
):
    """
    Обучает CatBoost на X, y, сохраняет модель в save_path,
    логирует параметры и метрики в MLflow.
    """
    # 1) Разбиение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2) Class weights
    counts = y_train.value_counts().to_dict()
    weight_ratio = counts[0] / counts[1]
    class_weights = [1, weight_ratio]

    # 3) Базовая модель
    base_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        class_weights=class_weights,
        od_type="Iter",
        od_wait=50,
        verbose=False,
    )

    # 4) Подготовка поиска по гиперпараметрам
    param_distributions = {
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5, 7],
        "iterations": [200, 500, 1000],
        "bagging_temperature": [0, 1, 2],
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    # 5) MLflow-запуск
    mlflow.set_experiment("agency_classifier_catboost")
    with mlflow.start_run():
        # лог параметров
        mlflow.log_param("n_samples", len(y))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("class_weight_ratio", weight_ratio)
        mlflow.log_param("param_search", param_search)

        # 6) Обучение
        if param_search:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=20,
                scoring="roc_auc",
                cv=cv,
                random_state=42,
                n_jobs=1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_roc_auc", search.best_score_)
        else:
            model = base_model.fit(X_train, y_train)

        # 7) Оценка на тесте
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_test, proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # 8) Сохраняем модель на диск
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)

        # 9) Логируем модель как артефакт
        mlflow.sklearn.log_model(model, artifact_path="model")

    return model
