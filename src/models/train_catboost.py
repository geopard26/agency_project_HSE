import os

import joblib
import numpy as np  # noqa: F401
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score  # noqa: F401
from sklearn.metrics import f1_score  # noqa: F401
from sklearn.metrics import precision_score  # noqa: F401
from sklearn.metrics import recall_score  # noqa: F401
from sklearn.metrics import roc_auc_score  # noqa: F401; noqa: F401
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)


def train_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    save_path: str = "models/catboost_model.pkl",
    param_search: bool = True,
    n_iter: int = 20,
) -> CatBoostClassifier:
    """
    Обучает CatBoost:
     - если param_search=True, подбирает гиперпараметры
     через RandomizedSearchCV с n_iter итераций;
     - иначе просто .fit() базовой модели.
    Сохраняет обученную модель в save_path и возвращает её.
    """
    # проверка размеров
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class weights
    counts = y_train.value_counts().to_dict()
    ratio = counts[0] / counts[1] if counts.get(1, 0) else 1.0
    class_weights = [1, ratio]

    base_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        class_weights=class_weights,
        od_type="Iter",
        od_wait=50,
        verbose=False,
    )

    if param_search:
        param_dist = {
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5, 7],
            "iterations": [200, 500, 1000],
            "bagging_temperature": [0, 1, 2],
        }
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        base_model.fit(X_train, y_train)
        model = base_model

    # сохраняем
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(model, save_path)

    return model
