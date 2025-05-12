import os

import joblib
from catboost import CatBoostClassifier


def train_catboost(
    X, y, save_path: str = "models/catboost_model.pkl", param_search: bool = False
):
    """
    Обучает CatBoost (без RandomizedSearchCV, если param_search=False),
    сохраняет модель в save_path и возвращает её.
    """
    # 1) Обучение простой модели
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    ).fit(X, y)

    # 2) Сохраняем на диск
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    joblib.dump(model, save_path)

    return model
