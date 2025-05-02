import multiprocessing
import os

import joblib
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.preprocessing.process_data import run_pipeline


def train_model(
    raw_csv: str = "data/raw/data.csv",
    processed_csv: str = "data/processed/data.csv",
    model_path: str = "models/random_forest.pkl",
):
    # 1) Предобработка данных, если ещё не сделано
    if not os.path.exists(processed_csv):
        os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
        run_pipeline(raw_csv, processed_csv)

    # 2) Загрузка и разделение на X, y
    df = pd.read_csv(processed_csv)
    X = df.drop(columns=["id", "is_agency"])
    y = df["is_agency"]

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4) Модель и сетка гиперпараметров
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
        "class_weight": [None, "balanced"],
    }

    # 5) Grid Search с 5-fold стратифицированной кросс-валидацией
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    # 6) Запуск поиска лучшей модели
    with parallel_backend("threading"):
        grid.fit(X_train, y_train)

    # 7) Результаты
    print("Best parameters found:", grid.best_params_)
    print("Best cross-validated ROC-AUC:", grid.best_score_)

    # 8) Сохранение финальной модели
    best_model = grid.best_estimator_
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    train_model()
# import os

# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

# from src.preprocessing.process_data import run_pipeline


# def train_model(
#     processed_csv: str = "data/processed/data.csv",
#     model_path: str = "models/random_forest.pkl",
# ):
#     run_pipeline()
#     df = pd.read_csv(processed_csv, encoding="utf-8-sig")
#     X = df.drop(columns=["id", "is_agency"])
#     y = df["is_agency"]
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X, y)
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")
#     return model


# if __name__ == "__main__":
#     train_model()
