import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.models.predict import FEATURE_NAMES, THRESHOLD  # порог 0.754
from src.preprocessing.process_data import clean_and_feature_engineer


def main():
    # 1) Загрузка и разделение
    df = pd.read_csv("data/processed/data.csv", encoding="utf-8-sig")
    X = df.drop(columns=["id", "is_agency"])
    y = df["is_agency"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2) Предобработка (на всякий случай)
    X_test = clean_and_feature_engineer(X_test.copy())
    X_test = X_test.reindex(columns=FEATURE_NAMES, fill_value=0)

    # 3) Загрузка модели
    model = joblib.load("models/catboost_model.pkl")

    # 4) Предсказания
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    # 5) Метрики
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print("=== Основные метрики для CatBoost-модели ===")
    for name, val in metrics.items():
        print(f" {name: >10s} : {val: .4f}")


if __name__ == "__main__":
    main()
