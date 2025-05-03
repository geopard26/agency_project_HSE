import joblib  # pip install joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def evaluate_model(model, X_test, y_test):
    """
    Функция для предсказаний и печати метрик.
    """
    y_pred = model.predict(X_test)
    # у GBC тоже есть predict_proba
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print("=== Основные метрики ===")
    for name, val in metrics.items():
        print(f" {name: >10s} : {val: .4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # 1) Загрузка данных
    df = pd.read_csv("data/processed/data.csv")

    # 2) Сплит признаков и цели
    X = df.drop(columns=["id", "is_agency"])
    y = df["is_agency"]

    # 3) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load("models/hgb_model.pkl'")

    # 5–7) Оценка
    evaluate_model(model, X_test, y_test)
