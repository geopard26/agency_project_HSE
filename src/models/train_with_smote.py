import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def train_and_evaluate():
    # — 1) Загрузка и подготовка данных
    df = pd.read_csv("data/processed/data.csv")
    X = df.drop(columns=["id", "is_agency"])
    # оставляем только числовые признаки
    X = X.select_dtypes(include=[np.number])
    y = df["is_agency"]

    # — 2) Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # — 3) Pipeline: imputation → SMOTEENN → Random Forest
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("smoteenn", SMOTEENN(random_state=42)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )

    # — 4) Обучение
    pipeline.fit(X_train, y_train)

    # — 5) Предсказания probs и дефолтные метки (threshold=0.5)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    print("=== Отчет при threshold=0.5 ===")
    print(classification_report(y_test, y_pred_default))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba): .4f}")

    # — 6) Подбор порога по максимальному F1
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
    # синхронизируем длины
    thresholds = np.append(thresholds, 1.0)
    f1_scores = (
        2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-9)
    )
    best_idx = np.nanargmax(f1_scores[:-1])
    best_thr = thresholds[best_idx]
    print(f"\nBest threshold by F1: {best_thr: .3f}, F1: {f1_scores[best_idx]: .3f}")
    y_pred_f1 = (y_proba >= best_thr).astype(int)
    print("\n=== Отчет при пороге max-F1 ===")
    print(classification_report(y_test, y_pred_f1))

    # — 7) Топ-5 порогов по precision
    candidates = []
    for thr in np.unique(y_proba):
        preds = (y_proba >= thr).astype(int)
        if preds.sum() == 0:
            continue
        p = precision_score(y_test, preds)
        r = recall_score(y_test, preds)
        f = f1_score(y_test, preds)
        candidates.append((thr, p, r, f))
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]

    print("\nTop-5 thresholds by precision:")
    for thr, p, r, f in candidates:
        print(f" thr={thr: .3f} → precision={p: .3f}, recall={r: .3f}, f1={f: .3f}")


if __name__ == "__main__":
    train_and_evaluate()
