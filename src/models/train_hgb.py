import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import parallel_backend
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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


def evaluate_model(model, X_test, y_test):
    """
    Предсказание и печать основных метрик на тестовой выборке
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print("=== Основные метрики на тесте ===")
    for name, val in metrics.items():
        print(f" {name: >10s} : {val: .4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # 1) Загрузка данных
    df = pd.read_csv("data/processed/data.csv")
    X = df.drop(columns=["id", "is_agency"])
    y = df["is_agency"]

    # 2) Разбиение на train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3) Pipeline с SMOTE и классификатором
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),  # заполнение медианой
            ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.1,
                    max_depth=6,
                    early_stopping=True,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # 4) Настройка гиперпараметров через RandomizedSearchCV
    param_distributions = {
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__min_samples_leaf": [1, 3, 5],
        "model__max_bins": [255, 100, 50],
        "model__l2_regularization": [0.0, 1.0, 10.0],
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    # 5) Поиск лучших параметров
    with parallel_backend("threading"):
        search.fit(X_train, y_train)
    print("Best ROC-AUC: {:.4f}".format(search.best_score_))
    print("Best parameters:", search.best_params_)

    # 6) Оценка на тестовой выборке
    best_model = search.best_estimator_
    evaluate_model(best_model, X_test, y_test)

    # 7) Сохранение модели
    joblib.dump(best_model, "models/hgb_model.pkl")
    print("Модель сохранена в models/hgb_model.pkl")
