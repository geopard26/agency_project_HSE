import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# 1) Загрузка данных
df = pd.read_csv("data/processed/data.csv")

# 2) Сплит признаков и цели
X = df.drop(columns=["id", "is_agency"])
y = df["is_agency"]

# 3) Разбиение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5) Предсказания
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 6) Вычисление метрик
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

# 7) Детальный отчёт
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
