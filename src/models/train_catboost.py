import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)

# 1. Загрузка и разделение «train / val / test»
df = pd.read_csv("data/processed/data.csv", encoding="utf-8-sig")
X = df.drop(columns=["id", "is_agency"])
y = df["is_agency"]

# 20% — тест, из оставшихся 25% — валидация
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
)

# 2. Посчитаем веса — редкому классу больший
counts = y_train.value_counts().to_dict()
ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
class_weights = {0: 1, 1: ratio}

# 3. Инициализация модели
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    class_weights=class_weights,
    od_type="Iter",  # overfitting detector
    od_wait=50,
    verbose=100,
)

# 4. Гиперпараметрический поиск с eval_set для ранней остановки
param_dist = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [1, 3, 5, 7],
    "iterations": [200, 500, 1000],
    "bagging_temperature": [0, 1, 2],
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=1,
    verbose=1,
)
# Передаём в RandomizedSearchCV параметры обучения
search.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

best_model = search.best_estimator_

# 5. Подбор порога на валидации
proba_val = best_model.predict_proba(X_val)[:, 1]
# 1) Считаем кривую
precisions, recalls, thresholds = precision_recall_curve(y_val, proba_val)

# 2) Так как thresholds на 1 короче, чем precisions:
#    берем все, кроме последнего значения precision
precisions_for_thresh = precisions[:-1]

# 3) Ищем индексы, где precision >= целевое
desired_precision = 0.80
mask = precisions_for_thresh >= desired_precision

# 4) Если таких нет — ставим разумный дефолт (например, 0.5)
if not mask.any():
    best_threshold = 0.5
else:
    # выбираем самый высокий threshold, дающий нужный precision
    best_threshold = float(thresholds[mask][-1])


# 6. Финальный фит на полной тренировочной части и проверка на тесте
best_model.fit(
    X_train_full,
    y_train_full,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    verbose=100,
)
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"Test AUC: {test_auc: .4f}")

# 7. Сохраняем и модель, и порог
joblib.dump(
    {"model": best_model, "threshold": best_threshold},
    "models/catboost_with_threshold.pkl",
)

print(f"Selected threshold for precision ≥ {desired_precision}: {best_threshold: .3f}")
