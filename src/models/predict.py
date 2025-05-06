import os

import joblib
import pandas as pd

from src.preprocessing.process_data import clean_and_feature_engineer

# ————————————
# Константы
MODEL_PATH = "models/catboost_model.pkl"
PROCESSED_CSV = "data/processed/data.csv"
THRESHOLD = 0.754  # precision ≥ 0.80
# ————————————

# Считаем хедер processed CSV, уберём id и целевой столбец, оставим только фичи
_processed_cols = pd.read_csv(PROCESSED_CSV, nrows=0).columns.tolist()
FEATURE_NAMES = [c for c in _processed_cols if c not in ("id", "is_agency")]

# Кэш модели в памяти
_model = None


def load_model(path: str | None = None):
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        _model = joblib.load(path)
    return _model


def predict_one(raw_features: dict) -> tuple[float, int]:
    """
    На вход — словарь {feature_name: value, ...} (без 'id' и 'is_agency'),
    возвращает (proba, label) для класса 1 по порогу THRESHOLD.
    """
    # 1) Оборачиваем в DataFrame
    df = pd.DataFrame([raw_features])

    # 2) Очищаем и генерируем признаки
    df_proc = clean_and_feature_engineer(df)

    # 3) Выравниваем колонки по обученным фичам, добавляем нули, где не хватает
    df_proc = df_proc.reindex(columns=FEATURE_NAMES, fill_value=0)

    # 4) Загрузка модели и предсказание
    model = load_model()
    probs = model.predict_proba(df_proc)
    proba = probs[0][1]
    label = int(proba >= THRESHOLD)

    return proba, label


if __name__ == "__main__":
    # Пример: возьмём первый готовый sample из processed CSV
    sample = pd.read_csv(PROCESSED_CSV, nrows=1).to_dict(orient="records")[0]
    raw = {k: v for k, v in sample.items() if k not in ("id", "is_agency")}

    proba, label = predict_one(raw)
    print(f"Probability of agency: {proba: .4f}, Label: {label}")
