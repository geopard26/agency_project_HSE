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


def load_model():
    """
    Загружает и кеширует модель из текущей константы MODEL_PATH.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(raw_features: dict) -> tuple[float, int]:
    """
    На вход — словарь признаков, возвращает (proba, label) по THRESHOLD.
    """
    df = pd.DataFrame([raw_features])
    df_proc = clean_and_feature_engineer(df)
    df_proc = df_proc.reindex(columns=FEATURE_NAMES, fill_value=0)

    model = load_model()
    probs = model.predict_proba(df_proc)
    # поддерживаем и numpy-матрицу, и список списков
    proba = probs[0][1]
    label = int(proba >= THRESHOLD)
    return proba, label


if __name__ == "__main__":
    # Пример: возьмём первый готовый sample из processed CSV
    sample = pd.read_csv(PROCESSED_CSV, nrows=1).to_dict(orient="records")[0]
    raw = {k: v for k, v in sample.items() if k not in ("id", "is_agency")}

    proba, label = predict_one(raw)
    print(f"Probability of agency: {proba: .4f}, Label: {label}")
