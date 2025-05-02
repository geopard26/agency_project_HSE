import os

import joblib
import pandas as pd

from src.preprocessing.process_data import clean_and_feature_engineer

# Кэш модели в памяти
_model = None


def load_model(path: str = "models/random_forest.pkl"):
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        _model = joblib.load(path)
    return _model


def predict_one(raw_features: dict) -> float:
    # 1) Оборачиваем в DataFrame
    df = pd.DataFrame([raw_features])

    # 2) Очищаем и генерируем признаки
    df_proc = clean_and_feature_engineer(df)

    # 3) Подгружаем модель
    model = load_model()

    # 4) Выравниваем колонки: оставляем только те, что в model.feature_names_in_
    feature_names = model.feature_names_in_
    df_proc = df_proc.reindex(columns=feature_names, fill_value=0)

    # 5) Предсказываем вероятность
    THRESHOLD = 0.98  # выбрали по вашим результатам

    proba = model.predict_proba(df_proc)[0, 1]
    label = int(proba >= THRESHOLD)
    return proba, label


if __name__ == "__main__":
    # Пример: тестовые данные из первой строки processed CSV
    sample = pd.read_csv("data/processed/data.csv", nrows=1).to_dict(orient="records")[
        0
    ]
    # Из sample нужно убрать ключи id и is_agency,
    # чтобы они не мешали чистке и предсказанию:
    raw = {k: v for k, v in sample.items() if k not in ("id", "is_agency")}
    print("Probability of agency:", predict_one(raw))
