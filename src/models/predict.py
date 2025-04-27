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
    """
    Принимает словарь исходных признаков (те же колонки, что в data/raw/data.csv),
    возвращает вероятность is_agency == 1.
    """
    # 1) Оборачиваем в DataFrame
    df = pd.DataFrame([raw_features])

    # 2) Применяем ту же логику обработки, что и к большому датасету
    df_proc = clean_and_feature_engineer(df)

    # 3) Загружаем модель и предсказываем
    model = load_model()
    proba = model.predict_proba(df_proc)[0, 1]
    return proba

if __name__ == "__main__":
    # Пример: тестовые данные из первой строки processed CSV
    sample = pd.read_csv("data/processed/data.csv", nrows=1).to_dict(orient="records")[0]
    # Из sample нужно убрать ключи id и is_agency,
    # чтобы они не мешали чистке и предсказанию:
    raw = {k: v for k, v in sample.items() if k not in ("id", "is_agency")}
    print("Probability of agency:", predict_one(raw))

