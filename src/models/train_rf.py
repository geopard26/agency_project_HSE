import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.process_data import run_pipeline


def train_model(
    processed_csv: str = "data/processed/data.csv",
    model_path: str = "models/random_forest.pkl",
):
    run_pipeline()
    df = pd.read_csv(processed_csv, encoding="utf-8-sig")
    X = df.drop(columns=["id", "is_agency"])
    y = df["is_agency"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model


if __name__ == "__main__":
    train_model()
