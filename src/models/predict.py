import os

import joblib
import pandas as pd

from src.preprocessing.process_data import clean_and_feature_engineer


def load_model(path: str = None):
    """
    Загружает модель+threshold из joblib-файла.
    Если path=None, ищет models/catboost_with_threshold.pkl в корне проекта.
    """
    if path is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        path = os.path.join(project_root, "models", "catboost_with_threshold.pkl")
    data = joblib.load(path)
    return data["model"], data["threshold"]


# один раз при импорте
MODEL, THRESHOLD = load_model()

# Список признаков модели — нужен для упорядочивания в predict_df
FEATURE_NAMES = list(MODEL.feature_names_)


def predict_df(df_raw: pd.DataFrame, model=None, threshold=None) -> pd.DataFrame:
    """
    На вход — сырые DataFrame.
    На выход — DataFrame с добавленными колонками:
      - agency_proba
      - agency_label
    """
    if model is None:
        model = MODEL
    if threshold is None:
        threshold = THRESHOLD

    # 1) Применяем чистку и фичеринжиниринг
    df_proc = clean_and_feature_engineer(df_raw.copy())

    # 2) Убираем id/целевой столбец, если есть
    drop_cols = [c for c in ("id", "is_agency") if c in df_proc.columns]
    X = df_proc.drop(columns=drop_cols)

    # 2.1) Приводим к тому же набору и порядку фич, что при обучении
    X = X.reindex(columns=FEATURE_NAMES, fill_value=0)

    # 3) Делаем предсказания
    proba = model.predict_proba(X)[:, 1]
    label = (proba >= threshold).astype(int)

    # 4) Собираем финальный DataFrame
    df_out = df_raw.copy()
    df_out["agency_proba"] = proba
    df_out["agency_label"] = label
    return df_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict agency for raw CSV")
    parser.add_argument(
        "-i", "--input-csv", required=True, help="сырой CSV (utf-8-sig)"
    )
    parser.add_argument(
        "-m", "--model-path", default=None, help="путь к модели+threshold"
    )
    parser.add_argument(
        "-o", "--output-csv", default=None, help="куда сохранить результаты"
    )
    args = parser.parse_args()

    model, threshold = load_model(args.model_path)
    df_raw = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    df_pred = predict_df(df_raw, model, threshold)

    if args.output_csv:
        df_pred.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"Predictions saved to {args.output_csv}")
    else:
        print(df_pred[["agency_proba", "agency_label"]].head())
