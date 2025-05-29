import ast
import os

import pandas as pd

from src.logging_config import get_logger


def load_raw(path="data/raw/data.csv"):
    # читаем с BOM, чтобы кириллица не «ломалась»
    return pd.read_csv(path, encoding="utf-8-sig")


logger = get_logger(__name__)


def clean_and_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Заполнить пропуски
    logger.debug("Running clean_and_feature_engineer on %d rows", len(df))

    def try_parse(x):
        if isinstance(x, str) and (x.startswith("{") or x.startswith("[")):
            try:
                return ast.literal_eval(x)
            except Exception:
                return x
        return x

    for col in ["country", "city", "home_town", "site", "relation", "langs"]:
        if col in df.columns:
            df[col] = df[col].apply(try_parse)

    for col in ["home_phone", "games", "inspired_by"]:
        if col in df:
            df[col] = df[col].fillna(0)
    # НОВАЯ ЛОГИКА для контактов из VK API:
    contact_cols = [
        c for c in ["country", "city", "home_town", "site", "relation"] if c in df
    ]

    def has_contact(val):
        # VK API: dict {"id":…, "title":…}
        if isinstance(val, dict):
            val = val.get("title")
        if pd.isna(val):
            return 0
        s = str(val).strip()
        return 0 if (not s or s == "0") else 1

    if contact_cols:
        df["index_contacts"] = df[contact_cols].applymap(has_contact).mean(axis=1)

    # НОВАЯ ЛОГИКА для langs из VK API (список словарей):
    if "langs" in df:
        # langs у VK API = [{'id':…, 'title':'English'}, …]
        def extract_langs(cell):
            if isinstance(cell, list):
                return [d.get("title") for d in cell if isinstance(d, dict)]
            return []

        df["langs_list"] = df["langs"].apply(extract_langs)
        unique_langs = {lang for lst in df["langs_list"] for lang in lst}
        for lang in unique_langs:
            df[f"lang_{lang}"] = df["langs_list"].apply(lambda lst: int(lang in lst))
        df.drop(columns=["langs", "langs_list"], inplace=True)
    # 2. Бинаризовать телефоны
    for col in ["mobile_phone", "home_phone"]:
        if col in df:
            df[col] = df[col].apply(
                lambda x: 1 if pd.notna(x) and str(x).strip() and str(x) != "0" else 0
            )
    # 3. Новый признак index_contacts
    contact_cols = [
        "country",
        "city",
        "home_town",
        "mobile_phone",
        "home_phone",
        "relation",
        "site",
    ]
    existing = [c for c in contact_cols if c in df]
    if existing:
        # функция, которая определяет, считается ли контакт «наличием»
        def has_contact(val):
            if pd.isna(val):
                return 0
            s = str(val).strip()
            # пуcтота или "0" не считаются
            if not s or s == "0":
                return 0
            return 1

        # для каждой строки считаем долю полей, где has_contact=1
        df["index_contacts"] = df[existing].applymap(has_contact).mean(axis=1)

    # 4. Развёртка языков langs → lang_<язык>
    if "langs" in df:
        langs_series = (
            df["langs"]
            .dropna()
            .astype(str)
            .str.strip("[]")
            .str.replace("'", "")
            .str.split(", ")
        )
        unique_langs = {
            lan for sub in langs_series for lan in sub if isinstance(sub, list)
        }
        for lang in unique_langs:
            df[f"lang_{lang}"] = df["langs"].apply(
                lambda x: 1 if isinstance(x, str) and lang in x else 0
            )

    # 5. Университет (1/0)
    if "university_name" in df:
        df["university_name_binary"] = df["university_name"].apply(
            lambda x: 1 if pd.notna(x) and str(x).strip() and str(x) != "0" else 0
        )

    # 6. Удаляем ненужные текстовые колонки
    drop_cols = ["site", "about", "quotes", "inspired_by", "langs", "university_name"]
    for c in drop_cols:
        if c in df:
            df.drop(columns=c, inplace=True)

    drop_cols = [...]
    for c in drop_cols:
        if c in df:
            df.drop(columns=c, inplace=True)

    # Вот это вставляем:
    df = df.select_dtypes(include=["number"])

    return df


def save_processed(df: pd.DataFrame, path="data/processed/data.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Processed data saved to {path}")


def run_pipeline():
    raw = load_raw()
    proc = clean_and_feature_engineer(raw)
    save_processed(proc)


if __name__ == "__main__":
    run_pipeline()
