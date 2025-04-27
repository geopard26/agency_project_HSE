import os
import argparse

import pandas as pd
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

# Подгрузка .env
load_dotenv()

from src.db.session import SessionLocal, init_db
from src.db.models import Profile

def populate_from_csv(csv_path: str, label_col: str, id_col: str = "id"):
    """
    Читает CSV и добавляет/обновляет записи в таблицу profiles.
    features хранится как JSON (остальные колонки, кроме id и label).
    """
    # Создаём схему, если ещё не создана
    init_db()

    # Читаем CSV
    df = pd.read_csv(csv_path)

    db = SessionLocal()
    for _, row in df.iterrows():
        user_id = str(row[id_col])
        label = bool(row[label_col])

        # Формируем JSON-словарь признаков
        features = {
            col: row[col]
            for col in df.columns
            if col not in (id_col, label_col)
        }

        # Вставка или обновление
        obj = db.query(Profile).filter_by(user_id=user_id).first()
        if not obj:
            obj = Profile(user_id=user_id, features=features, label=label)
            db.add(obj)
        else:
            obj.features = features
            obj.label = label

        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            print(f"Пропущен дубликат user_id={user_id}")

    db.close()
    print("✅ Заполнение базы из CSV завершено.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate profiles table from CSV"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Путь до CSV-файла, например: data.csv"
    )
    parser.add_argument(
        "--label-col", required=True,
        help="Имя колонки с меткой (0/1 или True/False)"
    )
    parser.add_argument(
        "--id-col", default="id",
        help="Имя колонки с user_id (по умолчанию 'id')"
    )
    args = parser.parse_args()

    populate_from_csv(args.csv, args.label_col, args.id_col)

