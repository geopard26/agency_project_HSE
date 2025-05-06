import pandas as pd
import pytest  # noqa: F401
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import src.db.populate_csv as populate_mod
import src.db.session as session_mod
from src.db.models import Profile


def setup_memory_db(monkeypatch):
    """
    Подменяет SessionLocal и engine в обоих модулях на in-memory SQLite
    и создаёт таблицы.
    """
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=eng)

    # подменяем и в session.py
    monkeypatch.setattr(session_mod, "engine", eng)
    monkeypatch.setattr(session_mod, "SessionLocal", Session)
    # и в populate_csv, чтобы init_db взял наш engine
    monkeypatch.setattr(populate_mod, "SessionLocal", Session)
    monkeypatch.setattr(populate_mod, "init_db", session_mod.init_db)

    # создаём структуру БД
    session_mod.init_db()
    return Session


def test_populate_from_csv_insert_and_update(monkeypatch, tmp_path):
    """
    Тестируем, что populate_from_csv:
      - вставляет новые записи,
      - обновляет существующие при повторном запуске.
    """
    # Настраиваем in-memory БД
    Session = setup_memory_db(monkeypatch)

    # Создаём тестовый CSV с двумя записями
    df = pd.DataFrame({"id": [1, 2], "label": [1, 0], "f1": [10, 20], "f2": ["a", "b"]})
    csv1 = tmp_path / "data1.csv"
    df.to_csv(csv1, index=False)

    # Первый прогон — вставка
    populate_mod.populate_from_csv(str(csv1), label_col="label", id_col="id")

    sess = Session()
    profiles = sess.query(Profile).order_by(Profile.user_id).all()
    assert len(profiles) == 2

    p1, p2 = profiles
    assert p1.user_id == "1" and p1.label is True
    assert p1.features["f1"] == 10 and p1.features["f2"] == "a"
    assert p2.user_id == "2" and p2.label is False

    # Изменяем CSV: меняем метки и фичи
    df2 = pd.DataFrame(
        {"id": [1, 2], "label": [0, 1], "f1": [100, 200], "f2": ["x", "y"]}
    )
    csv2 = tmp_path / "data2.csv"
    df2.to_csv(csv2, index=False)

    # Второй прогон — обновление существующих записей
    sess.close()
    populate_mod.populate_from_csv(str(csv2), label_col="label", id_col="id")
    sess2 = Session()
    profiles_updated = sess2.query(Profile).order_by(Profile.user_id).all()
    u1, u2 = profiles_updated
    assert u1.label is False and u1.features["f1"] == 100 and u1.features["f2"] == "x"
    assert u2.label is True and u2.features["f1"] == 200 and u2.features["f2"] == "y"

    sess.close()
