import pytest  # noqa: F401
from sqlalcchemy.pool import StaticPool
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

import src.db.session as session_mod
from src.db.models import Base  # noqa: F401


def test_init_db_creates_tables(monkeypatch):
    """
    Проверяем, что init_db создаёт таблицы из моделей в памяти.
    """
    # 1) Создаём in-memory движок
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    # 2) Подменяем module-level engine в session.py
    monkeypatch.setattr(session_mod, "engine", eng)

    # 3) Проверяем, что до init_db таблиц нет
    insp = inspect(eng)
    assert insp.get_table_names() == []

    # 4) init_db должна создать таблицу profiles
    session_mod.init_db()
    tables = insp.get_table_names()
    assert "profiles" in tables


def test_sessionlocal_returns_working_session(monkeypatch):
    """
    SessionLocal дает валидную сессию, на которой можно выполнять простые запросы.
    """
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=eng)
    monkeypatch.setattr(session_mod, "SessionLocal", Session)

    sess = session_mod.SessionLocal()
    # Проверка, что на сессии можно выполнить простой SQL
    result = sess.execute(text("SELECT 1")).scalar()
    assert result == 1
    sess.close()
