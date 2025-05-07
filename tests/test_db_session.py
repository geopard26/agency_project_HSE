import pytest  # noqa: F401
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.db.session as session_mod
from src.db.models import Base  # noqa: F401


def test_init_db_creates_tables(monkeypatch):
    """
    Проверяем, что init_db создаёт таблицу 'profiles' в in-memory SQLite.
    Используем StaticPool, чтобы все соединения шли в одну БД.
    """
    # 1) Создаём in-memory движок с StaticPool
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # 2) Подменяем глобальный engine в session модуля
    monkeypatch.setattr(session_mod, "engine", eng)

    # 3) До init_db — нет ни одной таблицы
    insp = inspect(eng)
    assert insp.get_table_names() == []

    # 4) Вызываем init_db — должна появиться таблица profiles
    session_mod.init_db()
    tables = insp.get_table_names()
    assert "profiles" in tables


def test_sessionlocal_returns_working_session(monkeypatch):
    """
    Проверяем, что SessionLocal даёт рабочую сессию,
    на которой можно выполнить простой запрос.
    """
    from sqlalchemy import text

    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=eng)
    # Подменяем SessionLocal в модуле
    monkeypatch.setattr(session_mod, "SessionLocal", Session)

    sess = session_mod.SessionLocal()
    result = sess.execute(text("SELECT 1")).scalar()
    assert result == 1
    sess.close()
