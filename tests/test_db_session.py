import pytest  # noqa: F401
from sqlalchemy import create_engine, inspect  # noqa: F401
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool  # noqa: F401

import src.db.session as session_mod
from src.db.models import Base  # noqa: F401


def test_models_metadata_registered():
    """
    Проверяем, что в метаданных SQLAlchemy зарегистрирована таблица 'profiles'.
    """
    # Base.metadata.tables – dict с ключами-именами всех таблиц
    assert "profiles" in Base.metadata.tables


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
