import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Profile


@pytest.fixture(scope="module")
def engine():
    """In-memory SQLite engine."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture(scope="module")
def connection(engine):
    """Соединение для всего модуля тестов."""
    conn = engine.connect()
    yield conn
    conn.close()


@pytest.fixture
def setup_db(connection):
    """
    Создаёт все таблицы перед тестом и возвращает сессию.
    После теста сессия закрывается.
    """
    Base.metadata.create_all(bind=connection)
    Session = sessionmaker(bind=connection)
    session = Session()
    yield session
    session.close()


def test_profile_crud_operations(setup_db):
    session = setup_db

    # --- INSERT ---
    p = Profile(user_id="user1", features={"a": 1}, label=True)
    session.add(p)
    session.commit()
    assert p.id is not None

    # --- SELECT & DEFAULT created_at ---
    fetched = session.query(Profile).filter_by(user_id="user1").one()
    assert isinstance(fetched.created_at, datetime.datetime)
    assert fetched.features == {"a": 1}
    assert fetched.label is True

    # --- UPDATE ---
    fetched.features = {"b": 2}
    fetched.label = False
    session.commit()

    updated = session.query(Profile).get(fetched.id)
    assert updated.features == {"b": 2}
    assert updated.label is False

    # --- DELETE ---
    session.delete(updated)
    session.commit()
    count = session.query(Profile).filter_by(user_id="user1").count()
    assert count == 0


def test_user_id_unique_constraint(setup_db):
    session = setup_db

    # Первый объект с user_id="user2" вставляется нормально
    p1 = Profile(user_id="user2", features={}, label=False)
    session.add(p1)
    session.commit()

    # Второй с таким же user_id должен вызвать IntegrityError
    p2 = Profile(user_id="user2", features={}, label=True)
    session.add(p2)
    with pytest.raises(IntegrityError):
        session.commit()
