import os  # noqa: F401

import pytest
from dotenv import load_dotenv
from sqlalchemy import inspect

from src.db.models import Base, Profile
from src.db.session import SessionLocal, engine, init_db


@pytest.fixture(scope="module", autouse=True)
def postgres_db():
    # подгружаем .env.dev
    load_dotenv(".env.dev")
    # создаём схему
    init_db()
    yield
    # сбрасываем в конце
    Base.metadata.drop_all(bind=engine)


def test_init_db_creates_profiles_table():
    insp = inspect(engine)
    assert "profiles" in insp.get_table_names()


def test_profile_crud_operations():
    session = SessionLocal()
    # вставка
    p = Profile(user_id="testuser", features={"foo": 1}, label=True)
    session.add(p)
    session.commit()

    # выборка
    fetched = session.query(Profile).filter_by(user_id="testuser").one()
    assert fetched.label is True

    # обновление
    fetched.label = False
    session.commit()
    re_fetched = session.query(Profile).get(fetched.id)
    assert re_fetched.label is False

    # удаление
    session.delete(re_fetched)
    session.commit()
    assert session.query(Profile).filter_by(user_id="testuser").count() == 0

    session.close()
