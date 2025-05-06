import logging

import vk_api

from src.parser import PARSER_FIELDNAMES, get_users_info, map_user_to_row, save_to_csv


def test_parser_fieldnames():
    """
    Убеждаемся, что константа PARSER_FIELDNAMES содержит правильные колонки
    и что их порядок не меняется.
    """
    expected = [
        "id",
        "first_name",
        "last_name",
        "bdate",
        "city",
        "country",
        "home_town",
        "mobile_phone",
        "home_phone",
        "university_name",
        "relation",
        "personal",
        "connections",
        "site",
        "friends_count",
        "followers_count",
        "activities",
        "interests",
        "music",
        "movies",
        "tv",
        "books",
        "games",
        "about",
        "quotes",
        "career",
        "military",
        "occupation",
    ]
    assert PARSER_FIELDNAMES == expected


def test_map_user_to_row_full():
    """
    Проверяем, что map_user_to_row выравнивает поля из вложенных словарей
    и возвращает ровно ключи из PARSER_FIELDNAMES.
    """
    user = {
        "id": "123",
        "first_name": "A",
        "last_name": "B",
        "bdate": "1.1.1990",
        "city": {"title": "C"},
        "country": {"title": "D"},
        "home_town": "HT",
        "mobile_phone": "11111",
        "home_phone": "22222",
        "university_name": "Uni",
        "relation": "3",
        "personal": "P",
        "connections": "conn",
        "site": "s",
        "friends_count": 5,
        "followers_count": 6,
        "activities": "act",
        "interests": "intr",
        "music": "m",
        "movies": "mov",
        "tv": "tv",
        "books": "b",
        "games": "g",
        "about": "a",
        "quotes": "q",
        "career": ["dev"],
        "military": ["none"],
        "occupation": {"name": "occ"},
    }

    row = map_user_to_row(user)

    # Ключи и порядок
    assert list(row.keys()) == PARSER_FIELDNAMES

    # Проверяем важные поля
    assert row["id"] == "123"
    assert row["city"] == "C"
    assert row["country"] == "D"
    assert row["home_town"] == "HT"
    assert row["occupation"] == "occ"


def test_map_user_to_row_missing(monkeypatch):
    """
    Если в user нет city и occupation, то в результирующем row будут пустые строки.
    """
    user = {"id": "1", "first_name": "X", "last_name": "Y"}
    row = map_user_to_row(user)

    assert row["id"] == "1"
    assert row["first_name"] == "X"
    assert row["city"] == ""  # не было city и home_town
    assert row["occupation"] == ""  # не было occupation


def test_get_users_info_success():
    """
    Передаём fake vk_client, возвращающий заранее заданный список словарей.
    Убедимся, что get_users_info возвращает этот список без изменений.
    """
    fake_users = [{"id": 1}, {"id": 2}]

    class FakeClient:
        @property
        def users(self):
            return type("U", (), {"get": staticmethod(lambda **kw: fake_users)})()

    result = get_users_info(["1", "2"], vk_client=FakeClient())
    assert result == fake_users


def test_get_users_info_api_error(caplog):
    """
    Передаём fake vk_client, в котором users.get бросает ApiError.
    Проверяем, что get_users_info возвращает пустой список
    и пишет warning в лог.
    """
    caplog.set_level(logging.WARNING)

    class FakeClientError:
        @property
        def users(self):
            def raiser(**kw):
                raise vk_api.exceptions.ApiError("test error")

            return type("U", (), {"get": staticmethod(raiser)})()

    result = get_users_info(["1"], vk_client=FakeClientError())
    assert result == []

    # Лог должен содержать warning об ApiError
    warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("API error" in str(w) for w in warnings)


def test_save_to_csv(tmp_path):
    """
    Проверяем, что save_to_csv:
      - создаёт файл с header из PARSER_FIELDNAMES
      - добавляет новые строки без повторного header
    """
    # User на вход — в формате, как возвращает VK API
    user = {
        "id": "10",
        "first_name": "A",
        "last_name": "B",
        "bdate": "",
        "city": {"title": "X"},
        "country": {"title": "Y"},
        "home_town": "",
        "mobile_phone": "123",
        "home_phone": "",
        "university_name": "",
        "relation": "",
        "personal": None,
        "connections": None,
        "site": "",
        "friends_count": 5,
        "followers_count": 0,
        "activities": "",
        "interests": "",
        "music": "",
        "movies": "",
        "tv": "",
        "books": "",
        "games": "",
        "about": "",
        "quotes": "",
        "career": [],
        "military": [],
        "occupation": {"name": ""},
    }
    out_file = tmp_path / "out.csv"

    # Первый вызов: header + 1 строка
    save_to_csv(str(out_file), [user])
    text = out_file.read_text(encoding="utf-8-sig").splitlines()
    # Header
    header = text[0].split(",")
    assert header == PARSER_FIELDNAMES
    # Одна строка данных
    assert len(text) == 1 + 1

    # Второй вызов: без дублирования header
    save_to_csv(str(out_file), [user])
    text2 = out_file.read_text(encoding="utf-8-sig").splitlines()
    assert len(text2) == 1 + 2  # header + 2 data-строки
