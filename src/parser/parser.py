import csv
import logging
import os
import time
from typing import Any, Dict, List, Optional

import vk_api
from dotenv import load_dotenv

from src.logging_config import get_logger  # noqa: F401

# 1) Конфигурация
load_dotenv()  # подгрузит .env в os.environ
logger = logging.getLogger(__name__)
VK_TOKEN = os.getenv("VK_TOKEN")
# if not VK_TOKEN:
#     raise ValueError("VK_TOKEN не задан в .env")

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.INFO
)


# 2) Константа: поля CSV в порядке вывода
PARSER_FIELDNAMES: List[str] = [
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


def get_users_info(
    user_ids: List[str], vk_client: Optional[Any] = None, token: Optional[str] = None
) -> List[Dict[str, Any]]:
    actual_token = token or VK_TOKEN
    if vk_client is None and not actual_token:
        raise ValueError("VK_TOKEN не задан в .env и не передан параметр token")
    # 1) инициализация vk-клиента (либо используем мок из тестов)
    if vk_client is None:
        logger.info("Connecting to VK API with token %s…", actual_token)
        session = vk_api.VkApi(token=actual_token)
        vk = session.get_api()
    else:
        vk = vk_client

    users_info: List[Dict[str, Any]] = []
    fields = ",".join(
        [
            "bdate",
            "city",
            "country",
            "home_town",
            "contacts",
            "education",
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
    )

    for uid in user_ids:
        try:
            # Запрашиваем один user_id за раз
            response = vk.users.get(user_ids=uid, fields=fields)
            # VK API возвращает список: [] или [{...}]
            # VK всегда возвращает список: либо [{}], либо [{'deactivated':'deleted'}]
            if response and isinstance(response, list):
                user = response[0]
                # 1) если аккаунт удалён/забанен, придёт поле 'deactivated'
                if user.get("deactivated") is not None:
                    users_info.append({"id": uid, "__error": True})
                # 2) если нет нормального поля id или first_name — тоже считаем ошибкой
                elif not user.get("id") or not user.get("first_name"):
                    users_info.append({"id": uid, "__error": True})
                else:
                    # всё ок
                    users_info.append(user)
            else:
                users_info.append({"id": uid, "__error": True})
        except vk_api.exceptions.ApiError as e:
            logger.warning("Error fetching %s: %s", uid, e, exc_info=True)
            users_info.append({"id": uid, "__error": True})
            time.sleep(1)

    return users_info


def map_user_to_row(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует один словарь user (как из VK API) в «плоский» словарь
    для записи в CSV по PARSER_FIELDNAMES.
    """
    # city
    city_val = ""
    city = user.get("city")
    if isinstance(city, dict):
        city_val = city.get("title", "")
    elif user.get("home_town"):
        city_val = user.get("home_town", "")

    # country
    country_val = ""
    country = user.get("country")
    if isinstance(country, dict):
        country_val = country.get("title", "")

    # occupation
    occ = user.get("occupation")
    occ_val = occ.get("name", "") if isinstance(occ, dict) else ""

    # По умолчанию берем значение или пустую строку
    row = {
        "id": user.get("id", ""),
        "first_name": user.get("first_name", ""),
        "last_name": user.get("last_name", ""),
        "bdate": user.get("bdate", ""),
        "city": city_val,
        "country": country_val,
        "home_town": user.get("home_town", ""),
        "mobile_phone": user.get("mobile_phone", ""),
        "home_phone": user.get("home_phone", ""),
        "university_name": user.get("university_name", ""),
        "relation": user.get("relation", ""),
        "personal": user.get("personal", ""),
        "connections": user.get("connections", ""),
        "site": user.get("site", ""),
        "friends_count": user.get("friends_count", ""),
        "followers_count": user.get("followers_count", ""),
        "activities": user.get("activities", ""),
        "interests": user.get("interests", ""),
        "music": user.get("music", ""),
        "movies": user.get("movies", ""),
        "tv": user.get("tv", ""),
        "books": user.get("books", ""),
        "games": user.get("games", ""),
        "about": user.get("about", ""),
        "quotes": user.get("quotes", ""),
        "career": user.get("career", ""),
        "military": user.get("military", ""),
        "occupation": occ_val,
    }

    # Оставим ровно те ключи, что в PARSER_FIELDNAMES
    return {k: row.get(k) for k in PARSER_FIELDNAMES}


def save_to_csv(filename: str, users_info: List[Dict[str, Any]]) -> None:
    """
    Сохраняет список словарей users_info в CSV file filename.
    Заголовок записывается только если файл новый или пустой.
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    file_exists = os.path.isfile(filename)
    needs_header = not file_exists or os.path.getsize(filename) == 0

    with open(filename, "a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=PARSER_FIELDNAMES)
        if needs_header:
            writer.writeheader()
        for user in users_info:
            row = map_user_to_row(user)
            writer.writerow(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VK Parser: fetch user profiles and save to CSV"
    )
    parser.add_argument(
        "--user-ids",
        required=True,
        help="Список ID через запятую, например: id1,id2,id3",
    )
    parser.add_argument(
        "--output",
        default="data.csv",
        help="Путь до выходного CSV-файла",
    )
    args = parser.parse_args()

    ids = args.user_ids.split(",")
    users = get_users_info(ids)
    save_to_csv(args.output, users)
    logger.info(f"Данные сохранены в {args.output}")
