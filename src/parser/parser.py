import csv
import logging  # noqa: F401
import os
import time
from typing import Any, Dict, List, Optional

import vk_api
from dotenv import load_dotenv  # noqa: F401

from src.logging_config import get_logger  # noqa: F401

# 1) Конфигурация
logger = get_logger(__name__)
VK_TOKEN = os.getenv("VK_TOKEN")
PARSER_FIELDNAMES = [
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
    user_ids: List[str],
    vk_client: Optional[Any] = None,
    token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Забирает профили из VK API (или из переданного vk_client) и
    возвращает их в виде списка словарей, содержащих только поля из PARSER_FIELDNAMES.
    В случае падения маппинга на каком-то профиле кладёт __error=True.
    """
    # 1) Инициализируем API-клиент
    if vk_client is None:
        actual_token = token or VK_TOKEN
        logger.info("Connecting to VK API with token %s...", actual_token[:10] + "…")
        session = vk_api.VkApi(token=actual_token)
        vk = session.get_api()
    else:
        vk = vk_client

    users_info: List[Dict[str, Any]] = []
    batch_size = 100
    fields = (
        "bdate,city,country,home_town,contacts,education,site,"
        "friends_count,followers_count,activities,interests,"
        "music,movies,tv,books,games,about,quotes,career,military,occupation"
    )

    # 2) Делаем батчевые запросы
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i : i + batch_size]
        try:
            response = vk.users.get(user_ids=",".join(batch), fields=fields)
        except vk_api.exceptions.ApiError as e:
            logger.error("API error [%s]: %s", getattr(e, "code", ""), e)
            time.sleep(1)
            continue

        # 3) Проходим по каждому юзеру из ответа
        for user in response:
            try:
                # приводим id к int, если строка
                raw_id = user.get("id", user.get("user_id"))
                user["id"] = int(raw_id)
                # маппим только нужные поля
                row = map_user_to_row(user)
            except Exception:
                row = {
                    "id": int(user.get("id")) if user.get("id") is not None else None,
                    "__error": True,
                }
            users_info.append(row)

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
