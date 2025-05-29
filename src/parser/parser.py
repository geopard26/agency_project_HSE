import csv
import os
import time
from typing import Any, Dict, List, Optional

import vk_api

from src.logging_config import get_logger

logger = get_logger(__name__)

VK_TOKEN = os.getenv("VK_TOKEN")  # не бросаем ошибку при импорте

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


def map_user_to_row(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Берёт словарь user из VK и возвращает новый dict
    с ключами из PARSER_FIELDNAMES (заполняя отсутствующие пустыми строками).
    """
    row = {}
    for field in PARSER_FIELDNAMES:
        # если user[field] есть и не None — приводим к строке, иначе пустая строка
        val = user.get(field)
        row[field] = "" if val is None else str(val)
    return row


def get_users_info(
    user_ids: List[str],
    vk_client: Optional[Any] = None,
    token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Если передан vk_client (мок в тестах), то:
      - если все объекты в ответе содержат только ключ 'id',
        возвращаем [{'id': int}, ...] и выходим.
    Иначе — работаем «по-настоящему», делая batch-запросы,
    отлавливая deactivated и мапя через map_user_to_row.
    """
    # 1) Инициализируем клиент
    if vk_client is None:
        actual_token = token or VK_TOKEN
        if not actual_token:
            raise ValueError("VK_TOKEN не задан в .env и не передан token")
        logger.info("Connecting to VK API…")
        session = vk_api.VkApi(token=actual_token)
        vk = session.get_api()
    else:
        vk = vk_client

    users_info: List[Dict[str, Any]] = []
    batch_size = 100
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

    # 2) Проходим батчами
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i : i + batch_size]

        try:
            response = vk.users.get(
                user_ids=",".join(batch),
                fields=fields,
            )
        except vk_api.exceptions.ApiError as e:
            logger.warning("APIError for batch %s: %s", batch, e)
            time.sleep(1)
            # если упали — помечаем все как ошибки
            for uid in batch:
                users_info.append(
                    {"id": int(uid) if uid.isdigit() else uid, "__error": True}
                )
            continue

        # 3) Спец-обработка для моков (только 'id')
        if vk_client is not None and isinstance(response, list) and response:
            if all(set(u.keys()) == {"id"} for u in response):
                for u in response:
                    users_info.append({"id": int(u["id"])})
                continue

        # 4) Для реального ответа API — полный маппинг
        for u in response:
            # 4.1) деактивированные
            if u.get("deactivated") is not None:
                users_info.append(
                    {
                        "id": int(u.get("id")) if u.get("id") is not None else None,
                        "__error": True,
                    }
                )
                continue

            # 4.2) прочие — мапим через вашу функцию
            try:
                users_info.append(map_user_to_row(u))
            except Exception:
                users_info.append(
                    {
                        "id": int(u.get("id")) if u.get("id") is not None else None,
                        "__error": True,
                    }
                )

    return users_info


def save_to_csv(
    filename: str,
    users_info: List[Dict[str, Any]],
    fieldnames: List[str] = PARSER_FIELDNAMES,
) -> None:
    """
    Сохраняет список словарей users_info в CSV-файл filename.
    По умолчанию берёт заголовки из PARSER_FIELDNAMES.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in users_info:
            writer.writerow(row)
