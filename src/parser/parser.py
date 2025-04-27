import csv
import os
import time

import vk_api
from dotenv import load_dotenv

load_dotenv()  # подгрузит .env в os.environ
VK_TOKEN = os.getenv("VK_TOKEN")
if not VK_TOKEN:
    raise ValueError("VK_TOKEN не задан в .env")


def get_users_info(user_ids, token=VK_TOKEN):
    session = vk_api.VkApi(token=token)
    vk = session.get_api()

    users_info = []
    batch_size = 100
    fields = (
        "bdate,city,country,home_town,contacts,education,site,"
        "friends_count,followers_count,activities,interests,"
        "music,movies,tv,books,games,about,quotes,career,military,occupation"
    )
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i : i + batch_size]
        try:
            response = vk.users.get(user_ids=",".join(batch), fields=fields)
            for user in response:
                # здесь можно обрабатывать поля дополнительно
                users_info.append(user)
        except vk_api.exceptions.ApiError as e:
            print(f"API error [{e.code if hasattr(e, 'code') else ''}]: {e}")
            time.sleep(1)
    return users_info


def save_to_csv(filename, users_info):
    fieldnames = [
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
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writeheader()
        for user in users_info:
            # Пример маппинга полей; подстройте по вашим данным:
            city = (
                user.get("city", {}).get("title")
                if isinstance(user.get("city"), dict)
                else user.get("city", user.get("home_town", ""))
            )
            writer.writerow(
                {
                    "id": user.get("id", ""),
                    "first_name": user.get("first_name", ""),
                    "last_name": user.get("last_name", ""),
                    "bdate": user.get("bdate", ""),
                    "city": city,
                    "country": user.get("country", {}).get("title", ""),
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
                    "occupation": user.get("occupation", {}).get("name", ""),
                }
            )


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
        "--output", default="data.csv", help="Путь до выходного CSV-файла"
    )
    args = parser.parse_args()

    user_ids = args.user_ids.split(",")
    data = get_users_info(user_ids)
    save_to_csv(args.output, data)
    print(f"Данные сохранены в {args.output}")
