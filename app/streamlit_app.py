import os
import re
import sys

# Добавляем корень проекта в sys.path, чтобы импорты из src работали
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st

from src.db.models import Profile
from src.db.session import SessionLocal, init_db
from src.models.predict import predict_one
from src.parser.parser import get_users_info
from src.tasks.retrain_task import retrain_model

# Порог для бинаризации вероятности
THRESHOLD = 0.98


# Функция, чтобы из любой строки извлечь id или короткое имя
def extract_vk_id(s: str) -> str:
    s = s.strip()
    # если это URL вида https://vk.com/id123 или https://vk.com/screen_name
    m = re.search(r"vk\.com/([A-Za-z0-9_]+)", s)
    if m:
        return m.group(1)
    # иначе считаем, что это и есть id или короткое имя
    return s


# Инициализируем БД
init_db()


st.title("Определение агентности по профилю VK")

st.markdown(
    "Введите в поле ниже ID пользователя или URL его страницы VK\n"
    "— можно вводить сразу несколько (по одному на строке)."
)

input_text = st.text_area(
    "ID или URL VK (каждая строка — отдельный пользователь):",
    height=120,
)

if st.button("Предсказать для всех"):
    # 1) Парсим ввод, извлекаем чистые user_id
    raw_lines = input_text.splitlines()
    user_ids = [extract_vk_id(line) for line in raw_lines if line.strip()]

    if not user_ids:
        st.error("Нужно ввести хотя бы один ID или ссылку.")
    else:
        # 2) Пишем, что работаем
        with st.spinner("Получаем данные профилей и считаем вероятности..."):
            features_list = get_users_info(user_ids)

            # 3) Получаем предсказания
            results = []
            for user_id, feats in zip(user_ids, features_list):
                proba, pred_label = predict_one(feats)
                results.append((user_id, feats, proba, pred_label))

        # 4) Для каждого результата выводим и форму для Approve/Reject
        db = SessionLocal()
        for user_id, feats, proba, pred_label in results:
            st.markdown("---")
            st.markdown(f"**Пользователь: ** {user_id}")
            st.markdown(f"- Вероятность агентности: **{proba: .1%}**")
            st.markdown(
                f"- Предсказанная метка: "
                f"{'Да' if pred_label else 'Нет'} "
                f"(threshold={THRESHOLD})"
            )

            col1, col2 = st.columns(2)
            if col1.button("Согласен", key=f"app_{user_id}"):
                manual_label = True
            elif col2.button("Изменить оценку2", key=f"rej_{user_id}"):
                manual_label = False
            else:
                manual_label = None

            if manual_label is not None:
                # Сохраняем или обновляем в БД
                prof = db.query(Profile).filter_by(user_id=user_id).first()
                if not prof:
                    prof = Profile(
                        user_id=user_id,
                        features=feats,
                        label=manual_label,
                    )
                    db.add(prof)
                else:
                    prof.features = feats
                    prof.label = manual_label

                db.commit()
                st.success(f"Метка сохранена для {user_id}!")
                # запускаем фоновое переобучение
                retrain_model.delay()

        db.close()
