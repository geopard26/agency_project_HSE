import logging
import os
import re
import sys

# Добавляем корень проекта в sys.path, чтобы импорты из src работали
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Настройка Sentry
import sentry_sdk
import streamlit as st
from sentry_sdk.integrations.logging import LoggingIntegration

sentry_logging = LoggingIntegration(
    level=logging.INFO,  # собираем INFO+ как breadcrumbs
    event_level=logging.ERROR,  # ошибки ERROR+ в Sentry
)
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENV", "dev"),
    integrations=[sentry_logging],
    traces_sample_rate=0.1,
)

from src.db.models import Profile
from src.db.session import SessionLocal, init_db
from src.logging_config import get_logger, setup_logging
from src.models.predict import THRESHOLD, predict_one
from src.parser.parser import get_users_info
from src.tasks.retrain_task import retrain_model

# Инициализируем логирование
setup_logging("DEBUG")
logger = get_logger(__name__)

# Инициализируем БД (миграции могут быть настроены через Alembic позже)
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


def extract_vk_id(s: str) -> str:
    """Из любой строки извлечь id или короткое имя."""
    s = s.strip()
    m = re.search(r"vk\.com/([A-Za-z0-9_]+)", s)
    if m:
        return m.group(1)
    return s


if st.button("Предсказать для всех"):
    logger.info("Button 'Предсказать для всех' clicked.")
    raw_lines = input_text.splitlines()
    user_ids = [extract_vk_id(line) for line in raw_lines if line.strip()]
    logger.debug("Extracted VK IDs: %s", user_ids)

    if not user_ids:
        msg = "Нужно ввести хотя бы один ID или ссылку."
        logger.warning(msg)
        st.error(msg)
    else:
        try:
            with st.spinner("Получаем данные профилей и считаем вероятности..."):
                logger.info("Fetching user info for %d users", len(user_ids))
                features_list = get_users_info(user_ids)
                logger.info("Fetched %d profiles", len(features_list))

                # 3) Получаем предсказания
                results = []
                for user_id, feats in zip(user_ids, features_list):
                    logger.debug("Predicting for user %s", user_id)
                    proba, pred_label = predict_one(feats)
                    results.append((user_id, feats, proba, pred_label))
                    logger.debug(
                        "Result for %s: proba=%.4f, label=%d",
                        user_id,
                        proba,
                        pred_label,
                    )

        except Exception as e:
            logger.exception("Error during fetching or prediction")
            st.error(f"Произошла ошибка: {e}")
        else:
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
                elif col2.button("Изменить оценку", key=f"rej_{user_id}"):
                    manual_label = False
                else:
                    manual_label = None

                if manual_label is not None:
                    try:
                        logger.info(
                            "Saving manual label %s for user %s", manual_label, user_id
                        )
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
                        logger.info("Label saved for user %s", user_id)
                        st.success(f"Метка сохранена для {user_id}!")

                        # запускаем фоновое переобучение
                        logger.info("Enqueueing retrain_model task")
                        retrain_model.delay()

                    except Exception as e:
                        logger.exception("Failed to save label for %s", user_id)
                        st.error(f"Не удалось сохранить метку: {e}")

            db.close()
