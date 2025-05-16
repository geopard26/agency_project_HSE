import logging
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Добавляем корень проекта в sys.path, чтобы импорты из src работали
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Настройка Sentry (оставляем вашу реализацию)
import sentry_sdk
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

# Ваши импорты
from src.db.models import Profile
from src.db.session import SessionLocal, init_db
from src.logging_config import get_logger, setup_logging
from src.models.predict import FEATURE_NAMES, THRESHOLD, load_model, predict_one
from src.parser.parser import get_users_info
from src.tasks.retrain_task import retrain_model

# Инициализируем логирование и БД
setup_logging(os.getenv("LOG_LEVEL", "DEBUG"))
logger = get_logger(__name__)
init_db()  # пока ещё используем init_db, миграции подключим позже

# ===== Простая авторизация =====
PASSWORD = os.getenv("STREAMLIT_PASSWORD", "secret")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Вход в приложение")
    pwd = st.text_input("Введите пароль:", type="password")
    if st.button("Войти"):
        if pwd == PASSWORD:
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Неверный пароль")
    st.stop()

# ===== Основной UI =====
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
    """Из строки https://vk.com/… достаёт id или screen_name."""
    s = s.strip()
    m = re.search(r"vk\.com/([A-Za-z0-9_]+)", s)
    return m.group(1) if m else s


# ===== Кэшированные обёртки =====
@st.cache_data(ttl=3600)
def get_users_info_cached(user_ids: list[str]) -> list[dict]:
    logger.info("Caching VK API call for %d users", len(user_ids))
    return get_users_info(user_ids)


@st.cache_resource
def load_model_cached():
    logger.info("Loading model from disk")
    return load_model()


# ===== Кнопка и обработка =====
if st.button("Предсказать для всех"):
    logger.info("Button 'Предсказать для всех' clicked.")
    raw_lines = input_text.splitlines()
    user_ids = [extract_vk_id(line) for line in raw_lines if line.strip()]
    logger.debug("Extracted IDs: %s", user_ids)

    if not user_ids:
        msg = "Нужно ввести хотя бы один ID или ссылку."
        logger.warning(msg)
        st.error(msg)
    else:
        try:
            with st.spinner("Получаем данные профилей и считаем вероятности..."):
                logger.info("Fetching user info for %d users", len(user_ids))
                features_list = get_users_info_cached(user_ids)
                logger.info("Fetched %d profiles", len(features_list))

                model = load_model_cached()
                results = []
                for uid, feats in zip(user_ids, features_list):
                    logger.debug("Predicting for %s", uid)
                    proba, label = predict_one(feats)
                    results.append((uid, feats, proba, label))
        except Exception as e:
            logger.exception("Error during fetch/predict")
            st.error(f"Произошла ошибка: {e}")
        else:
            db = SessionLocal()
            for uid, feats, proba, label in results:
                st.markdown("---")
                st.markdown(f"**Пользователь: ** {uid}")
                st.markdown(f"- Вероятность агентности: **{proba: .1%}**")

                st.markdown(
                    f"- Predicted: {'Да' if label else 'Нет'} _(threshold={THRESHOLD})_"
                )

                col1, col2 = st.columns(2)
                if col1.button("Согласен", key=f"ok_{uid}"):
                    manual = True
                elif col2.button("Не согласен", key=f"no_{uid}"):
                    manual = False
                else:
                    manual = None

                if manual is not None:
                    try:
                        logger.info("Saving manual label %s for %s", manual, uid)
                        prof = db.query(Profile).filter_by(user_id=uid).first()
                        if not prof:
                            prof = Profile(user_id=uid, features=feats, label=manual)
                            db.add(prof)
                        else:
                            prof.features = feats
                            prof.label = manual
                        db.commit()
                        st.success(f"Метка сохранена для {uid}!")
                        retrain_model.delay()
                    except Exception as e:
                        logger.exception("Failed to save label for %s", uid)
                        st.error(f"Не удалось сохранить метку: {e}")

            db.close()

            # ===== График важности признаков =====
            st.markdown("### Важность признаков")
            importances = model.feature_importances_
            fig, ax = plt.subplots()
            ax.bar(range(len(FEATURE_NAMES)), importances)
            ax.set_xticks(range(len(FEATURE_NAMES)))
            ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
            ax.set_ylabel("Importance")
            st.pyplot(fig, use_container_width=True)

            # ===== История из базы =====
            st.markdown("### История предсказаний")
            db = SessionLocal()
            recent = (
                db.query(Profile).order_by(Profile.created_at.desc()).limit(10).all()
            )
            db.close()
            if recent:
                hist_df = pd.DataFrame(
                    [
                        {
                            "user_id": p.user_id,
                            "label": p.label,
                            "created_at": p.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        for p in recent
                    ]
                )
                st.dataframe(hist_df, use_container_width=True)
            else:
                st.info("Нет сохранённых предсказаний.")
