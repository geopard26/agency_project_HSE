import logging
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import sentry_sdk
import streamlit as st
from dotenv import load_dotenv
from sentry_sdk.integrations.logging import LoggingIntegration

# 1) Путь и .env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

# 2) Sentry
sentry_logging = LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN", ""),
    environment=os.getenv("ENV", "dev"),
    integrations=[sentry_logging],
    traces_sample_rate=0.1,
)

# 3) Импорты ML-слоя
from src.db.models import Profile
from src.db.session import SessionLocal, init_db

# 4) Логирование
from src.logging_config import get_logger, setup_logging
from src.models.predict import FEATURE_NAMES, load_model
from src.parser.parser import get_users_info
from src.preprocessing.process_data import clean_and_feature_engineer

setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

# 5) Инициализация БД (для dev/tests), в проде через Alembic
init_db()

# 6) Настройка страницы
st.set_page_config(
    page_title="Агентность VK",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Авторизация + VK_TOKEN =====
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "vk_token" not in st.session_state:
    st.session_state.vk_token = os.getenv("VK_TOKEN", "")

if not st.session_state.authenticated:
    st.title("🔐 Вход в приложение")
    with st.form("login_form"):
        pwd = st.text_input("Пароль:", type="password")
        vk = st.text_input(
            "VK_TOKEN:",
            type="password",
            value=st.session_state.vk_token,
            help="Секретный токен VK API",
        )
        submit = st.form_submit_button("Войти")
        if submit:
            if pwd == os.getenv("STREAMLIT_PASSWORD", "secret"):
                st.session_state.authenticated = True
                st.session_state.vk_token = vk
                os.environ["VK_TOKEN"] = vk
                st.success("Успешно авторизованы!")
            else:
                st.error("Неверный пароль")
    st.stop()

st.sidebar.success("✅ Авторизация пройдена")

# ===== Порог =====
threshold = st.sidebar.slider(
    "Порог P(агентности)", 0.0, 1.0, float(os.getenv("DEFAULT_THRESHOLD", 0.754)), 0.01
)

# ===== Заголовок и ввод =====
st.title("🕵️ Определяем агентность VK")
st.markdown("Введите ID или URL VK (по одному на строке):")
input_text = st.text_area("ID или URL VK:", height=120)


def extract_vk_id(s: str) -> str:
    s = s.strip()
    m = re.search(r"vk\.com/([A-Za-z0-9_]+)", s)
    return m.group(1) if m else s


# ===== Кэширование =====
@st.cache_data(ttl=600)
def get_users_info_cached(ids: list[str]) -> list[dict]:
    logger.info("Запрос %d профилей VK", len(ids))
    return get_users_info(ids)


@st.cache_resource
def load_model_cached():
    logger.info("Загружаем модель")
    return load_model()


# ===== Предсказание =====
results = []
if st.button("Предсказать"):
    ids = [extract_vk_id(link) for link in input_text.splitlines() if link.strip()]
    if not ids:
        st.error("Введите хотя бы один ID или URL")
    else:
        progress = st.progress(0)
        try:
            with st.spinner("Считаем..."):
                feats_list = get_users_info_cached(ids)
                model = load_model_cached()

                for i, user in enumerate(feats_list):
                    uid = user.get("id", ids[i])
                    progress.progress((i + 1) / len(ids))

                    # 1) Ошибочный профиль
                    if user.get("__error"):
                        results.append({"user_id": uid, "error": True})
                        continue
                    # 2) Нормальный профиль — делаем предсказание
                    df = pd.DataFrame([user])
                    X = clean_and_feature_engineer(df)
                    X = X.reindex(columns=FEATURE_NAMES, fill_value=0)
                    proba = model.predict_proba(X)[0, 1]
                    label = int(proba >= threshold)
                    results.append(
                        {"user_id": uid, "proba": proba, "label": label, "error": False}
                    )

                st.success("Готово")
        except Exception as e:
            logger.exception("Ошибка при предсказании")
            st.error(f"Ошибка: {e}")
# ===== Вывод результатов =====
if results:
    st.markdown("## 📊 Результаты")
    df_res = pd.DataFrame(results)

    for _, row in df_res.iterrows():
        uid = row["user_id"]
        if row.get("error", False):
            st.error(f"Пользователь «{uid}» не найден или ID некорректен")
            continue

        proba = row["proba"]
        label = row["label"]

        # Ссылка
        st.markdown(f"🔗 vk.com/{uid}")

        # Цветной блок вероятности + метка
        col1, col2 = st.columns(2)
        col1.markdown(
            f"""<div style='padding: 8px; 
            border-radius: 6px; 
            background-color: """  # noqa: E702
            f"{'#e2f8e8' if label else '#fde2e2'}; '>"  # noqa: E702
            f"""<h2 style='margin: 0; 
            color: {'#006b2c' if label else '#a12a2a'}'>"""  # noqa: E702
            f"{proba: .1%}"
            f"</h2></div>",
            unsafe_allow_html=True,
        )
        col2.metric("Метка", "Агентный" if label else "НЕ агентный")

        # Кнопки для ручной проверки
        a1, a2 = st.columns(2)
        if a1.button("Согласен", key=f"ok_{uid}"):
            manual = True
        elif a2.button("Не согласен", key=f"no_{uid}"):
            manual = False
        else:
            manual = None

        if manual is not None:
            db = SessionLocal()
            prof = db.query(Profile).filter_by(user_id=uid).first()
            if not prof:
                prof = Profile(
                    user_id=uid, features=row.get("features", {}), label=manual
                )
                db.add(prof)
            else:
                prof.label = manual
            db.commit()
            db.close()
            st.success(f"Метка для {uid} обновлена на {manual}")

        st.divider()

    # Экспорт CSV
    valid = df_res[~df_res["error"]].drop(columns=["error"], errors="ignore")
    st.download_button("Скачать CSV", valid.to_csv(index=False), "results.csv")

    # Топ-3 признака
    with st.expander("Топ-5 признаков"):
        imps = dict(zip(FEATURE_NAMES, load_model_cached().feature_importances_))
        top3 = sorted(imps.items(), key=lambda x: x[1], reverse=True)[1:6]
        df_imp = pd.DataFrame(top3, columns=["Признак", "Важность"])
        st.table(df_imp)
        fig, ax = plt.subplots()
        ax.bar(df_imp["Признак"], df_imp["Важность"])
        ax.set_title("Топ-5 признаков")
        st.pyplot(fig, use_container_width=True)

    # История
    with st.expander("История (последние записи)"):
        n = st.slider("Показывать последних", 1, 50, 10)
        db = SessionLocal()
        hist = db.query(Profile).order_by(Profile.created_at.desc()).limit(n).all()
        db.close()
        if hist:
            df_h = pd.DataFrame(
                [
                    {
                        "user_id": p.user_id,
                        "метка": "Агентный" if p.label else "НЕ агентный",
                        "дата": p.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    for p in hist
                ]
            )
            st.table(df_h)
        else:
            st.info("Нет меток")

# ===== Фоновой retrain =====
if results and st.button("Re-train модель сейчас"):
    from src.tasks.retrain_task import retrain_model

    retrain_model.delay()
    st.info("Задача переобучения отправлена в очередь")
