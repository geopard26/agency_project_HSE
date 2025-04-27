import streamlit as st
from src.parser.parser import get_users_info
from src.models.predict import predict_one
from src.db.session import init_db, SessionLocal
from src.db.models import Profile
from src.tasks.retrain_task import retrain_model

# Инициализируем БД
init_db()

st.title("Интерактивная модель агентности VK")

user_id = st.text_input("Введите ID пользователя VK (например id123456):")

if st.button("Предсказать"):
    with st.spinner("Парсим профиль и считаем вероятность..."):
        features = get_users_info([user_id])[0]
        prob = predict_one(features)

    st.markdown(f"**Вероятность агентности:** {prob:.1%}")

    st.markdown("Проверьте результат и сохраните метку:")
    col1, col2 = st.columns(2)
    if col1.button("Approve"):
        label = True
    elif col2.button("Reject"):
        label = False
    else:
        label = None

    if label is not None:
        # Сохраняем в БД
        db = SessionLocal()
        prof = db.query(Profile).filter_by(user_id=user_id).first()
        if not prof:
            prof = Profile(user_id=user_id, features=features, label=label)
            db.add(prof)
        else:
            prof.features = features
            prof.label = label
        db.commit()
        db.close()

        st.success("Метка сохранена!")
        # Запускаем переобучение в фоне
        retrain_model.delay()
        st.info("Переобучение модели запущено в фоновом режиме.")

