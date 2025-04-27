from src.models.train_rf import train_model
from src.tasks.celery_app import celery_app


@celery_app.task
def retrain_model():
    train_model()
