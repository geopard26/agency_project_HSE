import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

celery_app = Celery(
    'tasks',
    broker=os.getenv('BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('RESULT_BACKEND', None)
)
celery_app.autodiscover_tasks(['src.tasks'])

