.PHONY: install install-dev test lint train preprocess populate run worker

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

test:
	pytest -q

lint:
	black --check src app tests
	flake8 src app tests
	isort --check-only src app tests

preprocess:
	python3 src/preprocessing/process_data.py

populate:
	python3 -m src.db.populate_csv --csv data/raw/data.csv --label-col is_agency --id-col id

train:
	python3 -m src.models.train_rf

run:
	streamlit run app/streamlit_app.py

worker:
	celery -A src.tasks.celery_app.celery_app worker --loglevel=info
