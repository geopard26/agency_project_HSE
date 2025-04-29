# Тема ВКР VK

Проект реализует систему автоматического обнаружения «агентности» пользователей VK (ВКонтакте) на основе признаков, извлечённых из их персональных страниц. Включает сбор данных, предобработку, обучение модели, веб-интерфейс для онлайн-предсказаний и механизмы переобучения в фоне.

---

## Содержание

- [Особенности](#особенности)  
- [Архитектура](#архитектура)  
- [Быстрый старт](#быстрый-старт)  
  - [1. Клонирование репозитория](#1-клонирование-репозитория)  
  - [2. Виртуальное окружение и зависимости](#2-виртуальное-окружение-и-зависимости)  
  - [3. Переменные окружения](#3-переменные-окружения)  
  - [4. Инициализация БД и загрузка данных](#4-инициализация-бд-и-загрузка-данных)  
  - [5. Предобработка и обучение модели](#5-предобработка-и-обучение-модели)  
  - [6. Запуск приложения](#6-запуск-приложения)  
- [Docker](#docker)  
- [Структура проекта](#структура-проекта)  
- [Разработка](#разработка)  
  - [Pre-commit hooks](#pre-commit-hooks)  
  - [Makefile](#makefile)  
- [CI/CD](#cicd)  
- [Лицензия](#лицензия)  

---

## Особенности

- **Сбор данных**: парсер VK API для получения признаков профиля.  
- **Ручная разметка**: отбор профилей с агентностью и без.  
- **Машинное обучение**: Random Forest на отмасштабированных и one-hot признаках.  
- **Веб-интерфейс**: Streamlit-приложение для онлайн-предсказаний и верификации.  
- **Фоновое переобучение**: Celery + Redis автоматически обновляют модель по новым меткам.  
- **Контейнеризация**: Docker-образ для запуска в любых средах.  
- **CI/CD**: GitHub Actions для линтинга, тестов и сборки Docker.

---

## Архитектура

```text
┌───────────────────────┐        ┌─────────────┐
│ Streamlit UI          │◀──────▶│ Predict API │
│ (app/streamlit_app.py)│        └─────────────┘
└───────────┬───────────┘               │
            │                          ▼
            │                   Random Forest
            │                   (models/random_forest.pkl)
            │                          │
         Celery                        │
    retrain_task.delay()               │
            │                          │
            ▼                          │
    Background Worker                  │
(src/tasks/* + Redis)                  │
            │                          ▼
            └────────▶ SQLite DB ◀─────┘
                     (data/app.db)

```
## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/geopard26/agency_project_HSE.git
cd agency_project_HSE
```
### 2. Виртуальное окружение и зависимости

```bash
python3 -m venv .venv
source .venv/bin/activate

# Обновляем pip и устанавливаем зависимости
pip install --upgrade pip
pip install -r requirements.txt        # ядро: парсер, БД, предобработка, модель
pip install -r requirements-app.txt    # приложение: Streamlit, Celery, Redis
pip install -r requirements-dev.txt    # dev: pytest, линтеры, pre-commit
```

### 3. Переменные окружения

```bash
cp .env.example .env
nano .env
```

Внутри ```.env```:

```bash
VK_TOKEN=vk1.a.ваш_токен_VK
DATABASE_URL=sqlite:///./data/app.db
BROKER_URL=redis://localhost:6379/0
```

### 4. Инициализация БД и загрузка данных

```bash
# Создаёт файл data/app.db
python3 -c "from src.db.session import init_db; init_db()"

# Копируем CSV с данными в папку data/raw
mkdir -p data/raw
cp /path/to/your/data.csv data/raw/data.csv

# Заполняем таблицу Profile из CSV
python3 -m src.db.populate_csv \
  --csv data/raw/data.csv \
  --label-col is_agency \
  --id-col id
```

### 5. Предобработка и обучение модели

```bash
make preprocess      # или python3 src/preprocessing/process_data.py
make train           # или python3 -m src.models.train_rf
```

После этого появятся:

- ```data/processed/data.csv```
- ```models/random_forest.pkl```

### 6. Запуск приложения

*Redis + Celery*

В одном терминале:

```bash
brew install redis        # если ещё не установлено
brew services start redis
make worker               # или celery -A src.tasks.celery_app.celery_app worker --loglevel=info
```

*Streamlit*

В другом терминале:

```bash
make run                 # или streamlit run app/streamlit_app.py
```

Откройте в браузере: [http://localhost:8501](http://localhost:8501)


## Docker

Сборка образа:

```bash
docker build -t geopard26/agency_project_hse:latest .
```

*Запуск контейнера:*

```bash
docker run --rm -p 8501:8501 \
  --env-file .env \
  geopard26/agency_project_hse:latest
```

## Структура проекта

```text
.
├── app/                      
│   └── streamlit_app.py      # Streamlit UI 
├── data/
│   ├── raw/                  # исходные CSV
│   └── processed/            # после предобработки
├── models/                   # сохранённые модели (.pkl)
├── src/
│   ├── parser/               # парсер VK API
│   ├── db/                   # SQLAlchemy, init, populate
│   ├── preprocessing/        # очистка и фичеринг
│   ├── models/               # train_rf.py, predict.py
│   └── tasks/                # Celery tasks
├── tests/                    # pytest-тесты
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── requirements-app.txt
├── requirements-dev.txt
├── .env.example
├── .pre-commit-config.yaml
├── setup.cfg                 # конфигурация flake8/isort
└── .github/
    └── workflows/ci.yml      # GitHub Actions CI

```

## Разработка

*Pre-commit hooks*

Установка и активация:

```bash
pip install pre-commit
pre-commit install
```

При каждом ```git commit``` автоматически запустятся:

- black

- isort

- flake8

## Makefile

```makefile
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
```
## CI/CD

GitHub Actions (файл ```.github/workflows/ci.yml```) автоматически при пуше в ветку ```main``` выполняет:

1. Checkout

2. Setup Python 3.10

3. Установку всех зависимостей

4. ```make lint```

5. ```make test```

6. Сборку Docker-образа

## Лицензия

Исследование выполнено в рамках гранта, предоставленного Министерством науки и высшего образования Российской Федерации (№ соглашения о предоставлении гранта: 075-15-2022-325)

Данный проект распространяется на условиях согласования.  
© 2025 Павел Егоров  

