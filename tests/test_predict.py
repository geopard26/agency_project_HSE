import os

import joblib
import pandas as pd
import pytest

# Перед импортом модуля нужно гарантировать,
# что файл data/processed/data.csv существует,
# иначе при импорте predict.py упадёт pd.read_csv.
os.makedirs("data/processed", exist_ok=True)
pd.DataFrame(columns=["id", "is_agency", "f1", "f2"]).to_csv(
    "data/processed/data.csv", index=False, encoding="utf-8-sig"
)

import src.models.predict as predict_mod


def teardown_module(module):
    # Убираем созданный файл, чтобы не мешать другим тестам
    try:
        os.remove("data/processed/data.csv")
    except FileNotFoundError:
        pass


def test_load_model_file_not_found(monkeypatch):
    """Если MODEL_PATH не существует, load_model выбрасывает FileNotFoundError."""
    monkeypatch.setattr(predict_mod, "MODEL_PATH", "no_such_file.pkl")
    predict_mod._model = None  # сброс кеша
    with pytest.raises(FileNotFoundError):
        predict_mod.load_model()


def test_load_model_success(tmp_path, monkeypatch):
    """load_model корректно грузит объект и кеширует его."""
    # создаём dummy-модель
    dummy = {"foo": "bar"}
    p = tmp_path / "m.pkl"
    joblib.dump(dummy, str(p))

    monkeypatch.setattr(predict_mod, "MODEL_PATH", str(p))
    predict_mod._model = None

    m1 = predict_mod.load_model()
    m2 = predict_mod.load_model()
    assert m1 == dummy
    assert m1 is m2  # второй раз вернулся тот же объект из кеша


def test_predict_one_above_threshold(monkeypatch):
    """
    Проверяем, что predict_one возвращает (proba, 1) когда proba >= THRESHOLD.
    """
    # 1) Подменяем препроцессинг так, чтобы он выдавал DataFrame ровно из raw
    monkeypatch.setattr(predict_mod, "clean_and_feature_engineer", lambda df: df)
    # 2) Фиксируем FEATURE_NAMES
    monkeypatch.setattr(predict_mod, "FEATURE_NAMES", ["a", "b"])

    # 3) Подменяем загрузку модели на fake с нужным predict_proba
    class FakeModel:
        def predict_proba(self, df):
            # df должен иметь колонки a, b – порядок неважен для теста
            return [[0.1, 0.8]]  # p=0.8

    monkeypatch.setattr(predict_mod, "load_model", lambda: FakeModel())

    raw = {"a": 10, "b": 20, "extra": 999}
    proba, label = predict_mod.predict_one(raw)
    assert pytest.approx(proba, rel=1e-6) == 0.8
    assert label == 1


def test_predict_one_below_threshold(monkeypatch):
    """
    Проверяем, что predict_one возвращает (proba, 0) когда proba < THRESHOLD.
    """
    monkeypatch.setattr(predict_mod, "clean_and_feature_engineer", lambda df: df)
    monkeypatch.setattr(predict_mod, "FEATURE_NAMES", ["x", "y"])

    class FakeModel:
        def predict_proba(self, df):
            return [[0.7, 0.3]]  # p=0.3

    monkeypatch.setattr(predict_mod, "load_model", lambda: FakeModel())

    raw = {"x": 1, "y": 2}
    proba, label = predict_mod.predict_one(raw)
    assert proba == 0.3
    assert label == 0
