import os  # noqa: F401

import pandas as pd
import pytest

from src.tasks.retrain_task import retrain_model


class DummySelf:
    """
    Фейковый объект self для задачи bind=True,
    чтобы отловить retry.
    """

    def __init__(self):
        self.retry_called = False
        self.retry_args = None

    def retry(self, exc, countdown, max_retries):
        self.retry_called = True
        self.retry_args = dict(exc=exc, countdown=countdown, max_retries=max_retries)
        # имитируем retry, выбрасывая специфичное исключение
        raise RuntimeError("RETRY")


@pytest.fixture(autouse=True)
def _patch_io(tmp_path, monkeypatch):
    """
    Подменяем load_raw и clean_and_feature_engineer на простые функции,
    чтобы не читать реальные CSV.
    """
    # 1) Пишем временный CSV
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "is_agency": [0, 1, 0],
            "f1": [0.1, 0.5, -0.2],
        }
    )
    raw_csv = tmp_path / "raw.csv"
    df.to_csv(raw_csv, index=False)

    # 2) Мокаем load_raw и clean_and_feature_engineer
    import src.preprocessing.process_data as proc_mod

    monkeypatch.setenv("RAW_DATA_PATH", str(raw_csv))
    monkeypatch.setattr(proc_mod, "load_raw", lambda path=None: df.copy())
    # clean возвращает на входе уже готовый df
    monkeypatch.setattr(proc_mod, "clean_and_feature_engineer", lambda df: df.copy())

    yield


def test_retrain_model_success(monkeypatch):
    calls = {}

    # Мокаем train_catboost так, чтобы проверить, что ему передали правильные X,y
    def fake_train_catboost(X, y, save_path, **kwargs):
        calls["X_shape"] = X.shape
        calls["y_shape"] = y.shape
        calls["save_path"] = save_path
        return "OK"

    monkeypatch.setenv("MODEL_SAVE_PATH", "out-model.pkl")
    monkeypatch.setattr("src.tasks.retrain_task.train_catboost", fake_train_catboost)

    # Запускаем задачу
    self = DummySelf()
    result = retrain_model.run(self)

    # Проверяем, что в результате статус success
    assert result == {"status": "success"}
    # проверяем, что train_catboost получил (3,2) включая id и is_agency?
    # на самом деле train_catboost удаляет id,is_agency -> n_features = 1
    assert calls["X_shape"] == (3, 1)
    assert calls["y_shape"] == (3,)
    assert calls["save_path"] == "out-model.pkl"


def test_retrain_model_retry_on_missing_column(monkeypatch):
    # 1) Мокаем load_raw/clean так, чтобы отсутствовал is_agency
    import src.preprocessing.process_data as proc_mod

    monkeypatch.setattr(
        proc_mod, "load_raw", lambda path=None: pd.DataFrame({"id": [1, 2]})
    )
    monkeypatch.setattr(
        proc_mod, "clean_and_feature_engineer", lambda df: pd.DataFrame({"id": [1, 2]})
    )

    # 2) Запускаем и ожидаем, что будет retry (RuntimeError("RETRY"))
    self = DummySelf()
    with pytest.raises(RuntimeError):
        retrain_model.run(self)

    assert self.retry_called
    assert self.retry_args["countdown"] == 60
    assert self.retry_args["max_retries"] == 3
    assert "missing" in str(self.retry_args["exc"]) or "is_agency" in str(
        self.retry_args["exc"]
    )
