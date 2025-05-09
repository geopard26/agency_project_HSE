import os  # noqa: F401
import shutil  # noqa: F401
import tempfile  # noqa: F401

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from src.models.train_catboost import train_catboost


def make_binary_dataset(n_samples=20, n_features=5):
    """
    Синтетический бинарный датасет для проверки.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # создаём слегка несбалансированные метки
    y = pd.Series(rng.choice([0, 1], size=n_samples, p=[0.7, 0.3]), name="is_agency")
    return X, y


@pytest.mark.parametrize("param_search, n_iter", [(False, 1), (True, 2)])
def test_train_catboost_creates_model_and_file(tmp_path, param_search, n_iter):
    # 1) Готовим папку для модели
    out_dir = tmp_path / "models"
    out_dir.mkdir()
    model_path = out_dir / "cb.pkl"

    # 2) Синтетика
    X, y = make_binary_dataset(n_samples=50, n_features=3)

    # 3) Запуск обучения
    model = train_catboost(
        X,
        y,
        save_path=str(model_path),
        param_search=param_search,
        n_iter=n_iter,
    )

    # 4) Проверяем, что вернулся объект CatBoostClassifier
    assert isinstance(model, CatBoostClassifier)

    # 5) Проверяем, что файл действительно создан
    assert model_path.exists()

    # 6) Файл можно загрузить обратно через joblib
    import joblib

    loaded = joblib.load(str(model_path))
    assert isinstance(loaded, CatBoostClassifier)


def test_train_catboost_bad_inputs_raise():
    # передаём X,y несовместимых форм
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1])  # mismatch length
    with pytest.raises(ValueError):
        train_catboost(X, y, param_search=False)
