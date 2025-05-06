import os  # noqa: F401

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.process_data import (
    clean_and_feature_engineer,
    load_raw,
    save_processed,
)


def make_raw_df():
    """Создаём DataFrame, покрывающий все ветки clean_and_feature_engineer."""
    return pd.DataFrame(
        {
            # 1. Для fillna + бинаризации home_phone/games/inspired_by
            "home_phone": [np.nan, "123", 0],
            "games": [np.nan, "g", 0],
            "inspired_by": [np.nan, "x", np.nan],
            # 2. Для бинаризации mobile_phone
            "mobile_phone": [0, "555", ""],
            # 3. Для index_contacts
            "country": ["RU", 0, ""],
            "city": ["Moscow", "", "NY"],
            "home_town": ["", "Berlin", 0],
            "relation": [1, 0, 2],
            "site": ["", "vk.com", np.nan],  # Оставляем ЭТОТ ключ
            # 4. Для langs
            "langs": ["['Русский', 'English']", np.nan, "['Deutsch']"],
            # 5. Для university_name_binary
            "university_name": ["HSE", "", "0"],
            # Текстовые поля, которые должны быть удалены
            "about": ["x", "y", "z"],
            "quotes": ["q", "w", "e"],
            # Произвольная текстовая колонка, чтобы проверить select_dtypes
            "some_text": ["foo", "bar", "baz"],
            # Числовые, которые должны остаться
            "numeric1": [1.5, 2.5, 3.5],
            "numeric2": [10, 20, 30],
        }
    )


def test_fill_and_binary_and_index_and_university_and_langs():
    df = make_raw_df()
    out = clean_and_feature_engineer(df.copy())

    # 1) home_phone, games, inspired_by: NaN→0, потом бинаризация home_phone
    assert set(out["home_phone"].unique()) == {0, 1}
    assert "games" in df.columns  # games → fillna(0) but then select_dtypes drops it

    # 2) mobile_phone бинаризация: 0→0, ''→1, '555'→1
    assert out["mobile_phone"].tolist() == [0, 1, 1]

    # 3) index_contacts: среднее по ненулевым контактам
    # row0: country,city,home_town,site = non-zero in 'country','city' → 2/4 = 0.5
    assert pytest.approx(out.loc[0, "index_contacts"], rel=1e-3) == 0.5
    # проверьте, что столбец создан
    assert "index_contacts" in out.columns

    # 4) langs → lang_<язык>
    assert "lang_Русский" in out.columns
    assert "lang_English" in out.columns
    assert "lang_Deutsch" in out.columns
    # правильное заполнение
    assert out.loc[0, "lang_Русский"] == 1
    assert out.loc[1, "lang_Русский"] == 0

    # 5) university_name_binary: HSE→1, ''→0, '0'→0
    assert out["university_name_binary"].tolist() == [1, 0, 0]

    # 6) drop текстовых: 'site','about','quotes','inspired_by','langs','university_name'
    for col in [
        "site",
        "about",
        "quotes",
        "inspired_by",
        "langs",
        "university_name",
        "some_text",
    ]:
        assert col not in out.columns

    # 7) остаются только числовые: проверяем numeric1, numeric2, mobile_phone,
    # home_phone, index_contacts, lang_*, university_name_binary
    for col in [
        "numeric1",
        "numeric2",
        "mobile_phone",
        "home_phone",
        "index_contacts",
        "lang_Русский",
        "lang_English",
        "lang_Deutsch",
        "university_name_binary",
    ]:
        assert col in out.columns


def test_save_processed(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out_file = tmp_path / "out.csv"
    # 1) первый вызов: создался header + 3 строки
    save_processed(df, path=str(out_file))
    text = out_file.read_text(encoding="utf-8-sig").splitlines()
    # header и три записи
    assert text[0] == "a,b"
    assert len(text) == 1 + 3

    # 2) второй вызов: без дублирования header
    save_processed(df, path=str(out_file))
    text2 = out_file.read_text(encoding="utf-8-sig").splitlines()
    assert len(text2) == 1 + 6


def test_load_raw(tmp_path, monkeypatch):
    # создаём CSV с BOM
    csv = tmp_path / "test.csv"
    data = "col1,col2\n1,2\n3,4"
    csv.write_bytes(b"\xef\xbb\xbf" + data.encode("utf-8"))
    # подменим путь в функции load_raw
    monkeypatch.chdir(tmp_path)
    df = load_raw(path=str(csv.name))
    # проверяем, что считалось правильно
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[1]["col2"] == 4
