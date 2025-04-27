import pandas as pd
from src.preprocessing.process_data import clean_and_feature_engineer

def make_raw_df():
    return pd.DataFrame({
        'home_phone':   [None, '123', 0],
        'games':        [None, 5, None],
        'inspired_by':  [None, 1, None],
        'mobile_phone': [0, '890', 0],
        'country':      ['A', 0, 'B'],
        'langs':        ["['Русский', 'English']", None, "[]"],
        'university_name': [None, 'Uni', '0'],
        'id':           [1, 2, 3],
        'is_agency':    [0, 1, 0],
    })

def test_clean_numeric_only_and_features():
    df = make_raw_df()
    df2 = clean_and_feature_engineer(df)

    # должны остаться только числовые колонки
    assert all(dtype.kind in ['i','u','f'] for dtype in df2.dtypes)

    # должен быть признак university_name_binary
    assert 'university_name_binary' in df2.columns

    # lang_Русский и lang_English
    lang_cols = [c for c in df2.columns if c.startswith('lang_')]
    assert 'lang_Русский' in lang_cols
    assert 'lang_English' in lang_cols
