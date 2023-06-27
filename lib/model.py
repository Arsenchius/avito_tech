from typing import Tuple, Union

import pandas as pd
from catboost import Pool, CatBoostClassifier

from data.preprocessing import preprocess_data


def load_model(model_path: str) -> CatBoostClassifier:
    """Load the catboost model."""
    return CatBoostClassifier().load_model(model_path)


def task1(df: pd.DataFrame, model_path: str, path_to_stop_words: str) -> pd.DataFrame:
    df = preprocess_data(df, path_to_stop_words)

    model = load_model(model_path)
    df = df[model.feature_names_]

    predictions = model.predict_proba(df)[:, 1]
    return predictions


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
