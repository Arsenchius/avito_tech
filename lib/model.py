from typing import Tuple, Union

import pandas as pd
from catboost import Pool, CatBoostClassifier

from data.preprocessing import preprocess_data


def load_model(model_path: str) -> CatBoostClassifier:
    """Load the catboost model."""
    return CatBoostClassifier().load_model(model_path)


def task1(df: DataFrame) -> DataFrame:
    model_path = "/app/models/catboost/catboost_base_classifier.cbm"
    df = preprocess_data(df)

    model = load_model(model_path)
    df = df[model.feature_names_]

    predictions = model.predict_proba(df)[:, 1]
    # indices = range(len(predictions))
    # predictions = pd.DataFrame(
    #     {"index": range(len(predictions)), "prediction": predictions}
    # )
    return predictions


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
