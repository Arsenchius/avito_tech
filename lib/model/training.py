import yaml
from typing import List

import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split


def _train_model(
    path_to_params: str,
    path_to_train_data: str,
    path_to_model: str,
    cat_features: List[str],
    drop_cols: List[str],
    target: str,
    text_features: List[str],
) -> None:
    """
    Trains model with best tuned params
    """
    df_train = pd.read_csv(path_to_train_data)
    df_train, df_val = train_test_split(
        df_train, test_size=0.1, random_state=43, stratify=df_train["category"]
    )
    df_train.drop(drop_cols, axis=1, inplace=True)
    df_val.drop(drop_cols, axis=1, inplace=True)

    with open(path_to_params, "r") as f:
        best_params = yaml.safe_load(f)
    best_params["eval_metric"] = "AUC"
    best_params["bootstrap_type"] = "Bayesian"

    train_pool = Pool(
        data=df_train.drop([target], axis=1),
        cat_features=cat_features,
        label=df_train[target],
        text_features=text_features,
    )
    val_pool = Pool(
        data=df_val.drop([target], axis=1),
        cat_features=cat_features,
        label=df_val[target],
        text_features=text_features,
    )

    gbm = CatBoostClassifier(**best_params)
    gbm.fit(train_pool, eval_set=val_pool, verbose=200)
    gbm.save_model(path_to_model)
