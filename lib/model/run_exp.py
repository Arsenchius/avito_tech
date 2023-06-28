import yaml
import os
from typing import List

import pandas as pd
import numpy as np
import hydra
import optuna
from omegaconf import OmegaConf, DictConfig
from optuna.trial import Trial
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from prettytable import PrettyTable
from catboost import Pool, CatBoostClassifier
import logging

from training import _train_model

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_exp(cfg: DictConfig) -> None:
    """
    Run experiments, which includes:
        - Tuning model hyperparameters
        - Training model with best params
        - Dumping exp results

    Args:
        dict from config file

        config file structure:
            id - exp id
            best_params_path - path, where to save best params
            cat_features - list of categorical features
            target - target value
            drop_cols - list of features, which should be removed
            text_features - list of textual features
            number_of_trials - number of trials(rounds) for moedl tuning
            output_results_path - path, where to dump results
            model_path: path, where to save trained model
    """
    log.info(f"Start model tuning for exp {cfg.experiments.id}")
    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective(
        trial,
        cfg.train_data_path,
        cfg.experiments.cat_features,
        cfg.experiments.drop_cols,
        cfg.experiments.target,
        cfg.experiments.text_features,
    )
    study.optimize(func, n_trials=cfg.experiments.number_of_trials)
    log.info(f"Tuning finished with best roc-auc score: {study.best_value}")

    best_params = study.best_params
    with open(cfg.experiments.best_params_path, "w") as f:
        yaml.dump(best_params, f)
    log.info("Parameters saved to file")

    log.info("Start model training...")
    _train_model(
        cfg.experiments.best_params_path,
        cfg.train_data_path,
        cfg.experiments.model_path,
        cfg.experiments.cat_features,
        cfg.experiments.drop_cols,
        cfg.experiments.target,
        cfg.experiments.text_features,
    )
    log.info("Model trained and saved!")

    log.info("Model testing...")
    _dump_results(
        cfg.val_data_path,
        cfg.experiments.model_path,
        cfg.experiments.output_results_path,
        cfg.experiments.drop_cols,
        cfg.experiments.target,
        cfg.experiments.text_features,
        cfg.experiments.cat_features,
    )
    log.info("Results dumped")


def _dump_results(
    path_to_data: str,
    path_to_model: str,
    path_to_results: str,
    cols_to_drop: List[str],
    target: str,
    text_features: List[str],
    cat_features: List[str],
) -> None:
    """
    Dumps calculated metrics results in prettytable format
    """
    df = pd.read_csv(path_to_data)
    test_pool = Pool(
        data=df.drop(cols_to_drop, axis=1),
        cat_features=cat_features,
        label=df[target],
        text_features=text_features,
    )

    model = CatBoostClassifier()
    model.load_model(path_to_model)

    df["predict_proba"] = model.predict_proba(test_pool)[:, 1]
    df["predict"] = model.predict(test_pool)
    categories = dict(df.category.value_counts())
    all_roc_auc = []
    all_f1 = []
    all_accuracy = []
    category_table = PrettyTable(["Category", "ROC-AUC", "Accuracy", "F1"])
    overall_table = PrettyTable(["ROC-AUC", "Accuracy", "F1"])
    table = PrettyTable(["ROC-AUC", "Accuracy", "F1"])
    for category in categories:
        tmp = df[df.category == category]
        predicted_y = tmp.predict
        predicted_proba_y = tmp.predict_proba
        true_y = tmp.is_bad
        # calulation each metric for each group
        roc_auc_category = roc_auc_score(true_y, predicted_proba_y)
        accuracy_category = accuracy_score(true_y, predicted_y)
        f1_category = f1_score(true_y, predicted_y)

        all_roc_auc.append(roc_auc_category)
        all_f1.append(f1_category)
        all_accuracy.append(accuracy_category)

        category_table.add_row(
            [category, roc_auc_category, accuracy_category, f1_category]
        )
    # calculation for mean values for each metric type:
    mean_roc_auc = sum(all_roc_auc) / len(all_roc_auc)
    mean_f1 = sum(all_f1) / len(all_f1)
    mean_accuracy = sum(all_accuracy) / len(all_accuracy)
    overall_table.add_row([mean_roc_auc, mean_accuracy, mean_f1])

    # calculation for all roc_auc, f1, accuracy
    roc_auc = roc_auc_score(df[target], df["predict_proba"])
    f1 = f1_score(df[target], df["predict"])
    accuracy = accuracy_score(df[target], df["predict"])
    table.add_row([roc_auc, accuracy, f1])

    with open(path_to_results, "w") as f:
        f.write("Category table:\n")
        f.write(str(category_table))
        f.write("\n")
        f.write("All values table:\n")
        f.write(str(table))
        f.write("\n")
        f.write("\n")
        f.write("Overall mean values table:\n")
        f.write(str(overall_table))
        f.write("\n")


def objective(
    trial: Trial,
    path_to_train_data: str,
    cat_features: List[str],
    drop_cols: List[str],
    target: str,
    text_features: List[str],
) -> float:
    """
    Calculates a trials for model params optimization
    """
    df_train = pd.read_csv(path_to_train_data)
    df_train, df_val = train_test_split(
        df_train, test_size=0.1, random_state=43, stratify=df_train["category"]
    )
    df_train.drop(drop_cols, axis=1, inplace=True)
    df_val.drop(drop_cols, axis=1, inplace=True)
    params = {
        "eval_metric": "AUC",
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.2),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
        "iterations": trial.suggest_int("iterations", 800, 1500, 100),
        "random_strength": trial.suggest_float("random_strength", 1e-2, 1.0),
        "depth": trial.suggest_int("depth", 6, 12),
        "bootstrap_type": "Bayesian",
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
    }

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

    gbm = CatBoostClassifier(**params)
    gbm.fit(train_pool, eval_set=val_pool, verbose=200)

    y_val_pred = gbm.predict_proba(val_pool)[:, 1]
    return roc_auc_score(df_val[target], y_val_pred)


if __name__ == "__main__":
    run_exp()
