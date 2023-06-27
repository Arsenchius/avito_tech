import os
import re
import json
import argparse
import warnings
from multiprocessing import Process
from functools import lru_cache
from typing import List

import pandas as pd
import numpy as np
from pymorphy2 import MorphAnalyzer
from tqdm import tqdm

warnings.filterwarnings("ignore")


def clean_text(text: str) -> str:
    """
    Clearing a text of garbage.

    Args:
        text: a text to clean.

    Returns:
        cleaned text.
    """
    text = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ\.,\(\)]+", " ", text)
    text = re.sub(r" +", r" ", text)
    text = re.sub(r"([^ \w])", r"\1", text)
    text = re.sub(r"([^\w ])", r" \1", text)
    text = re.sub(r"^ ", r"", text)
    text = re.sub(r"[\W_]+", " ", text)
    return text


@lru_cache(100_000)
def lemmatize(token: str, stemmer) -> str:
    """
    Get the normal form of the passed token.
    Uses lru_cache to speed up the computations.

    Args:
        token: word to find the normal form,
        stemmer: MorphAnalyzer,

    Returns:
        normal form of the token.
    """
    return stemmer.parse(token)[0].normal_form


def get_stopwords(path_to_stopwords: str) -> List[str]:
    """
    Get the list of russian stopwords.

    Args:
        path to file with stopwords

    Returns:
        list of stopwords
    """
    stopwords = []
    with open(
        path_to_stopwords,
        encoding="utf-8",
    ) as f:
        for line in f:
            stopwords.append(line.strip("\n"))

    return stopwords


def remove_stopwords(text: str, path_to_stopwords: str) -> List[str]:
    """
    Remove stopwords from text.

    Args:
        text: text to process,
        path_to_stopwords: path to file, where stored stopwords.

    Returns:
        processed list with all tokens from text, which are not in stopwords
    """
    stop_words = get_stopwords(path_to_stopwords)
    return [word for word in text if word not in stop_words]


def preprocess_text(text: str, stemmer, path_to_stopwords: str) -> str:
    """
    Clean and lemmatize text, remove stopwords.

    Args:
        text: text to process,
        stemmer: MorphAnalyzer,
        path_to_stopwords: path to file, where stored stopwords.

    Returns:
        preprocessed text.
    """
    text = clean_text(text).split()
    text = remove_stopwords(text, path_to_stopwords)
    text = " ".join([lemmatize(token, stemmer) for token in text])
    return text


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create some new features

    Args:
        df: dataframe.

    Returns:
        dataframe with new created features.
    """
    df["description_len"] = df.description.apply(lambda x: len(x))
    df["title_len"] = df.title.apply(lambda x: len(x))
    df["title_count_numbers"] = df.title.apply(lambda x: len(re.findall(r"\d", x)))
    df["description_count_numbers"] = df.description.apply(
        lambda x: len(re.findall(r"\d", x))
    )
    df["phone_normal"] = df.text.str.contains("((8|\\+7|\\ )(\\D{0,3}\\d){10})").astype(
        int
    )
    df["phone_strange"] = df.text.str.contains(
        "\\s+\\D{0,3}((8|\\+7)(\\D{0,3}\\d){10})\\D{0,3}\\s+"
    ).astype(int)
    df["has_site"] = df.text.str.contains(
        "(?i)(.ru|.com|.pro|.be|ru|com|pro|be|ссылка)"
    ).astype(int)
    df["has_email"] = df.text.str.contains(
        "(?i)(( )*(@|собака)( )*([\\w\\.-]|точка)+)"
    ).astype(int)
    df["home_phone"] = df.text.str.contains("(\\D{0,3}\\d){6}").astype(int)
    df["phone_operators"] = df.text.str.contains(
        "(?i)(мтс|mts|мегафон|megafon|билайн|beeline|теле2|tele2)"
    ).astype(int)
    df["messenger"] = df.text.str.contains(
        "(?i)(inst)|(instagram)|(инстаграм)|(инст)|(ig)|(vk)|(вк)|(discord)|(дс)|(телеграм)|(telegram)|(тг)|(tg)|(whats app)|(what's app)|(wa)|(ватс ап)|(вотс ап)"
    ).astype(int)
    return df


def preprocess_data(df: pd.DataFrame, path_to_stopwords: str) -> pd.DataFrame:
    """
    Preprocessing of all data

    Args:
        df: dataframe,
        path_to_stopwords: path to file, where stored stopwords.

    Returns:
        modified dataframe, with processed text data and new created features.
    """
    tqdm.pandas()
    df.description = df.description.fillna("")
    df.title = df.title.fillna("")
    df["text"] = df.title + " " + df.description
    df = create_features(df)
    stemmer = MorphAnalyzer()
    df["text"] = df.text.progress_apply(
        lambda text: preprocess_text(text, stemmer, path_to_stopwords)
    )
    df["filtered_text"] = df.text.apply(
        lambda text: re.sub("[^A-Za-z0-9\.\@\ \-\_]", " ", text)
    )
    df["filtered_text"] = df.filtered_text.apply(lambda text: re.sub(" +", " ", text))
    df["filtered_numbers"] = df.text.apply(
        lambda text: re.sub("[^0-9\+\(\)\-]", " ", text)
    )
    df["filtered_numbers"] = df.filtered_numbers.apply(
        lambda text: re.sub(" +", " ", text)
    )
    return df


def _run_part(df: pd.DataFrame, output_file_path: str, path_to_stopwords: str) -> None:
    """
    Partial data processing

    Args:
        df: dataframe for processing,
        output_file_path: path, where to save processed dataframe,
        path_to_stopwords: path to file, where stored stopwords.
    """
    df = preprocess_data(df, path_to_stopwords)
    df.dropna(subset=["is_bad"], inplace=True)
    df.to_csv(output_file_path, mode="a", index=False, header=False)


def run(args) -> None:
    """
    Data processing with multiprocessing
    """
    train_data_path = args.train_data_path
    output_dir_path = args.output_dir_path
    chunk_size = args.chunk_size
    path_to_stopwords = args.path_to_stopwords
    output_file_path = os.path.join(output_dir_path, "val_data.csv")
    pd.DataFrame(
        columns=[
            "title",
            "description",
            "subcategory",
            "category",
            "price",
            "region",
            "city",
            "datetime_submitted",
            "is_bad",
            "text",
            "description_len",
            "title_len",
            "title_count_numbers",
            "description_count_numbers",
            "phone_normal",
            "phone_strange",
            "has_site",
            "has_email",
            "home_phone",
            "phone_operators",
            "messenger",
            "filtered_text",
            "filtered_numbers",
        ]
    ).to_csv(output_file_path, index=False)

    reader = pd.read_csv(train_data_path, chunksize=chunk_size)

    part_jobs = []
    for df in reader:
        part_jobs.append(
            Process(target=_run_part, args=(df, output_file_path, path_to_stopwords))
        )

    for job in part_jobs:
        job.start()

    for job in part_jobs:
        job.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path", type=str, help="Path to input train data", required=True
    )
    parser.add_argument(
        "--output-dir-path",
        type=str,
        help="Path to output dir for dumped processed data",
        required=True,
    )
    parser.add_argument("--chunk-size", type=int, help="Chunk size", required=True)
    parser.add_argument(
        "--path-to-stopwords", type=str, help="Path to stopwords file", required=True
    )
    args = parser.parse_args()

    run(args)
