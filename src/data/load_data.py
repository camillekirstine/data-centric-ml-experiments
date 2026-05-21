from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


def load_adult_dataset():
    """
    Load Adult Income dataset
    from local raw data directory.
    """

    dataset_path = (
        PROJECT_ROOT /
        "data" /
        "raw" /
        "adult.csv"
    )

    df = pd.read_csv(
        dataset_path
    )

    X = df.drop(
        columns=["class"]
    )

    y = df["class"]

    return X, y