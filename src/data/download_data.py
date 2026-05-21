from pathlib import Path

import pandas as pd

from sklearn.datasets import fetch_openml


def download_adult_dataset():
    """
    Download Adult Income dataset
    and store locally.
    """

    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True
    )

    df = adult.frame

    raw_path = Path("data/raw")

    raw_path.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = raw_path / "adult.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    download_adult_dataset()