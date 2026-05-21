from pathlib import Path

import pandas as pd

from src.data.load_data import (
    load_adult_dataset
)

from src.data.preprocessing import (
    get_feature_types,
    create_cleaned_preprocessor,
    create_engineered_preprocessor
)


def export_processed_datasets():
    """
    Export cleaned and engineered
    datasets to processed directory.
    """

    X, y = load_adult_dataset()

    numeric_cols, categorical_cols = (
        get_feature_types(X)
    )

    preprocessing_configs = {

        "cleaned": create_cleaned_preprocessor(
            numeric_cols,
            categorical_cols
        ),

        "engineered": create_engineered_preprocessor(
            numeric_cols,
            categorical_cols
        )
    }

    output_dir = Path(
        "data/processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, preprocessor in (
        preprocessing_configs.items()
    ):

        X_processed = preprocessor.fit_transform(X)

        processed_df = pd.DataFrame(
            X_processed
        )

        processed_df["target"] = y.values

        output_path = (
            output_dir /
            f"adult_{name}.csv"
        )

        processed_df.to_csv(
            output_path,
            index=False
        )

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    export_processed_datasets()