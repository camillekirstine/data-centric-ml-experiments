import pandas as pd

from sklearn.pipeline import Pipeline

from src.evaluation.metrics import (
    evaluate_model
)


def run_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessing_levels,
    models
):
    """
    Run experiment across preprocessing
    levels and models.

    Parameters
    ----------
    X_train : pandas.DataFrame
    y_train : pandas.Series
    X_test : pandas.DataFrame
    y_test : pandas.Series

    preprocessing_levels : dict
        Dictionary of preprocessors.

    models : dict
        Dictionary of models.

    Returns
    -------
    results_df : pandas.DataFrame
        Experiment results.
    """

    results = []

    for preprocessing_name, preprocessor in preprocessing_levels.items():

        print(
            f"\n=== {preprocessing_name} Preprocessing ==="
        )

        for model_name, model in models.items():

            print(f"Training {model_name}...")

            # ==========================================
            # CREATE PIPELINE
            # ==========================================

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

            # ==========================================
            # TRAIN MODEL
            # ==========================================

            pipeline.fit(
                X_train,
                y_train
            )

            # ==========================================
            # EVALUATE MODEL
            # ==========================================

            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test
            )

            # ==========================================
            # STORE RESULTS
            # ==========================================

            results.append({

                "preprocessing": preprocessing_name,

                "model": model_name,

                **metrics
            })

    # ==========================================
    # CREATE RESULTS DATAFRAME
    # ==========================================

    results_df = pd.DataFrame(results)

    return results_df