import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.data.preprocessing import (
    get_feature_types,
    create_raw_preprocessor,
    create_cleaned_preprocessor,
    create_engineered_preprocessor
)

from src.models.model_definitions import (
    get_models
)

from src.experiments.run_experiment import (
    run_experiment
)

from src.utils.random_state import (
    RANDOM_STATE
)

from src.utils.config import (
    TEST_SIZE
)


def create_imbalanced_dataset(
    X,
    y,
    minority_fraction
):
    """
    Create imbalanced dataset.
    """
    
    df_combined = X.copy()
    
    df_combined["target"] = y.values
    
    majority_class = "<=50K"
    minority_class = ">50K"
    
    majority_df = df_combined[
        df_combined["target"] == majority_class
    ]
    
    minority_df = df_combined[
        df_combined["target"] == minority_class
    ]
    
    minority_sample = minority_df.sample(
        frac=minority_fraction,
        random_state=RANDOM_STATE
    )
    
    df_imbalanced = pd.concat([
        majority_df,
        minority_sample
    ])
    
    df_imbalanced = df_imbalanced.sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    
    X_imbalanced = df_imbalanced.drop(
        "target",
        axis=1
    )
    
    y_imbalanced = df_imbalanced["target"]
    
    return X_imbalanced, y_imbalanced


def run_class_imbalance_experiment(
    X,
    y,
    minority_fraction=0.5
):
    """
    Run class imbalance experiment.
    """
    
    X_imbalanced, y_imbalanced = (
        create_imbalanced_dataset(
            X,
            y,
            minority_fraction
        )
    )
    
    numeric_cols, categorical_cols = (
        get_feature_types(X_imbalanced)
    )
    
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_imbalanced,
            y_imbalanced,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_imbalanced
        )
    )
    
    preprocessing_levels = {
        
        "Raw": create_raw_preprocessor(
            numeric_cols,
            categorical_cols
        ),
        
        "Cleaned": create_cleaned_preprocessor(
            numeric_cols,
            categorical_cols
        ),
        
        "Engineered": create_engineered_preprocessor(
            numeric_cols,
            categorical_cols
        )
    }
    
    models = get_models()
    
    results_df = run_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessing_levels,
        models
    )
    
    return results_df