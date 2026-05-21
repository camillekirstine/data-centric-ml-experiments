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


def create_dataset_subset(
    X,
    y,
    fraction
):
    """
    Create reduced dataset subset.
    """
    
    X_subset, _, y_subset, _ = (
        train_test_split(
            X,
            y,
            train_size=fraction,
            stratify=y,
            random_state=RANDOM_STATE
        )
    )
    
    return X_subset, y_subset


def run_dataset_size_experiment(
    X,
    y,
    dataset_fraction=0.5
):
    """
    Run dataset size experiment.
    """
    
    if dataset_fraction < 1.0:
        
        X_subset, y_subset = (
            create_dataset_subset(
                X,
                y,
                dataset_fraction
            )
        )
        
    else:
        
        X_subset = X.copy()
        y_subset = y.copy()
    
    numeric_cols, categorical_cols = (
        get_feature_types(X_subset)
    )
    
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_subset,
            y_subset,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_subset
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