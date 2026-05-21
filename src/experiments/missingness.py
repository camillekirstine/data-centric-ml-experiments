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

from src.features.feature_engineering import (
    add_missing_values
)

from src.experiments.run_experiment import (
    run_experiment
)


def run_missingness_experiment(
    X,
    y,
    missing_fraction=0.1,
    random_state=42
):
    """
    Run missing values experiment.
    """
    
    X_missing = add_missing_values(
        X,
        missing_fraction=missing_fraction,
        random_state=random_state
    )
    
    numeric_cols, categorical_cols = (
        get_feature_types(X_missing)
    )
    
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_missing,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y
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