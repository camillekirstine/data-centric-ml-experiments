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
    add_irrelevant_features,
    add_redundant_features
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


def run_feature_relevance_experiment(
    X,
    y,
    feature_type="irrelevant"
):
    """
    Run feature relevance experiment.
    """
    
    if feature_type == "irrelevant":
        
        X_modified = add_irrelevant_features(
            X,
            n_features=5,
            random_state=RANDOM_STATE
        )
        
    elif feature_type == "redundant":
        
        X_modified = add_redundant_features(
            X,
            n_features=5,
            random_state=RANDOM_STATE
        )
        
    else:
        raise ValueError(
            "feature_type must be "
            "'irrelevant' or 'redundant'"
        )
    
    numeric_cols, categorical_cols = (
        get_feature_types(X_modified)
    )
    
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_modified,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
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