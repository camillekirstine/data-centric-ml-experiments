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
    add_noise
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


def run_noise_experiment(
    X,
    y,
    noise_level=0.1
):
    """
    Run noise robustness experiment.
    """
    
    X_noisy = add_noise(
        X,
        noise_level=noise_level,
        random_state=RANDOM_STATE
    )
    
    numeric_cols, categorical_cols = (
        get_feature_types(X_noisy)
    )
    
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_noisy,
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