import time
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.pipeline import Pipeline

from src.data.preprocessing import (
    get_feature_types,
    create_raw_preprocessor,
    create_cleaned_preprocessor,
    create_engineered_preprocessor,
    create_feature_selector
)

from src.models.model_definitions import (
    get_models
)

from src.evaluation.metrics import (
    evaluate_model
)

from src.utils.random_state import (
    RANDOM_STATE
)

from src.utils.config import (
    TEST_SIZE
)


def run_computational_cost_experiment(
    X,
    y
):
    """
    Run computational cost experiment.
    """
    
    numeric_cols, categorical_cols = (
        get_feature_types(X)
    )
    
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
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
    
    results = []
    
    for preprocessing_name, preprocessor in preprocessing_levels.items():
        
        print(f"\n=== {preprocessing_name} ===")
        
        for model_name, model in models.items():
            
            print(f"Training {model_name}...")
            
            if preprocessing_name == "Engineered":
                
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    
                    ("feature_selection",
                     create_feature_selector(k=20)),
                    
                    ("classifier", model)
                ])
                
            else:
                
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ])
            
            start_train = time.perf_counter()
            
            pipeline.fit(
                X_train,
                y_train
            )
            
            end_train = time.perf_counter()
            
            training_time = (
                end_train - start_train
            )
            
            start_predict = time.perf_counter()
            
            pipeline.predict(X_test)
            
            end_predict = time.perf_counter()
            
            prediction_time = (
                end_predict - start_predict
            )
            
            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test
            )
            
            efficiency_ratio = (
                metrics["f1_score"]
                / training_time
            )
            
            results.append({
                "preprocessing": preprocessing_name,
                "model": model_name,
                
                **metrics,
                
                "training_time_seconds":
                    training_time,
                
                "prediction_time_seconds":
                    prediction_time,
                
                "efficiency_ratio":
                    efficiency_ratio
            })
    
    results_df = pd.DataFrame(results)
    
    return results_df