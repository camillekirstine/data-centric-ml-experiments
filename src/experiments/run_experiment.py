import pandas as pd

from sklearn.pipeline import Pipeline

from src.data.preprocessing import (
    create_feature_selector
)

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
        
        print(f"\n=== {preprocessing_name} Preprocessing ===")
        
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
            
            pipeline.fit(
                X_train,
                y_train
            )
            
            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test
            )
            
            results.append({
                "preprocessing": preprocessing_name,
                "model": model_name,
                **metrics
            })
    
    results_df = pd.DataFrame(results)
    
    return results_df