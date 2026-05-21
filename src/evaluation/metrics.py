import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(
    pipeline,
    X_test,
    y_test
):
    """
    Evaluate a trained pipeline on test data.
    
    Parameters
    ----------
    pipeline : sklearn Pipeline
        Trained pipeline.
        
    X_test : pandas.DataFrame
        Test features.
        
    y_test : pandas.Series
        Test labels.
    
    Returns
    -------
    metrics : dict
        Dictionary containing evaluation metrics.
    """
    
    y_pred = pipeline.predict(X_test)
    
    if hasattr(pipeline, "predict_proba"):
        
        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]
        
        roc_auc = roc_auc_score(
            (y_test == ">50K").astype(int),
            y_prob
        )
        
    else:
        roc_auc = np.nan
    
    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    
    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=">50K"
    )
    
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    
    return metrics