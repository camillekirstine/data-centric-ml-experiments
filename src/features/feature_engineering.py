import numpy as np
import pandas as pd


def add_noise(
    X,
    noise_level=0.1,
    random_state=42
):
    """
    Add Gaussian noise to numeric features.
    """
    
    np.random.seed(random_state)
    
    X_noisy = X.copy()
    
    numeric_cols = X_noisy.select_dtypes(
        include=["int64", "float64"]
    ).columns
    
    for col in numeric_cols:
        
        noise = np.random.normal(
            0,
            noise_level,
            size=len(X_noisy)
        )
        
        X_noisy[col] += noise
    
    return X_noisy


def add_missing_values(
    X,
    missing_fraction=0.1,
    random_state=42
):
    """
    Randomly inject missing values.
    """
    
    np.random.seed(random_state)
    
    X_missing = X.copy()
    
    n_missing = int(
        np.prod(X_missing.shape)
        * missing_fraction
    )
    
    rows = np.random.randint(
        0,
        X_missing.shape[0],
        n_missing
    )
    
    cols = np.random.randint(
        0,
        X_missing.shape[1],
        n_missing
    )
    
    for row, col in zip(rows, cols):
        X_missing.iat[row, col] = np.nan
    
    return X_missing


def add_irrelevant_features(
    X,
    n_features=5,
    random_state=42
):
    """
    Add random irrelevant features.
    """
    
    np.random.seed(random_state)
    
    X_extended = X.copy()
    
    for i in range(n_features):
        
        X_extended[f"random_feature_{i}"] = (
            np.random.normal(
                0,
                1,
                size=len(X_extended)
            )
        )
    
    return X_extended


def add_redundant_features(
    X,
    n_features=5,
    random_state=42
):
    """
    Add redundant duplicated features.
    """
    
    np.random.seed(random_state)
    
    X_extended = X.copy()
    
    numeric_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns
    
    selected_cols = numeric_cols[:n_features]
    
    for col in selected_cols:
        
        X_extended[f"{col}_duplicate"] = (
            X_extended[col]
        )
    
    return X_extended