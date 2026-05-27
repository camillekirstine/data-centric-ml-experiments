from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)


def get_feature_types(X):
    
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    
    numeric_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    
    return numeric_cols, categorical_cols


def create_raw_preprocessor(
    numeric_cols,
    categorical_cols
):
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor


def create_cleaned_preprocessor(
    numeric_cols,
    categorical_cols
):
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor


def create_engineered_preprocessor(
    numeric_cols,
    categorical_cols,
    k=20
):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),

        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    engineered_pipeline = Pipeline([
        ("preprocessing", preprocessor),

        ("feature_selection", SelectKBest(
            score_func=mutual_info_classif,
            k=k
        ))
    ])

    return engineered_pipeline


def create_feature_selector(k=20):
    
    return SelectKBest(
        score_func=mutual_info_classif,
        k=k
    )