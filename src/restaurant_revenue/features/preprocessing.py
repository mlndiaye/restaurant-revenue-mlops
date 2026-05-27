"""
Preprocessing module for Restaurant Revenue Prediction.
Provides classes and functions for both stateless preprocessing (data cleaning,
age extraction) and stateful preprocessing (scaling, encoding) via scikit-learn.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


class RestaurantAgeExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate the age of a restaurant in days
    relative to a fixed reference date (2015-01-01).
    """
    def __init__(self, ref_date_str="2015-01-01"):
        self.ref_date_str = ref_date_str
        self.ref_date = pd.to_datetime(ref_date_str)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Open Date" in X.columns:
            # Parse dates flexibly (supporting multiple formats)
            open_date = pd.to_datetime(X["Open Date"], errors="coerce")
            X["days_since_open"] = (self.ref_date - open_date).dt.days
        else:
            # Handle cases where "Open Date" is missing by placing a default or 0
            X["days_since_open"] = 0
        return X


class SparsePFlagAdder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add binary flags (1 if P_x > 0, else 0) for highly
    sparse obfuscated numerical features.
    """
    def __init__(self, sparse_cols=None):
        # 17 features identified in EDA with >= 64% zeros
        if sparse_cols is None:
            self.sparse_cols = [
                "P14", "P15", "P16", "P17", "P18", "P24", "P25", "P26", "P27",
                "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37"
            ]
        else:
            self.sparse_cols = sparse_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.sparse_cols:
            if col in X.columns:
                X[f"{col}_active"] = (X[col] > 0).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        # Return names of input features plus the new ones
        if input_features is None:
            return None
        features = list(input_features)
        for col in self.sparse_cols:
            if col in features:
                features.append(f"{col}_active")
        return np.array(features)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies stateless cleaning to a DataFrame (extracts age, drops Id and City).
    Does not apply any stateful scaling or one-hot encoding.
    This can be safely called on the entire raw dataset to produce processed files.
    """
    df = df.copy()
    
    # 1. Age extraction
    extractor = RestaurantAgeExtractor()
    df = extractor.transform(df)
    
    # 2. Drop columns
    cols_to_drop = []
    if "Id" in df.columns:
        cols_to_drop.append("Id")
    if "City" in df.columns:
        cols_to_drop.append("City")
    if "Open Date" in df.columns:
        cols_to_drop.append("Open Date")
        
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


def get_preprocessor_pipeline(add_sparse_flags: bool = False, scale_numeric: bool = True) -> Pipeline:
    """
    Creates and returns a scikit-learn Pipeline for stateful preprocessing.
    This pipeline should be fit inside cross-validation.
    """
    steps = []
    
    # 1. Optional step: Add sparse flags
    if add_sparse_flags:
        steps.append(("add_flags", SparsePFlagAdder()))
        
    # Categorical columns to encode
    categorical_cols = ["City Group", "Type"]
    
    # Numerical columns to scale (all P features + days_since_open)
    p_cols = [f"P{i}" for i in range(1, 38)]
    numerical_cols = p_cols + ["days_since_open"]
    
    if add_sparse_flags:
        sparse_cols = [
            "P14", "P15", "P16", "P17", "P18", "P24", "P25", "P26", "P27",
            "P30", "P31", "P32", "P33", "P34", "P35", "P36", "P37"
        ]
        numerical_cols += [f"{col}_active" for col in sparse_cols]
        
    categorical_transformer = OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False)
    numerical_transformer = StandardScaler() if scale_numeric else "passthrough"
    
    col_transformer = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"  # Drop any other columns (like target column if passed in X)
    )
    
    steps.append(("preprocessor", col_transformer))
    
    return Pipeline(steps)


def process_and_save_data() -> None:
    """
    Main execution function to process raw data and save clean, stateless datasets
    to data/processed/.
    """
    from restaurant_revenue.data.load import load_data
    
    # Load raw data
    train, test = load_data()
    
    # Apply stateless cleaning (extract age, drop ID & City)
    train_cleaned = clean_dataframe(train)
    test_cleaned = clean_dataframe(test)
    
    # Define processed directory
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    train_cleaned.to_csv(processed_dir / "train_processed.csv", index=False)
    test_cleaned.to_csv(processed_dir / "test_processed.csv", index=False)
    print(f"Processed train and test datasets saved to {processed_dir}")


if __name__ == "__main__":
    process_and_save_data()
