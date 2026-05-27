"""
Evaluation script for Restaurant Revenue Prediction.
Computes validation metrics (RMSE, MAE, R2) and saves them as JSON for DVC tracking.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from restaurant_revenue.features.preprocessing import get_preprocessor_pipeline


def evaluate_model():
    project_root = Path(__file__).resolve().parents[3]
    processed_data_path = project_root / "data" / "processed" / "train_processed.csv"
    model_path = project_root / "models" / "best_model.pkl"
    metrics_path = project_root / "metrics.json"
    
    if not processed_data_path.exists():
        raise FileNotFoundError(f"Processed training file not found at {processed_data_path}.")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}.")
        
    df = pd.read_csv(processed_data_path)
    X = df.drop(columns=["revenue"])
    y = df["revenue"]
    
    # Run a 5-fold cross-validation to get realistic metrics
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Load the saved pipeline to inspect its properties
    saved_pipeline = joblib.load(model_path)
    regressor = saved_pipeline.named_steps["regressor"]
    preprocessor_pipeline = saved_pipeline.named_steps["preprocessor"]
    
    # Extract inner ColumnTransformer
    col_transformer = preprocessor_pipeline.named_steps["preprocessor"]
    
    # Check if scaling was used
    scale_numeric = col_transformer.transformers[0][1] != "passthrough"
    
    # Check if flags were added by checking if 'add_flags' step is present
    add_flags = "add_flags" in preprocessor_pipeline.named_steps
    
    oof_predictions = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        y_train_log = np.log1p(y_train)
        
        # Build identical pipeline structure
        preprocessor = get_preprocessor_pipeline(
            add_sparse_flags=add_flags,
            scale_numeric=scale_numeric
        )
        
        from sklearn.base import clone
        cloned_regressor = clone(regressor)
        
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", cloned_regressor)
        ])
        
        pipeline.fit(X_train, y_train_log)
        y_pred_log = pipeline.predict(X_val)
        oof_predictions[val_idx] = np.expm1(y_pred_log)
        
    rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    mae = mean_absolute_error(y, oof_predictions)
    r2 = r2_score(y, oof_predictions)
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    print(f"Evaluation results -> RMSE: {rmse:,.2f} | MAE: {mae:,.2f} | R2: {r2:.4f}")
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    evaluate_model()
