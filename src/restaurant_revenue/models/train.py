"""
Training script for Restaurant Revenue Prediction.
Loads processed data, trains the best model, logs to MLflow, and saves the pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from restaurant_revenue.features.preprocessing import get_preprocessor_pipeline

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Restaurant_Revenue_Prediction")


def train_best_model():
    # Paths
    project_root = Path(__file__).resolve().parents[3]
    processed_data_path = project_root / "data" / "processed" / "train_processed.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"Loading processed training data from {processed_data_path}...")
    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"Processed training file not found at {processed_data_path}. "
            "Please run preprocessing first."
        )
        
    df = pd.read_csv(processed_data_path)
    X = df.drop(columns=["revenue"])
    y = df["revenue"]
    
    # Define our best model configuration
    # RandomForestRegressor was selected as the best model during experiments
    model_name = "RandomForest"
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    
    # Preprocessor parameters (no scaling or flags needed for RF)
    scale_numeric = False
    add_flags = False
    
    print(f"Training {model_name} model...")
    with mlflow.start_run(run_name=f"Final_{model_name}"):
        # Log model metadata and parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("scale_numeric", scale_numeric)
        mlflow.log_param("add_sparse_flags", add_flags)
        
        # Log model-specific hyperparameters
        for param, val in model.get_params().items():
            if isinstance(val, (int, float, str, bool)) or val is None:
                mlflow.log_param(f"model_param_{param}", val)
                
        # Build the final pipeline
        preprocessor = get_preprocessor_pipeline(
            add_sparse_flags=add_flags,
            scale_numeric=scale_numeric
        )
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        # Fit on log-transformed target
        y_log = np.log1p(y)
        pipeline.fit(X, y_log)
        
        # Save model pipeline locally
        model_path = models_dir / "best_model.pkl"
        joblib.dump(pipeline, model_path)
        print(f"Model saved locally to {model_path}")
        
        # Log model in MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        print("Model run and artifacts successfully tracked in MLflow.")


if __name__ == "__main__":
    train_best_model()
