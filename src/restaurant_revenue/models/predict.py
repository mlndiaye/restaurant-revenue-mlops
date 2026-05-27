"""
Prediction script for Restaurant Revenue Prediction.
Loads a saved model pipeline, makes predictions on input data, and returns the results.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def load_model_pipeline(model_path: Path):
    """Loads the serialized scikit-learn pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}.")
    return joblib.load(model_path)


def predict(input_data: pd.DataFrame, model_path: Path) -> np.ndarray:
    """
    Predicts restaurant revenue using the loaded model pipeline.
    Applies inverse log transformation (expm1) to return predictions in the original scale.
    """
    pipeline = load_model_pipeline(model_path)
    
    # The pipeline expects a DataFrame containing the raw features
    # that clean_dataframe outputs (e.g. City Group, Type, P1-P37, days_since_open).
    # If the input contains Open Date, Id, or City, clean_dataframe must be run first.
    from restaurant_revenue.features.preprocessing import clean_dataframe
    
    cleaned_input = clean_dataframe(input_data)
    
    # Predict in log scale
    preds_log = pipeline.predict(cleaned_input)
    
    # Convert back to raw revenue
    preds_raw = np.expm1(preds_log)
    return preds_raw


def main():
    # Setup paths
    project_root = Path(__file__).resolve().parents[3]
    model_path = project_root / "models" / "best_model.pkl"
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_input_csv> [path_to_output_csv]")
        print("Defaulting to data/processed/test_processed.csv for demonstration...")
        input_csv_path = project_root / "data" / "processed" / "test_processed.csv"
        output_csv_path = project_root / "data" / "processed" / "predictions.csv"
    else:
        input_csv_path = Path(sys.argv[1])
        output_csv_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        
    if not input_csv_path.exists():
        print(f"Error: Input file {input_csv_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading input data from {input_csv_path}...")
    input_df = pd.read_csv(input_csv_path)
    
    # If the input contains 'revenue' (like processed train), drop it
    if "revenue" in input_df.columns:
        input_df = input_df.drop(columns=["revenue"])
        
    try:
        predictions = predict(input_df, model_path)
        print(f"Predictions generated successfully for {len(predictions)} rows.")
        
        # Display first few predictions
        for i, val in enumerate(predictions[:5]):
            print(f"Row {i}: Predicted Revenue = {val:,.2f}")
            
        if output_csv_path:
            # Create a dataframe with predictions
            output_df = pd.DataFrame({"Predicted_Revenue": predictions})
            output_df.to_csv(output_csv_path, index=False)
            print(f"Predictions saved to {output_csv_path}")
            
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
