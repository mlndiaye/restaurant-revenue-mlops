from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

# Global model variable
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    if MODEL_PATH.exists():
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            print(f"Model successfully loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint will be unavailable.")
    yield


app = FastAPI(
    title="Restaurant Revenue Prediction API",
    description="API to predict restaurant revenue for TFI using a trained machine learning pipeline.",
    version="1.0.0",
    lifespan=lifespan
)


class RestaurantInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    open_date: str = Field(..., alias="Open Date", description="Opening date in MM/DD/YYYY format")
    city_group: str = Field(..., alias="City Group", description="Big Cities or Other")
    restaurant_type: str = Field(..., alias="Type", description="Type of restaurant: FC, IL, or DT")
    city: Optional[str] = Field(None, alias="City", description="Name of the city (optional)")
    
    # 37 obfuscated numerical features P1 to P37
    P1: float
    P2: float
    P3: float
    P4: float
    P5: float
    P6: float
    P7: float
    P8: float
    P9: float
    P10: float
    P11: float
    P12: float
    P13: float
    P14: float
    P15: float
    P16: float
    P17: float
    P18: float
    P19: float
    P20: float
    P21: float
    P22: float
    P23: float
    P24: float
    P25: float
    P26: float
    P27: float
    P28: float
    P29: float
    P30: float
    P31: float
    P32: float
    P33: float
    P34: float
    P35: float
    P36: float
    P37: float


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted annual revenue in Turkish Lira (TRY)")


@app.get("/health")
def health_check():
    """Returns the API health status and loaded model details."""
    if model_pipeline is None:
        return {
            "status": "degraded",
            "message": "API is running but the model pipeline is not loaded.",
            "model_path": str(MODEL_PATH)
        }
    
    # Extract regressor name
    regressor = model_pipeline.named_steps["regressor"]
    return {
        "status": "healthy",
        "model": type(regressor).__name__,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict", response_model=Union[PredictionResponse, List[PredictionResponse]])
def predict_revenue(payload: Union[RestaurantInput, List[RestaurantInput]]):
    """
    Predicts the annual revenue for one or multiple restaurants.
    Accepts raw restaurant characteristics, cleans them, applies the preprocessing pipeline,
    computes predictions on the log scale, and returns predictions on the original TRY scale.
    """
    global model_pipeline
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model and restart the API."
        )

    # Convert payload into DataFrame
    is_list = isinstance(payload, list)
    items = payload if is_list else [payload]
    
    # Parse payload using aliases/field names
    records = []
    for item in items:
        # Use model_dump(by_alias=True) to match DataFrame column names expected by preprocessing
        records.append(item.model_dump(by_alias=True))
        
    df = pd.DataFrame(records)

    try:
        # Import clean_dataframe from our preprocessing module
        from restaurant_revenue.features.preprocessing import clean_dataframe
        
        # Clean input data (extracts age, drops Id/City)
        cleaned_df = clean_dataframe(df)
        
        # Predict on log scale
        preds_log = model_pipeline.predict(cleaned_df)
        
        # Invert target log transformation
        preds_raw = np.expm1(preds_log)
        
        # Format response
        responses = [PredictionResponse(prediction=float(pred)) for pred in preds_raw]
        return responses if is_list else responses[0]
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error generating predictions: {str(e)}"
        )
