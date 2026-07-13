import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

# Standardize path imports
from core.forecasting.train_pipeline import run_training_pipeline, generate_future_forecast
from core.forecasting.registry import get_product_models, load_training_report

router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ACTIVE_DATASET_PATH = os.path.join(DATA_DIR, "active_dataset.csv")

@router.post("/train")
async def train_models(smooth_outliers: bool = Query(True, description="Enable rolling MAD outlier smoothing prior to training")):
    """
    Triggers the training pipeline for all products in the active dataset.
    Trains Linear Regression, Ridge Regression, Random Forest, and Prophet.
    """
    if not os.path.exists(ACTIVE_DATASET_PATH):
        raise HTTPException(status_code=404, detail="No active dataset found. Please upload or load demo data first.")
        
    try:
        df_raw = pd.read_csv(ACTIVE_DATASET_PATH)
        report = run_training_pipeline(df_raw, smooth_outliers_flag=smooth_outliers)
        return {
            "message": "Models trained successfully.",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed during training run: {str(e)}")

@router.get("/report")
async def get_training_report():
    """
    Returns the persistent training report summary showing average errors,
    training durations, and product-by-product recommended models.
    """
    report = load_training_report()
    if not report:
        return JSONResponse(
            status_code=404,
            content={"message": "No training report found. Please run model training first."}
        )
    return report

@router.get("/predict")
async def predict_demand(
    product_id: str = Query(..., description="ID of the product to forecast"),
    model_name: str = Query(None, description="Specific model to use (defaults to recommended)"),
    horizon_days: int = Query(30, ge=7, le=90, description="Forecast horizon in days (7 to 90)")
):
    """
    Generates N-day future predictions for a product, showing actual historic sales
    alongside predictions and 95% confidence intervals.
    """
    if not os.path.exists(ACTIVE_DATASET_PATH):
        raise HTTPException(status_code=404, detail="No active dataset found. Load data first.")
        
    try:
        df_raw = pd.read_csv(ACTIVE_DATASET_PATH)
        forecast_res = generate_future_forecast(df_raw, product_id, model_name, horizon_days)
        return forecast_res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {str(e)}")

@router.get("/compare")
async def get_models_comparison(product_id: str = Query(..., description="ID of the product to compare models for")):
    """
    Returns validation performance metrics (MAE, RMSE, MAPE, R2) for all trained models 
    for a given product.
    """
    models = get_product_models(product_id)
    if not models:
        raise HTTPException(status_code=404, detail=f"No models found for product '{product_id}'. Train models first.")
    return models
