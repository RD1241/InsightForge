import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import other core modules
from core.forecasting.preprocessor import clean_dataset, aggregate_to_product_level, build_features
from core.forecasting.models import (
    LinearRegressionModel, RandomForestModel, ProphetModel,
    evaluate_predictions, PROPHET_AVAILABLE
)
from core.forecasting.registry import save_model, save_training_report, get_best_model_metadata, load_model, get_product_models


def _recursive_validation_predict(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Evaluates an ML model on the test set using the same recursive multi-step
    forecasting strategy that is used in production (generate_future_forecast).

    WHY THIS MATTERS
    ----------------
    If we simply call model.predict(test_df) the test rows already contain lag
    features computed from *actual* future sales (e.g. units_sold_lag_1 on the
    second test day = the real first test day's sales).  The model therefore
    "sees" the true future at every step — this is data leakage and produces
    artificially optimistic MAE / MAPE / R² scores.

    RECURSIVE WALK-FORWARD APPROACH
    --------------------------------
    1. Seed the rolling buffer with the last 30 days of *training* actuals.
    2. For each test step, reconstruct lag and rolling features from the buffer
       (same code as generate_future_forecast).
    3. Predict using the model, record the prediction, then append it to the
       buffer so the NEXT step's lag_1 = the CURRENT step's prediction.

    This matches production behaviour exactly and gives honest error rates.
    """
    # Seed buffer from the tail of the training set (actual actuals)
    sales_buffer = list(train_df['units_sold'].values[-30:])

    avg_price = float(test_df['price'].mean())

    predictions = []
    for _, row in test_df.iterrows():
        # Rebuild lag / rolling features from the rolling buffer
        lag_1  = sales_buffer[-1]
        lag_7  = sales_buffer[-7]  if len(sales_buffer) >= 7  else sales_buffer[0]
        lag_14 = sales_buffer[-14] if len(sales_buffer) >= 14 else sales_buffer[0]
        roll_7  = float(np.mean(sales_buffer[-7:]))
        roll_30 = float(np.mean(sales_buffer[-30:]))

        feat_row = pd.DataFrame([{
            'price':                  avg_price,
            'promotion_flag':         int(row.get('promotion_flag', 0)),
            'day_of_week':            int(row['day_of_week']),
            'month':                  int(row['month']),
            'is_weekend':             int(row['is_weekend']),
            'day_of_year':            int(row['day_of_year']),
            'units_sold_lag_1':       lag_1,
            'units_sold_lag_7':       lag_7,
            'units_sold_lag_14':      lag_14,
            'units_sold_roll_mean_7': roll_7,
            'units_sold_roll_mean_30':roll_30,
        }])

        pred_result = model.predict(feat_row)
        pred_val    = float(pred_result['predictions'][0])

        predictions.append(pred_val)
        sales_buffer.append(pred_val)   # compound — next step uses this prediction

    return np.array(predictions)

def run_training_pipeline(df_raw: pd.DataFrame) -> dict:
    """
    Trains all applicable forecasting models (Linear Regression, Random Forest, Prophet)
    on all products in the dataset using chronological splitting, compares them, 
    persists them in the registry, and saves a training report.
    """
    start_time_all = time.time()
    
    # 1. Clean and aggregate
    df_clean = clean_dataset(df_raw)
    df_product = aggregate_to_product_level(df_clean)
    df_features = build_features(df_product)
    
    unique_products = df_product['product_id'].unique()
    product_summaries = {}
    
    # Track overall timing
    for pid in unique_products:
        product_start = time.time()
        
        # Filter features for this product
        prod_df = df_features[df_features['product_id'] == pid].copy()
        p_name = prod_df['product_name'].iloc[0]
        category = prod_df['category'].iloc[0]
        
        # 30-day chronological split (standard for retail forecasting evaluation)
        train_df = prod_df.iloc[:-30].copy()
        test_df = prod_df.iloc[-30:].copy()
        
        if len(train_df) < 14:
            # Not enough history to train lag features
            continue
            
        # Target actuals for evaluation
        y_test = test_df['units_sold'].values
        
        models_trained = {}
        
        # --- Model 1: Linear Regression ---
        lr_model = LinearRegressionModel()
        lr_model.fit(train_df)
        # Use recursive walk-forward evaluation to avoid validation leakage
        lr_preds = _recursive_validation_predict(lr_model, train_df, test_df)
        lr_metrics = evaluate_predictions(y_test, lr_preds)
        save_model(pid, "Linear Regression", lr_model, lr_metrics)
        models_trained["Linear Regression"] = lr_metrics

        # --- Model 2: Random Forest ---
        rf_model = RandomForestModel()
        rf_model.fit(train_df)
        # Use recursive walk-forward evaluation to avoid validation leakage
        rf_preds = _recursive_validation_predict(rf_model, train_df, test_df)
        rf_metrics = evaluate_predictions(y_test, rf_preds)
        save_model(pid, "Random Forest", rf_model, rf_metrics)
        models_trained["Random Forest"] = rf_metrics

        # --- Model 3: Prophet (if available) ---
        # Prophet constructs its own future dataframe internally; its predict()
        # is already fully out-of-sample and does not suffer from lag leakage.
        if PROPHET_AVAILABLE:
            try:
                prophet_model = ProphetModel()
                prophet_model.fit(train_df)
                prophet_res    = prophet_model.predict(test_df)
                prophet_preds  = prophet_res["predictions"]
                prophet_metrics = evaluate_predictions(y_test, prophet_preds)
                save_model(pid, "Prophet", prophet_model, prophet_metrics)
                models_trained["Prophet"] = prophet_metrics
            except Exception as e:
                # Fallback silently on compile errors
                print(f"Prophet failed to train for product {pid}: {str(e)}")
                
        # Find best model
        best_model_name = min(models_trained, key=lambda k: models_trained[k]["MAE"])
        best_metrics = models_trained[best_model_name]
        
        product_summaries[pid] = {
            "product_id": pid,
            "product_name": p_name,
            "category": category,
            "best_model": best_model_name,
            "best_metrics": best_metrics,
            "all_models": models_trained,
            "training_time_seconds": round(time.time() - product_start, 2)
        }
        
    # Calculate average validation errors across all products
    avg_mae = np.mean([p["best_metrics"]["MAE"] for p in product_summaries.values()]) if product_summaries else 0.0
    avg_mape = np.mean([p["best_metrics"]["MAPE"] for p in product_summaries.values()]) if product_summaries else 0.0
    
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_rows": len(df_raw),
        "total_products_trained": len(product_summaries),
        "average_mae": round(float(avg_mae), 2),
        "average_mape": round(float(avg_mape), 2),
        "products": product_summaries,
        "total_training_time_seconds": round(time.time() - start_time_all, 2),
        "prophet_available": PROPHET_AVAILABLE
    }
    
    save_training_report(report)
    return report

def generate_future_forecast(df_raw: pd.DataFrame, product_id: str, model_name: str = None, horizon_days: int = 30) -> dict:
    """
    Generates N-days out-of-sample future forecasts for a product.
    If model_name is None, loads the recommended best model from the registry.
    Uses recursive lag updating for ML models.
    """
    df_clean = clean_dataset(df_raw)
    
    # Retrieve product metadata from best model or registry
    best_meta = get_best_model_metadata(product_id)
    if not best_meta:
        raise ValueError(f"No trained models found for product '{product_id}'. Please run training first.")
        
    if model_name is None:
        model_name = best_meta["model_name"]
        
    # Load model
    model = load_model(product_id, model_name)
    
    # Filter and prepare historic actuals
    df_product = aggregate_to_product_level(df_clean)
    prod_df = df_product[df_product['product_id'] == product_id].copy()
    
    if len(prod_df) < 30:
        raise ValueError(f"Not enough history (needs at least 30 days) to generate lags for product '{product_id}'")
        
    last_historic_date = prod_df['date'].max()
    
    # We need the last 30 days of actuals to initialize the lags and rolling averages
    history_subset = prod_df.sort_values(by='date').tail(30).copy()
    
    # Keep track of baseline price and promotion for the future (default: last price, no promo)
    avg_price = float(history_subset['price'].mean())
    
    # Generate future dates
    future_dates = [last_historic_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
    
    predictions = []
    lower_bounds = []
    upper_bounds = []
    
    if model_name == "Prophet":
        # Prophet handles future projections directly as long as we pass prices/promotions
        future_df = pd.DataFrame({
            "date": future_dates,
            "price": avg_price,
            "promotion_flag": 0
        })
        # Format the predict call
        res = model.predict(future_df)
        predictions = res["predictions"].tolist()
        lower_bounds = res["lower_bound"].tolist()
        upper_bounds = res["upper_bound"].tolist()
        
    else:
        # ML models (Linear Regression, Random Forest) require recursive forecasting!
        # Create a rolling buffer of actual sales
        sales_buffer = history_subset['units_sold'].tolist()
        
        # Build features step-by-step
        for fut_date in future_dates:
            # Calendar features
            day_of_week = fut_date.weekday()
            month = fut_date.month
            is_weekend = 1 if day_of_week in [5, 6] else 0
            day_of_year = fut_date.timetuple().tm_yday
            
            # Lags (offset from the end of the buffer)
            lag_1 = sales_buffer[-1]
            lag_7 = sales_buffer[-7]
            lag_14 = sales_buffer[-14]
            
            # Rolling averages
            roll_7 = np.mean(sales_buffer[-7:])
            roll_30 = np.mean(sales_buffer[-30:])
            
            # Construct feature row matching model columns
            feat_row = pd.DataFrame([{
                "price": avg_price,
                "promotion_flag": 0,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "day_of_year": day_of_year,
                "units_sold_lag_1": lag_1,
                "units_sold_lag_7": lag_7,
                "units_sold_lag_14": lag_14,
                "units_sold_roll_mean_7": roll_7,
                "units_sold_roll_mean_30": roll_30
            }])
            
            # Predict
            pred_res = model.predict(feat_row)
            pred_val = float(pred_res["predictions"][0])
            lower_val = float(pred_res["lower_bound"][0])
            upper_val = float(pred_res["upper_bound"][0])
            
            predictions.append(round(pred_val, 2))
            lower_bounds.append(round(lower_val, 2))
            upper_bounds.append(round(upper_val, 2))
            
            # Append prediction to buffer for the next recursive steps!
            sales_buffer.append(pred_val)
            
    # Format dates as strings
    future_date_strings = [d.strftime("%Y-%m-%d") for d in future_dates]
    
    # Also fetch the last 30 days of actual historical sales to plot alongside the forecast
    history_dates = history_subset['date'].dt.strftime("%Y-%m-%d").tolist()
    history_sales = history_subset['units_sold'].tolist()
    
    # Retrieve metrics for display
    product_registry = get_best_model_metadata(product_id)
    all_trained_models = get_product_models(product_id)
    
    model_metrics = {}
    for m in all_trained_models:
        if m["model_name"] == model_name:
            model_metrics = m["metrics"]
            
    return {
        "product_id": product_id,
        "product_name": history_subset['product_name'].iloc[0],
        "category": history_subset['category'].iloc[0],
        "model_used": model_name,
        "metrics": model_metrics,
        "recommendation_reason": best_meta.get("recommendation_reason", "") if model_name == best_meta["model_name"] else "",
        "forecast_horizon_days": horizon_days,
        "history": {
            "dates": history_dates,
            "sales": history_sales
        },
        "forecast": {
            "dates": future_date_strings,
            "predictions": predictions,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds
        }
    }
