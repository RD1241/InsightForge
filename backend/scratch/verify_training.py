import os
import sys
import pandas as pd

# Add the backend root directory to the python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from core.forecasting.train_pipeline import run_training_pipeline, generate_future_forecast
from core.forecasting.registry import get_best_model_metadata, get_product_models
from core.forecasting.preprocessor import clean_dataset

def run_tests():
    print("=== InsightForge Forecasting Engine Verification ===")
    
    # 1. Load active data
    csv_path = os.path.join(backend_dir, "data", "synthetic_retail_data.csv")
    if not os.path.exists(csv_path):
        print(f"CRITICAL: Dataset missing at: {csv_path}. Run verify_pipeline.py first.")
        return False
        
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded dataset: {csv_path} (Shape: {df_raw.shape})")
    
    # 2. Run Training Pipeline
    print("\nStep 1: Running Model Training Pipeline (Linear Regression, Random Forest, Prophet)...")
    report = run_training_pipeline(df_raw)
    
    print("\nTraining Run Report Summary:")
    print(f"  - Timestamp: {report['timestamp']}")
    print(f"  - Products trained: {report['total_products_trained']}")
    print(f"  - Average MAE: {report['average_mae']}")
    print(f"  - Average MAPE: {report['average_mape']}%")
    print(f"  - Total Training Time: {report['total_training_time_seconds']} seconds")
    print(f"  - Prophet Available: {report['prophet_available']}")
    
    # Verify report entries
    for pid, summary in report["products"].items():
        print(f"  * Product {pid} ({summary['product_name']}):")
        print(f"    - Best Model: {summary['best_model']}")
        print(f"    - Best MAE: {summary['best_metrics']['MAE']}")
        print(f"    - Models compared: {list(summary['all_models'].keys())}")
        
    # Check registry file
    registry_path = os.path.join(backend_dir, "models_store", "models_registry.json")
    if not os.path.exists(registry_path):
        print(f"CRITICAL: models_registry.json was not created at {registry_path}!")
        return False
    print("\nRegistry successfully created!")
    
    # 3. Retrieve Best Model details from Registry
    print("\nStep 2: Retrieving best model details from Registry for PRD_01...")
    best_meta = get_best_model_metadata("PRD_01")
    if not best_meta:
        print("CRITICAL: Failed to get best model metadata from registry!")
        return False
    print(f"  - Recommended Model Name: {best_meta['model_name']}")
    print(f"  - Validation Metrics: {best_meta['metrics']}")
    print(f"  - Model Path: {best_meta['model_path']}")
    print(f"  - recommendation_reason: {best_meta['recommendation_reason']}")
    
    # 4. Generate Future Forecast (Recursive vs. Prophet)
    print("\nStep 3: Generating 30-day future forecast for PRD_01 using best model...")
    # generate_future_forecast() now expects an already-cleaned dataframe (matches the
    # cached get_clean_df() callers use in the live app), not a raw one.
    forecast_res = generate_future_forecast(clean_dataset(df_raw), "PRD_01", horizon_days=30)
    
    print(f"  - Product Name: {forecast_res['product_name']}")
    print(f"  - Model Used: {forecast_res['model_used']}")
    print(f"  - History length: {len(forecast_res['history']['dates'])} days")
    print(f"  - Forecast horizon: {forecast_res['forecast_horizon_days']} days")
    print(f"  - Forecast dates: {forecast_res['forecast']['dates'][:3]} ... {forecast_res['forecast']['dates'][-3:]}")
    print(f"  - Forecast predictions (first 5 steps): {forecast_res['forecast']['predictions'][:5]}")
    print(f"  - Forecast lower bounds (first 5 steps): {forecast_res['forecast']['lower_bound'][:5]}")
    print(f"  - Forecast upper bounds (first 5 steps): {forecast_res['forecast']['upper_bound'][:5]}")
    
    # Verify that prediction bounds make sense (upper >= pred >= lower)
    preds = forecast_res['forecast']['predictions']
    lowers = forecast_res['forecast']['lower_bound']
    uppers = forecast_res['forecast']['upper_bound']
    for p, l, u in zip(preds, lowers, uppers):
        if not (u >= p >= l):
            print(f"CRITICAL: Forecast bounds are invalid! lower={l}, pred={p}, upper={u}")
            return False
            
    print("\n=== Forecasting Engine Verification Successful! ===")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
