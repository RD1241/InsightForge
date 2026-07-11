import os
import sys
import pandas as pd

# Add the backend root directory to the python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from core.forecasting.synthetic_data import generate_synthetic_data
from core.forecasting.preprocessor import validate_dataset, clean_dataset, aggregate_to_product_level, build_features

def verify_dataset_flow(df_raw, name):
    print(f"\n--- Verifying Flow for: {name} ---")
    
    # 1. Validate
    print("Step 1: Validating raw dataset...")
    validation_report = validate_dataset(df_raw)
    print("Validation Report Summary:")
    print(f"  - Is Valid: {validation_report['is_valid']}")
    print(f"  - Row Count: {validation_report['stats']['row_count']}")
    print(f"  - Unique Stores: {validation_report['stats']['unique_stores']}")
    print(f"  - Unique Products: {validation_report['stats']['unique_products']}")
    print(f"  - Unique Categories: {validation_report['stats']['unique_categories']}")
    print(f"  - Date Range: {validation_report['stats']['start_date']} to {validation_report['stats']['end_date']}")
    print(f"  - Errors: {len(validation_report['errors'])}")
    print(f"  - Warnings: {len(validation_report['warnings'])}")
    
    if not validation_report['is_valid']:
        print("CRITICAL: Validation failed!")
        for err in validation_report['errors']:
            print(f"    - Error: {err}")
        return False
        
    # 2. Clean
    print("Step 2: Cleaning dataset...")
    df_clean = clean_dataset(df_raw)
    print(f"  - Cleaned Shape: {df_clean.shape}")
    
    # 3. Aggregate
    print("Step 3: Aggregating store level data to product level...")
    df_product = aggregate_to_product_level(df_clean)
    print(f"  - Product Level Shape: {df_product.shape}")
    
    # 4. Feature Engineering
    print("Step 4: Engineering features...")
    df_features = build_features(df_product)
    print(f"  - Features Dataset Shape: {df_features.shape}")
    
    # Check engineered columns
    engineered_cols = [
        'day_of_week', 'month', 'is_weekend', 'day_of_year',
        'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
        'units_sold_roll_mean_7', 'units_sold_roll_mean_30',
        'product_name', 'category', 'price', 'stock_on_hand', 'promotion_flag'
    ]
    
    missing_cols = [col for col in engineered_cols if col not in df_features.columns]
    if missing_cols:
        print(f"CRITICAL: Missing engineered columns: {missing_cols}")
        return False
        
    # Check NaN count in engineered features
    nan_counts = df_features[engineered_cols].isna().sum().sum()
    if nan_counts > 0:
        print(f"CRITICAL: Found {nan_counts} NaN values in features!")
        print(df_features[engineered_cols].isna().sum())
        return False
        
    print(f"  - All features verified with 0 NaNs! Flow Success.")
    print("Aggregates Sample:")
    print(df_features[['date', 'product_id', 'product_name', 'category', 'units_sold', 'price', 'stock_on_hand']].head(3))
    
    # 5. Run EDA Report
    print("Step 5: Generating EDA report...")
    from core.forecasting.eda import generate_eda_report
    eda_report = generate_eda_report(df_clean)
    
    required_keys = [
        "dataset_overview", "descriptive_statistics", "top_products", 
        "category_performance", "sales_trend", "weekly_seasonality", 
        "monthly_seasonality", "correlation_matrix", "outliers"
    ]
    missing_keys = [k for k in required_keys if k not in eda_report]
    if missing_keys:
        print(f"CRITICAL: EDA report is missing required keys: {missing_keys}")
        return False
        
    print(f"  - EDA Report generated successfully! Outliers detected: {eda_report['outliers_count']}")
    print(f"  - Dataset overview stats: Stores={eda_report['dataset_overview']['unique_stores']}, Products={eda_report['dataset_overview']['unique_products']}, Categories={eda_report['dataset_overview']['unique_categories']}")
    return True


def run_verification():
    print("=== InsightForge Pipeline Dataset-Agnostic Verification ===")
    
    # Test 1: Synthetic Dataset Flow
    print("\n[TEST 1] Generating and testing Synthetic Retail Demo dataset...")
    data_dir = os.path.join(backend_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_synthetic = os.path.join(data_dir, "synthetic_retail_data.csv")
    df_synthetic = generate_synthetic_data(output_synthetic, num_years=2)
    
    if not verify_dataset_flow(df_synthetic, "Synthetic Retail Demo"):
        return False
        
    # Test 2: Kaggle Dataset Flow
    real_csv = os.path.join(os.path.dirname(backend_dir), "data", "train.csv")
    if os.path.exists(real_csv):
        print(f"\n[TEST 2] Loading and testing Real Kaggle Dataset from: {real_csv}...")
        # Load a subset of 100,000 rows to keep verification extremely fast
        df_real = pd.read_csv(real_csv, nrows=100000)
        print(f"Loaded {len(df_real)} rows from Kaggle train.csv.")
        
        if not verify_dataset_flow(df_real, "Kaggle Store Item Demand (Sample)"):
            return False
    else:
        print(f"\n[TEST 2] WARNING: Kaggle train.csv not found at: {real_csv}. Skipping real dataset verification.")
        
    print("\n=== All Pipeline Verifications Passed Successfully ===")
    return True

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
