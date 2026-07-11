import os
import sys
import pandas as pd

# Add the backend root directory to the python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from core.forecasting.synthetic_data import generate_synthetic_data
from core.forecasting.preprocessor import validate_dataset, clean_dataset, aggregate_to_product_level, build_features

def run_verification():
    print("=== InsightForge Pipeline Verification ===")
    
    # 1. Paths
    data_dir = os.path.join(backend_dir, "data")
    output_csv = os.path.join(data_dir, "synthetic_retail_data.csv")
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    
    # 2. Generate Data
    print(f"\nStep 1: Generating synthetic data at: {output_csv}...")
    df_raw = generate_synthetic_data(output_csv, num_years=2)
    print(f"Data generated successfully! Shape: {df_raw.shape}")
    
    # 3. Validate
    print("\nStep 2: Validating raw dataset...")
    validation_report = validate_dataset(df_raw)
    print("Validation Report Summary:")
    print(f"  - Is Valid: {validation_report['is_valid']}")
    print(f"  - Row Count: {validation_report['stats']['row_count']}")
    print(f"  - Unique Stores: {validation_report['stats']['unique_stores']}")
    print(f"  - Unique Products: {validation_report['stats']['unique_products']}")
    print(f"  - Unique Categories: {validation_report['stats']['unique_categories']}")
    print(f"  - Date Range: {validation_report['stats']['start_date']} to {validation_report['stats']['end_date']}")
    print(f"  - Errors count: {len(validation_report['errors'])}")
    print(f"  - Warnings count: {len(validation_report['warnings'])}")
    
    if not validation_report['is_valid']:
        print("CRITICAL: Validation failed!")
        for err in validation_report['errors']:
            print(f"  - Error: {err}")
        return False
        
    # 4. Clean
    print("\nStep 3: Cleaning dataset (deduplication, date parsing, imputation)...")
    df_clean = clean_dataset(df_raw)
    print(f"Cleaned dataset shape: {df_clean.shape}")
    
    # 5. Aggregate
    print("\nStep 4: Aggregating store level data to product level...")
    df_product = aggregate_to_product_level(df_clean)
    print(f"Product level dataset shape: {df_product.shape}")
    print("\nFirst few aggregated product-level records:")
    print(df_product.head(3))
    
    # 6. Feature Engineering
    print("\nStep 5: Engineering features (calendar, lags, rolling averages)...")
    df_features = build_features(df_product)
    print(f"Features dataset shape: {df_features.shape}")
    
    # Check engineered columns
    engineered_cols = [
        'day_of_week', 'month', 'is_weekend', 'day_of_year',
        'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
        'units_sold_roll_mean_7', 'units_sold_roll_mean_30'
    ]
    
    print("\nChecking engineered features presence:")
    for col in engineered_cols:
        status = "Present" if col in df_features.columns else "Missing"
        print(f"  - {col}: {status}")
        
    # Check for NaN values in features
    nan_summary = df_features[engineered_cols].isna().sum().to_dict()
    print("\nNaN values count in engineered features:")
    for col, count in nan_summary.items():
        print(f"  - {col}: {count}")
        
    print("\nSample engineered records for PRD_01:")
    print(df_features[df_features['product_id'] == 'PRD_01'][['date', 'units_sold'] + engineered_cols].head(10))
    
    print("\n=== Verification Completed Successfully ===")
    return True

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
