import os
import sys
import numpy as np
import pandas as pd

# Add backend dir to python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from core.forecasting.preprocessor import clean_dataset, aggregate_to_product_level, build_features
from core.forecasting.models import GradientBoostingModel, RidgeRegressionModel, evaluate_predictions

def smooth_outliers(df_product: pd.DataFrame, window: int = 14, threshold: float = 3.0) -> pd.DataFrame:
    """
    Identifies and smooths extreme sales spikes using a rolling median absolute deviation (MAD).
    Extreme outliers are clipped to the rolling threshold bounds.
    """
    df_smoothed = df_product.copy()
    for pid, grp in df_smoothed.groupby('product_id'):
        rolling_median = grp['units_sold'].rolling(window=window, min_periods=1, center=True).median()
        rolling_mad = grp['units_sold'].rolling(window=window, min_periods=1, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        # Avoid zero division
        rolling_mad = rolling_mad.fillna(1.0).replace(0.0, 1.0)
        
        upper_limit = rolling_median + threshold * rolling_mad
        lower_limit = np.maximum(0, rolling_median - threshold * rolling_mad)
        
        indices = grp.index
        df_smoothed.loc[indices, 'units_sold'] = np.clip(
            grp['units_sold'].values,
            lower_limit.values,
            upper_limit.values
        ).astype(int)
    return df_smoothed

def recursive_evaluate(model, train_df, test_df):
    sales_buffer = list(train_df['units_sold'].values[-30:])
    avg_price = float(test_df['price'].mean())
    feat_row = pd.DataFrame(np.zeros((1, len(model.feature_cols))), columns=model.feature_cols)
    feat_row.at[0, 'price'] = avg_price
    
    # Seed promotion lag
    last_train_promo = train_df['promotion_flag'].iloc[-1]
    promos = [last_train_promo] + list(test_df['promotion_flag'].values)
    
    predictions = []
    for idx, (_, row) in enumerate(test_df.iterrows(), start=1):
        lag_1  = sales_buffer[-1]
        lag_7  = sales_buffer[-7]  if len(sales_buffer) >= 7  else sales_buffer[0]
        lag_14 = sales_buffer[-14] if len(sales_buffer) >= 14 else sales_buffer[0]
        roll_7  = float(np.mean(sales_buffer[-7:]))
        roll_30 = float(np.mean(sales_buffer[-30:]))
        promo_lag = promos[idx - 1]
        
        feat_row.at[0, 'promotion_flag'] = int(row.get('promotion_flag', 0))
        feat_row.at[0, 'day_of_week'] = int(row['day_of_week'])
        feat_row.at[0, 'month'] = int(row['month'])
        feat_row.at[0, 'is_weekend'] = int(row['is_weekend'])
        feat_row.at[0, 'day_of_year'] = int(row['day_of_year'])
        feat_row.at[0, 'units_sold_lag_1'] = lag_1
        feat_row.at[0, 'units_sold_lag_7'] = lag_7
        feat_row.at[0, 'units_sold_lag_14'] = lag_14
        feat_row.at[0, 'units_sold_roll_mean_7'] = roll_7
        feat_row.at[0, 'units_sold_roll_mean_30'] = roll_30
        feat_row.at[0, 'promo_lag_1'] = int(promo_lag)
        
        pred_result = model.predict(feat_row, step=idx)
        pred_val = float(pred_result['predictions'][0])
        
        predictions.append(pred_val)
        sales_buffer.append(pred_val)
    return np.array(predictions)

def run_benchmark():
    print("=== Outlier Smoothing Pre-Training Benchmark ===")
    dataset_path = os.path.join(backend_dir, "data", "synthetic_retail_data.csv")
    if not os.path.exists(dataset_path):
        print("Error: synthetic dataset not found. Run verify_training.py first.")
        return
        
    df_raw = pd.read_csv(dataset_path)
    df_clean = clean_dataset(df_raw)
    df_product = aggregate_to_product_level(df_clean)
    
    # 1. Run without smoothing
    df_feat_raw = build_features(df_product)
    
    # 2. Run with smoothing
    df_smoothed = smooth_outliers(df_product)
    df_feat_smoothed = build_features(df_smoothed)
    
    products = df_product['product_id'].unique()
    
    results = []
    
    for pid in products:
        # Standard actuals (we always evaluate against true historical sales)
        prod_raw = df_feat_raw[df_feat_raw['product_id'] == pid].copy()
        train_raw = prod_raw.iloc[:-30].copy()
        test_raw = prod_raw.iloc[-30:].copy()
        y_test = test_raw['units_sold'].values
        
        # Smoothed features (only used to fit the model)
        prod_smooth = df_feat_smoothed[df_feat_smoothed['product_id'] == pid].copy()
        train_smooth = prod_smooth.iloc[:-30].copy()
        
        # Evaluate Gradient Boosting
        rf_raw = GradientBoostingModel()
        rf_raw.fit(train_raw)
        rf_raw_preds = recursive_evaluate(rf_raw, train_raw, test_raw)
        rf_raw_metrics = evaluate_predictions(y_test, rf_raw_preds)
        
        rf_smooth = GradientBoostingModel()
        rf_smooth.fit(train_smooth)
        rf_smooth_preds = recursive_evaluate(rf_smooth, train_smooth, test_raw) # Evaluate on actual y_test
        rf_smooth_metrics = evaluate_predictions(y_test, rf_smooth_preds)
        
        # Evaluate Ridge Regression
        ridge_raw = RidgeRegressionModel()
        ridge_raw.fit(train_raw)
        ridge_raw_preds = recursive_evaluate(ridge_raw, train_raw, test_raw)
        ridge_raw_metrics = evaluate_predictions(y_test, ridge_raw_preds)
        
        ridge_smooth = RidgeRegressionModel()
        ridge_smooth.fit(train_smooth)
        ridge_smooth_preds = recursive_evaluate(ridge_smooth, train_smooth, test_raw)
        ridge_smooth_metrics = evaluate_predictions(y_test, ridge_smooth_preds)
        
        results.append({
            "product_id": pid,
            "RF_raw_MAE": rf_raw_metrics["MAE"],
            "RF_smooth_MAE": rf_smooth_metrics["MAE"],
            "Ridge_raw_MAE": ridge_raw_metrics["MAE"],
            "Ridge_smooth_MAE": ridge_smooth_metrics["MAE"]
        })
        
    df_results = pd.DataFrame(results)
    print("\n--- Benchmark Results Table (MAE) ---")
    print(df_results.to_string(index=False))
    
    rf_raw_avg = df_results["RF_raw_MAE"].mean()
    rf_smooth_avg = df_results["RF_smooth_MAE"].mean()
    ridge_raw_avg = df_results["Ridge_raw_MAE"].mean()
    ridge_smooth_avg = df_results["Ridge_smooth_MAE"].mean()
    
    print("\n--- Summary Averages ---")
    print(f"Gradient Boosting (Raw):             {rf_raw_avg:.2f} MAE")
    print(f"Gradient Boosting (MAD Smoothed):    {rf_smooth_avg:.2f} MAE")
    print(f"Ridge Regression (Raw):          {ridge_raw_avg:.2f} MAE")
    print(f"Ridge Regression (MAD Smoothed): {ridge_smooth_avg:.2f} MAE")
    
    rf_change = ((rf_smooth_avg - rf_raw_avg) / rf_raw_avg) * 100
    ridge_change = ((ridge_smooth_avg - ridge_raw_avg) / ridge_raw_avg) * 100
    
    print(f"\nGradient Boosting MAE Change:        {rf_change:+.2f}%")
    print(f"Ridge Regression MAE Change:     {ridge_change:+.2f}%")
    
    if rf_smooth_avg <= rf_raw_avg:
        print("\nConclusion: Outlier smoothing IMPROVED accuracy (reduced average MAE) for Gradient Boosting.")
    else:
        print("\nConclusion: Outlier smoothing DEGRADED accuracy (increased average MAE) for Gradient Boosting.")

if __name__ == "__main__":
    run_benchmark()
