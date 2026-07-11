import os
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {
    'date': 'datetime64[ns]',
    'store_id': 'object',
    'product_id': 'object',
    'product_name': 'object',
    'category': 'object',
    'units_sold': 'int64',
    'price': 'float64',
    'stock_on_hand': 'int64',
    'promotion_flag': 'int64'
}

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validates the dataset structure, checks for required columns,
    missing values, and checks data types.
    Returns a dictionary summarizing validation results.
    """
    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # 1. Column Presence Check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        report["is_valid"] = False
        report["errors"].append(f"Missing required columns: {', '.join(missing_cols)}")
        return report
        
    # 2. Basic Dataset Info
    report["stats"]["row_count"] = len(df)
    report["stats"]["unique_stores"] = int(df['store_id'].nunique())
    report["stats"]["unique_products"] = int(df['product_id'].nunique())
    report["stats"]["unique_categories"] = int(df['category'].nunique())
    
    # 3. Date Parsing & Range
    try:
        temp_date = pd.to_datetime(df['date'], errors='coerce')
        invalid_dates = temp_date.isna().sum()
        if invalid_dates > 0:
            report["warnings"].append(f"Found {invalid_dates} rows with invalid date formats.")
        
        min_date = temp_date.min()
        max_date = temp_date.max()
        report["stats"]["start_date"] = min_date.strftime("%Y-%m-%d") if not pd.isna(min_date) else None
        report["stats"]["end_date"] = max_date.strftime("%Y-%m-%d") if not pd.isna(max_date) else None
        
        if not pd.isna(min_date) and not pd.isna(max_date):
            days_span = (max_date - min_date).days
            report["stats"]["days_span"] = days_span
            if days_span < 90:
                report["warnings"].append(f"Dataset covers only {days_span} days. Time series forecasting benefits from at least 90-180 days of history.")
    except Exception as e:
        report["is_valid"] = False
        report["errors"].append(f"Failed to parse dates: {str(e)}")
        
    # 4. Check for Nulls
    null_counts = df.isnull().sum().to_dict()
    report["stats"]["null_counts"] = null_counts
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        report["warnings"].append(f"Found {total_nulls} total missing values across columns.")
        
    # 5. Check Negative Values in crucial columns
    if (df['units_sold'] < 0).any():
        report["errors"].append("units_sold contains negative values.")
        report["is_valid"] = False
    if (df['price'] < 0).any():
        report["errors"].append("price contains negative values.")
        report["is_valid"] = False
    if (df['stock_on_hand'] < 0).any():
        report["warnings"].append("stock_on_hand contains negative values (treating as zero).")
        
    return report

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by removing duplicates, parsing dates,
    coercing data types, and handling missing values.
    """
    df_clean = df.copy()
    
    # Drop duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Parse dates
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['date'])
    
    # Handle missing values
    # Fill price missing values by group median or overall median, or forward fill
    df_clean['price'] = df_clean.groupby('product_id')['price'].transform(lambda x: x.ffill().bfill())
    df_clean['price'] = df_clean['price'].fillna(0.0)
    
    # Fill units_sold with 0
    df_clean['units_sold'] = df_clean['units_sold'].fillna(0).astype(int)
    
    # Fill stock_on_hand with forward fill then 0
    df_clean['stock_on_hand'] = df_clean.groupby(['store_id', 'product_id'])['stock_on_hand'].transform(lambda x: x.ffill().bfill())
    df_clean['stock_on_hand'] = df_clean['stock_on_hand'].fillna(0).astype(int)
    df_clean.loc[df_clean['stock_on_hand'] < 0, 'stock_on_hand'] = 0
    
    # Fill promotion_flag with 0
    df_clean['promotion_flag'] = df_clean['promotion_flag'].fillna(0).astype(int)
    
    # Sort chronologically
    df_clean = df_clean.sort_values(by=['date', 'product_id', 'store_id']).reset_index(drop=True)
    
    return df_clean

def aggregate_to_product_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates store-level details to product-level time series.
    - Sum units_sold and stock_on_hand
    - Average price
    - Max promotion_flag
    - Keep product_name and category details
    """
    # Group by date and product_id to preserve the product details
    agg_funcs = {
        'product_name': 'first',
        'category': 'first',
        'units_sold': 'sum',
        'price': 'mean',
        'stock_on_hand': 'sum',
        'promotion_flag': 'max'
    }
    
    df_product = df.groupby(['date', 'product_id']).agg(agg_funcs).reset_index()
    df_product = df_product.sort_values(by=['product_id', 'date']).reset_index(drop=True)
    return df_product

def build_features(df_product: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers time series features on product-aggregated dataframe:
    - Calendar features (day_of_week, month, is_weekend, day_of_year)
    - Lag features (1, 7, 14 days)
    - Rolling window features (7, 30 days mean of units_sold)
    """
    df_feat = df_product.copy()
    
    # Calendar features
    df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)
    df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
    
    # To compute lag and rolling averages correctly, we must group by product_id
    # Sort is critical: it was sorted by ['product_id', 'date'] in aggregate_to_product_level
    
    # Lags
    df_feat['units_sold_lag_1'] = df_feat.groupby('product_id')['units_sold'].shift(1)
    df_feat['units_sold_lag_7'] = df_feat.groupby('product_id')['units_sold'].shift(7)
    df_feat['units_sold_lag_14'] = df_feat.groupby('product_id')['units_sold'].shift(14)
    
    # Rolling averages (using closed='left' to prevent data leakage from the current day's value)
    df_feat['units_sold_roll_mean_7'] = df_feat.groupby('product_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df_feat['units_sold_roll_mean_30'] = df_feat.groupby('product_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    )
    
    # Backfill/Forwardfill features that contain NaN due to lags (e.g. first few days of history)
    # Using group-level fill to avoid blending data between different products
    features_to_fill = [
        'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
        'units_sold_roll_mean_7', 'units_sold_roll_mean_30'
    ]
    for col in features_to_fill:
        df_feat[col] = df_feat.groupby('product_id')[col].transform(lambda x: x.bfill().ffill().fillna(0.0))
        
    return df_feat
