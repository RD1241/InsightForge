import os
import pandas as pd
import numpy as np

# A realistic catalog of 50 products to map bare item IDs (like in the Kaggle dataset)
# to rich product names, categories, and baseline prices for the dashboard
PRODUCT_CATALOG = {
    1: ("Organic Whole Milk 1L", "Dairy", 3.49),
    2: ("Wheat Sliced Bread", "Bakery", 2.89),
    3: ("Free-Range Eggs 12pk", "Dairy", 4.99),
    4: ("Fuji Apples 1kg", "Produce", 3.99),
    5: ("Natural Shampoo 400ml", "Personal Care", 7.99),
    6: ("Greek Yogurt 500g", "Dairy", 4.29),
    7: ("White Sub Rolls 4pk", "Bakery", 1.99),
    8: ("Fresh Bananas 1kg", "Produce", 1.89),
    9: ("Orange Juice 1.5L", "Beverages", 4.49),
    10: ("Arabica Coffee Beans 250g", "Pantry", 8.99),
    11: ("English Breakfast Tea 50pk", "Pantry", 3.99),
    12: ("Soda Can 6pk", "Beverages", 5.49),
    13: ("Cheddar Cheese 250g", "Dairy", 3.79),
    14: ("Chocolate Chip Cookies", "Snacks", 2.99),
    15: ("Potato Chips Salted 150g", "Snacks", 2.49),
    16: ("Sparkling Water 1.5L", "Beverages", 1.29),
    17: ("Salted Butter 250g", "Dairy", 2.99),
    18: ("Croissants 4pk", "Bakery", 3.49),
    19: ("Fresh Strawberries 250g", "Produce", 2.99),
    20: ("Spaghetti Pasta 500g", "Pantry", 1.49),
    21: ("Tomato Pasta Sauce 400g", "Pantry", 2.29),
    22: ("Basmati Rice 1kg", "Pantry", 3.19),
    23: ("Extra Virgin Olive Oil 500ml", "Pantry", 9.99),
    24: ("Frozen Pizza 400g", "Frozen", 5.99),
    25: ("Vanilla Ice Cream 1L", "Frozen", 4.99),
    26: ("Dishwashing Liquid 500ml", "Household", 2.49),
    27: ("Laundry Detergent 1.5L", "Household", 10.99),
    28: ("Toilet Paper 9pk", "Household", 5.99),
    29: ("Toothpaste Mint 100ml", "Personal Care", 2.99),
    30: ("Shower Gel Lemon 250ml", "Personal Care", 3.49),
    31: ("Paper Towels 2pk", "Household", 2.29),
    32: ("Trash Bags 30L 20pk", "Household", 3.19),
    33: ("Canned Tuna in Oil 160g", "Pantry", 1.89),
    34: ("Canned Sweet Corn 400g", "Pantry", 1.19),
    35: ("Unsalted Peanuts 200g", "Snacks", 2.29),
    36: ("Dark Chocolate Bar 100g", "Snacks", 2.79),
    37: ("Multivitamin 60pk", "Personal Care", 12.99),
    38: ("Frozen Peas 500g", "Frozen", 1.79),
    39: ("Frozen French Fries 1kg", "Frozen", 3.49),
    40: ("Fresh Lemons 500g", "Produce", 1.99),
    41: ("Red Onion 1kg", "Produce", 2.19),
    42: ("White Sugar 1kg", "Pantry", 1.69),
    43: ("Sea Salt Grinder 100g", "Pantry", 2.49),
    44: ("Black Pepper Grinder 50g", "Pantry", 3.29),
    45: ("Hand Wash Soap 250ml", "Personal Care", 1.99),
    46: ("Dishwashing Tablets 30pk", "Household", 8.99),
    47: ("Facial Tissues 100pk", "Household", 1.79),
    48: ("Dog Kibble 2kg", "Household", 9.99),
    49: ("Cat Wet Food 4pk", "Household", 3.89),
    50: ("Baby Wipes 80pk", "Personal Care", 2.49)
}

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes input dataset columns dynamically.
    Maps:
      - 'sales' or 'units_sold' -> 'units_sold'
      - 'item' or 'product_id' -> 'product_id'
      - 'store' or 'store_id' -> 'store_id'
    
    Generates missing metadata (names, categories, prices, stock, promotions)
    if they are not present, making any dataset compatible with the application's rich features.
    """
    df_std = df.copy()
    
    # 1. Map Columns
    col_mapping = {
        'sales': 'units_sold',
        'item': 'product_id',
        'store': 'store_id'
    }
    df_std = df_std.rename(columns=col_mapping)
    
    # Ensure columns are string types for categorical IDs
    if 'product_id' in df_std.columns:
        df_std['product_id'] = df_std['product_id'].astype(str)
    if 'store_id' in df_std.columns:
        df_std['store_id'] = df_std['store_id'].astype(str)
        
    # Convert units_sold to numeric
    if 'units_sold' in df_std.columns:
        df_std['units_sold'] = pd.to_numeric(df_std['units_sold'], errors='coerce')
        
    # 2. Enrich Missing Columns from Catalog
    unique_products = df_std['product_id'].unique() if 'product_id' in df_std.columns else []
    
    # Check if we need to inject names, categories, or prices
    has_name = 'product_name' in df_std.columns
    has_cat = 'category' in df_std.columns
    has_price = 'price' in df_std.columns
    
    if not (has_name and has_cat and has_price):
        # Create mapping lookups
        name_map = {}
        cat_map = {}
        price_map = {}
        
        for pid_str in unique_products:
            # Try parsing to int for catalog mapping
            try:
                pid_int = int(float(pid_str))
            except ValueError:
                pid_int = -1
                
            if pid_int in PRODUCT_CATALOG:
                name, cat, price = PRODUCT_CATALOG[pid_int]
            else:
                # Fallback generator for out-of-catalog items
                name = f"Item {pid_str}"
                cat = "General"
                # Hash-based deterministic price between 1.99 and 10.99
                price = round(2.0 + (hash(pid_str) % 9) + 0.99, 2)
                
            name_map[pid_str] = name
            cat_map[pid_str] = cat
            price_map[pid_str] = price
            
        if not has_name:
            df_std['product_name'] = df_std['product_id'].map(name_map)
        if not has_cat:
            df_std['category'] = df_std['product_id'].map(cat_map)
        if not has_price:
            df_std['price'] = df_std['product_id'].map(price_map)
            
    # 3. Handle Promotion Flag
    if 'promotion_flag' not in df_std.columns:
        # If dataset contains 'promotion', rename it
        if 'promotion' in df_std.columns:
            df_std = df_std.rename(columns={'promotion': 'promotion_flag'})
        else:
            # Default to 0
            df_std['promotion_flag'] = 0
            
    df_std['promotion_flag'] = pd.to_numeric(df_std['promotion_flag'], errors='coerce').fillna(0).astype(int)
    
    # 4. Handle Stock on Hand
    # If not present, we will simulate stock levels based on sales
    if 'stock_on_hand' not in df_std.columns:
        if 'stock' in df_std.columns:
            df_std = df_std.rename(columns={'stock': 'stock_on_hand'})
        else:
            # Simulate a stock level: we start each store-product group with 5x average sales
            # and decrement/replenish. To do this efficiently, we can compute rolling averages
            df_std['stock_on_hand'] = 0
            
            # Group by product/store to compute average sales rate
            sales_avg = df_std.groupby(['store_id', 'product_id'])['units_sold'].transform('mean')
            # Fallback for empty/NaN
            sales_avg = sales_avg.fillna(10.0)
            
            # Stock replenishes to 7x mean sales whenever it falls below 2x mean sales
            # We can approximate the daily stock by starting high and simulating depletions
            # Since simulation over 900k rows can be slow if done step-by-step, we use a vectorized cycle:
            # Day number mod 7 is used to reset stock to a baseline minus sales depletions.
            df_std['date_temp'] = pd.to_datetime(df_std['date'])
            day_cycle = (df_std['date_temp'] - df_std['date_temp'].min()).dt.days % 7
            df_std['stock_on_hand'] = np.maximum(
                int(10), 
                np.round(sales_avg * (8 - day_cycle) - df_std['units_sold']).astype(int)
            )
            df_std = df_std.drop(columns=['date_temp'])
            
    df_std['stock_on_hand'] = pd.to_numeric(df_std['stock_on_hand'], errors='coerce').fillna(0).astype(int)
    
    return df_std

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
    
    # Standardize columns internally to run checks
    try:
        df_mapped = standardize_dataframe(df)
    except Exception as e:
        report["is_valid"] = False
        report["errors"].append(f"Failed to map columns to standardized schema: {str(e)}")
        return report
        
    # Check essential columns after mapping
    required_cols = ['date', 'store_id', 'product_id', 'units_sold']
    missing = [c for c in required_cols if c not in df_mapped.columns]
    if missing:
        report["is_valid"] = False
        report["errors"].append(f"Missing essential columns after schema detection: {', '.join(missing)}")
        return report
        
    # 2. Basic Dataset Info
    report["stats"]["row_count"] = len(df_mapped)
    report["stats"]["unique_stores"] = int(df_mapped['store_id'].nunique())
    report["stats"]["unique_products"] = int(df_mapped['product_id'].nunique())
    report["stats"]["unique_categories"] = int(df_mapped['category'].nunique())
    
    # 3. Date Parsing & Range
    try:
        temp_date = pd.to_datetime(df_mapped['date'], errors='coerce')
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
    null_counts = df_mapped.isnull().sum().to_dict()
    report["stats"]["null_counts"] = null_counts
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        report["warnings"].append(f"Found {total_nulls} total missing values across columns.")
        
    # 5. Check Negative Values in crucial columns
    if (df_mapped['units_sold'] < 0).any():
        report["errors"].append("units_sold contains negative values.")
        report["is_valid"] = False
    if (df_mapped['price'] < 0).any():
        report["errors"].append("price contains negative values.")
        report["is_valid"] = False
        
    return report

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by removing duplicates, parsing dates,
    coercing data types, and handling missing values.
    """
    df_mapped = standardize_dataframe(df)
    
    # Drop duplicates
    df_clean = df_mapped.drop_duplicates()
    
    # Parse dates
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['date'])
    
    # Handle missing values
    df_clean['price'] = df_clean.groupby('product_id')['price'].transform(lambda x: x.ffill().bfill())
    df_clean['price'] = df_clean['price'].fillna(1.99)
    
    df_clean['units_sold'] = df_clean['units_sold'].fillna(0).astype(int)
    
    # Fill stock_on_hand with forward fill then 0
    df_clean['stock_on_hand'] = df_clean.groupby(['store_id', 'product_id'])['stock_on_hand'].transform(lambda x: x.ffill().bfill())
    df_clean['stock_on_hand'] = df_clean['stock_on_hand'].fillna(10).astype(int)
    df_clean.loc[df_clean['stock_on_hand'] < 0, 'stock_on_hand'] = 0
    
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


def ensure_regular_daily_grid(df_product: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantees a strict, continuous daily calendar for every product.

    Retail datasets frequently omit dates where zero units were sold (no
    transaction recorded), creating invisible gaps in the time series.  When
    lag features are computed with .shift(N) on a gapped series the N-th shift
    no longer corresponds to exactly N calendar days — it silently picks up
    data from N *rows* ago, which may be several real days earlier.  This
    corrupts every lag and rolling-window feature.

    This function:
      1. Builds the complete global date range (global min → global max).
      2. Reindexes each product independently to that range.
      3. Fills missing sales and promotion_flag with 0 (no transaction = 0 units sold).
      4. Forward-fills then back-fills price and stock_on_hand (last known
         price / stock persists until a new record is observed).
      5. Carries metadata columns (product_name, category) forward.

    Returns a DataFrame with no date gaps, ready for lag/rolling feature
    engineering.
    """
    global_min = df_product['date'].min()
    global_max = df_product['date'].max()
    full_date_range = pd.date_range(start=global_min, end=global_max, freq='D')

    filled_parts = []
    for pid, grp in df_product.groupby('product_id'):
        grp = grp.set_index('date').sort_index()
        # Reindex to full calendar
        grp = grp.reindex(full_date_range)

        # Restore product_id (lost after reindex)
        grp['product_id'] = pid

        # Zero-fill sales and promotions — absence of transaction = 0 sold
        grp['units_sold'] = grp['units_sold'].fillna(0).astype(int)
        grp['promotion_flag'] = grp['promotion_flag'].fillna(0).astype(int)

        # Forward-fill then back-fill slowly-changing columns
        for col in ['price', 'stock_on_hand']:
            if col in grp.columns:
                grp[col] = grp[col].ffill().bfill()

        # Carry metadata forward (product_name, category may be NaN on filled rows)
        for col in ['product_name', 'category']:
            if col in grp.columns:
                grp[col] = grp[col].ffill().bfill()

        # Restore date as a regular column
        grp.index.name = 'date'
        grp = grp.reset_index()
        filled_parts.append(grp)

    df_aligned = pd.concat(filled_parts, ignore_index=True)
    df_aligned = df_aligned.sort_values(by=['product_id', 'date']).reset_index(drop=True)
    return df_aligned

def build_features(df_product: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers time series features on product-aggregated dataframe.

    IMPORTANT — daily grid alignment is performed first:
    Before any shift/rolling operation the dataframe is passed through
    ensure_regular_daily_grid() so that every product has a continuous
    daily entry.  This guarantees that shift(N) always corresponds to
    exactly N *calendar* days, not N rows.

    Features engineered:
    - Calendar: day_of_week, month, is_weekend, day_of_year
    - Lags: 1, 7, 14 days
    - Rolling windows: 7-day mean, 30-day mean (shift-1 to avoid leakage)
    """
    # Align to complete daily calendar BEFORE computing any lag / rolling features
    df_feat = ensure_regular_daily_grid(df_product)

    # Calendar features
    df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)
    df_feat['day_of_year'] = df_feat['date'].dt.dayofyear

    # Lag features — grouped per product so lags never cross product boundaries
    df_feat['units_sold_lag_1']  = df_feat.groupby('product_id')['units_sold'].shift(1)
    df_feat['units_sold_lag_7']  = df_feat.groupby('product_id')['units_sold'].shift(7)
    df_feat['units_sold_lag_14'] = df_feat.groupby('product_id')['units_sold'].shift(14)

    # Promotion lag feature - captures the post-promotion hangover effect
    df_feat['promo_lag_1'] = df_feat.groupby('product_id')['promotion_flag'].shift(1)

    # Rolling averages — shift(1) applied inside transform to prevent target leakage
    df_feat['units_sold_roll_mean_7'] = df_feat.groupby('product_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df_feat['units_sold_roll_mean_30'] = df_feat.groupby('product_id')['units_sold'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    )

    # Fill any remaining NaNs at the start of each product's history
    features_to_fill = [
        'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
        'units_sold_roll_mean_7', 'units_sold_roll_mean_30', 'promo_lag_1'
    ]
    for col in features_to_fill:
        df_feat[col] = df_feat.groupby('product_id')[col].transform(
            lambda x: x.bfill().ffill().fillna(0.0)
        )

    return df_feat
