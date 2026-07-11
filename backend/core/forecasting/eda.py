import pandas as pd
import numpy as np

def generate_eda_report(df_clean: pd.DataFrame) -> dict:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on the standardized, cleaned dataset.
    Returns structured data that can be directly rendered as Plotly charts and tables in the UI.
    """
    report = {}
    
    # 1. Dataset Overview & Data Types
    total_rows = len(df_clean)
    missing_report = {}
    dtypes_report = {}
    for col in df_clean.columns:
        null_count = int(df_clean[col].isnull().sum())
        missing_report[col] = {
            "null_count": null_count,
            "null_percentage": round((null_count / total_rows) * 100, 2)
        }
        dtypes_report[col] = str(df_clean[col].dtype)
        
    report["dataset_overview"] = {
        "total_records": total_rows,
        "unique_stores": int(df_clean['store_id'].nunique()),
        "unique_products": int(df_clean['product_id'].nunique()),
        "unique_categories": int(df_clean['category'].nunique()),
        "start_date": df_clean['date'].min().strftime("%Y-%m-%d"),
        "end_date": df_clean['date'].max().strftime("%Y-%m-%d"),
        "missing_values": missing_report,
        "data_types": dtypes_report
    }
    
    # 2. Descriptive Statistics (Numerical Columns)
    desc_stats = {}
    num_cols = ['units_sold', 'price', 'stock_on_hand']
    for col in num_cols:
        if col in df_clean.columns:
            stats = df_clean[col].describe()
            desc_stats[col] = {
                "count": int(stats["count"]),
                "mean": round(float(stats["mean"]), 2),
                "std": round(float(stats["std"]), 2),
                "min": round(float(stats["min"]), 2),
                "q25": round(float(stats["25%"]), 2),
                "median": round(float(stats["50%"]), 2),
                "q75": round(float(stats["75%"]), 2),
                "max": round(float(stats["max"]), 2)
            }
    report["descriptive_statistics"] = desc_stats
    
    # 3. Product Sales Performance
    # Aggregate sales and revenue per product
    product_agg = df_clean.groupby(['product_id', 'product_name', 'category']).agg(
        total_sales=('units_sold', 'sum'),
        avg_price=('price', 'mean')
    ).reset_index()
    product_agg['total_revenue'] = round(product_agg['total_sales'] * product_agg['avg_price'], 2)
    product_agg['total_sales'] = product_agg['total_sales'].astype(int)
    
    # Sort for top products
    top_selling = product_agg.sort_values(by='total_sales', ascending=False).head(5)
    report["top_products"] = top_selling.to_dict(orient='records')
    
    # 4. Category Performance
    cat_agg = df_clean.groupby('category').agg(
        total_sales=('units_sold', 'sum'),
        total_records=('units_sold', 'count')
    ).reset_index()
    cat_agg['total_sales'] = cat_agg['total_sales'].astype(int)
    report["category_performance"] = cat_agg.to_dict(orient='records')
    
    # 5. Daily Sales Trend (aggregated product level)
    daily_sales = df_clean.groupby('date')['units_sold'].sum().reset_index()
    report["sales_trend"] = {
        "dates": daily_sales['date'].dt.strftime("%Y-%m-%d").tolist(),
        "sales": daily_sales['units_sold'].astype(int).tolist()
    }
    
    # 6. Seasonality Analysis
    # 6a. Weekly Seasonality (Averages by Day of Week)
    # Monday=0, Sunday=6
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
    weekly_avg = df_clean.groupby('day_of_week')['units_sold'].mean().reset_index()
    days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    weekly_avg['day_name'] = weekly_avg['day_of_week'].map(days_map)
    report["weekly_seasonality"] = {
        "labels": weekly_avg['day_name'].tolist(),
        "values": np.round(weekly_avg['units_sold'].values, 2).tolist()
    }
    
    # 6b. Monthly Seasonality (Averages by Month)
    df_clean['month'] = df_clean['date'].dt.month
    monthly_avg = df_clean.groupby('month')['units_sold'].mean().reset_index()
    months_map = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    monthly_avg['month_name'] = monthly_avg['month'].map(months_map)
    report["monthly_seasonality"] = {
        "labels": monthly_avg['month_name'].tolist(),
        "values": np.round(monthly_avg['units_sold'].values, 2).tolist()
    }
    
    # 7. Correlation Heatmap (using product aggregated variables)
    # Group by product and date to do heatmap on model variables
    df_prod = df_clean.groupby(['date', 'product_id']).agg({
        'units_sold': 'sum',
        'price': 'mean',
        'stock_on_hand': 'sum',
        'promotion_flag': 'max',
    }).reset_index()
    
    df_prod['day_of_week'] = df_prod['date'].dt.dayofweek
    df_prod['month'] = df_prod['date'].dt.month
    df_prod['is_weekend'] = df_prod['day_of_week'].isin([5, 6]).astype(int)
    
    corr_cols = ['units_sold', 'price', 'stock_on_hand', 'promotion_flag', 'day_of_week', 'is_weekend', 'month']
    corr_matrix = df_prod[corr_cols].corr()
    
    report["correlation_matrix"] = {
        "columns": corr_cols,
        "matrix": np.round(corr_matrix.values, 3).tolist()
    }
    
    # 8. Outlier Detection (IQR Method)
    # Detect outliers per product time series
    outliers_list = []
    
    # For speed, aggregate to daily totals per product first
    for pid, group in df_prod.groupby('product_id'):
        q1 = group['units_sold'].quantile(0.25)
        q3 = group['units_sold'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        outliers = group[(group['units_sold'] < lower_bound) | (group['units_sold'] > upper_bound)]
        
        # Get up to 10 sample outliers for display
        for _, row in outliers.head(10).iterrows():
            # Get name from df_clean
            p_name = df_clean[df_clean['product_id'] == pid]['product_name'].iloc[0]
            outliers_list.append({
                "product_id": pid,
                "product_name": p_name,
                "date": row['date'].strftime("%Y-%m-%d"),
                "units_sold": int(row['units_sold']),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2)
            })
            
    report["outliers"] = outliers_list
    report["outliers_count"] = len(outliers_list)
    
    return report
