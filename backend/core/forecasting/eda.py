import pandas as pd
import numpy as np

EMPTY_EDA_REPORT = {
    "dataset_overview": {}, "descriptive_statistics": {}, "top_products": [],
    "category_performance": [], "sales_trend": {"dates": [], "sales": []},
    "weekly_seasonality": {"labels": [], "values": []},
    "monthly_seasonality": {"labels": [], "values": []},
    "correlation_matrix": {"columns": [], "matrix": []},
    "outliers": [], "outliers_count": 0,
    "revenue_by_category": [], "monthly_revenue_trend": {"labels": [], "values": []},
    "stock_status_distribution": [], "fast_slow_movers": {"fastest_growing": [], "slowest_moving": []},
    "promotion_impact": [],
}

def generate_eda_report(
    df_clean: pd.DataFrame,
    product_ids: list = None,
    categories: list = None,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on the standardized, cleaned dataset.
    Returns structured data that can be directly rendered as Plotly charts and tables in the UI.

    product_ids/categories/start_date/end_date apply an optional filter before any aggregation.
    Only allocates a filtered copy when a filter is actually supplied, so the unfiltered (default)
    call path — the one the caller caches — pays no extra cost.
    """
    if product_ids or categories or start_date or end_date:
        df_clean = df_clean.copy()
        if product_ids:
            df_clean = df_clean[df_clean['product_id'].isin(product_ids)]
        if categories:
            df_clean = df_clean[df_clean['category'].isin(categories)]
        if start_date:
            df_clean = df_clean[df_clean['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df_clean = df_clean[df_clean['date'] <= pd.to_datetime(end_date)]

    if len(df_clean) == 0:
        return dict(EMPTY_EDA_REPORT)

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
    # A column with zero variance in the (possibly filtered) window — e.g. 'month' when
    # a date filter narrows the range to a single calendar month — makes Pearson
    # correlation mathematically undefined (NaN), which isn't valid JSON. Undefined-due-
    # to-no-variance is display-equivalent to "no relationship observed", so it's filled
    # with 0 rather than left as NaN.
    corr_matrix = corr_matrix.fillna(0.0)

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

    # 9. Revenue by Category — computed per-row (units_sold * price) rather than the
    # total_sales * avg_price approximation used for top_products, since this is the
    # headline revenue number rather than a quick ranking.
    df_clean['_revenue'] = df_clean['units_sold'] * df_clean['price']
    rev_by_cat = df_clean.groupby('category')['_revenue'].sum().round(2).reset_index()
    rev_by_cat = rev_by_cat.rename(columns={'_revenue': 'total_revenue'})
    rev_by_cat = rev_by_cat.sort_values(by='total_revenue', ascending=False)
    report["revenue_by_category"] = rev_by_cat.to_dict(orient='records')

    # 10. Monthly Revenue Trend — revenue over calendar time (distinct from the
    # month-of-year seasonality above, which averages Jan-of-every-year together).
    df_clean['_year_month'] = df_clean['date'].dt.to_period('M').astype(str)
    monthly_rev = df_clean.groupby('_year_month')['_revenue'].sum().round(2).sort_index()
    report["monthly_revenue_trend"] = {
        "labels": monthly_rev.index.tolist(),
        "values": monthly_rev.values.tolist()
    }

    # 11. Stock Status Distribution — reuses df_prod (already computed above for the
    # correlation matrix) and the exact same thresholds as the Forecast page's per-product
    # decision support (train_pipeline.py's generate_future_forecast), so the two pages
    # never disagree about what "low stock" means.
    status_counts = {"OUT_OF_STOCK": 0, "CRITICAL_LOW": 0, "LOW_STOCK": 0, "HEALTHY": 0}
    for pid, group in df_prod.groupby('product_id'):
        current_stock = float(group.sort_values('date')['stock_on_hand'].iloc[-1])
        avg_daily_sales = float(group['units_sold'].mean()) or 1.0
        safety_stock_threshold = 2.0 * avg_daily_sales
        if current_stock == 0:
            status_counts["OUT_OF_STOCK"] += 1
        elif current_stock < safety_stock_threshold:
            status_counts["CRITICAL_LOW"] += 1
        elif current_stock < 5.0 * avg_daily_sales:
            status_counts["LOW_STOCK"] += 1
        else:
            status_counts["HEALTHY"] += 1
    report["stock_status_distribution"] = [
        {"status": status, "count": count} for status, count in status_counts.items()
    ]

    # 12. Fast / Slow Movers — last 30 days vs. the prior 30 days, per product.
    max_date = df_prod['date'].max()
    recent_start = max_date - pd.Timedelta(days=29)
    prior_start = recent_start - pd.Timedelta(days=30)
    prior_end = recent_start - pd.Timedelta(days=1)

    recent_sales = df_prod[df_prod['date'] >= recent_start].groupby('product_id')['units_sold'].sum()
    prior_sales = df_prod[(df_prod['date'] >= prior_start) & (df_prod['date'] <= prior_end)].groupby('product_id')['units_sold'].sum()
    movers = pd.DataFrame({'recent_sales': recent_sales, 'prior_sales': prior_sales}).fillna(0)
    # Only rank products with enough history in both windows to make % change meaningful
    movers = movers[movers['prior_sales'] > 0]
    movers['pct_change'] = round(((movers['recent_sales'] - movers['prior_sales']) / movers['prior_sales']) * 100, 1)
    movers = movers.reset_index().merge(
        df_clean[['product_id', 'product_name']].drop_duplicates(subset=['product_id']),
        on='product_id', how='left'
    )
    movers_sorted = movers.sort_values(by='pct_change', ascending=False)
    report["fast_slow_movers"] = {
        "fastest_growing": movers_sorted.head(5)[['product_id', 'product_name', 'pct_change', 'recent_sales', 'prior_sales']].to_dict(orient='records'),
        "slowest_moving": movers_sorted.tail(5)[['product_id', 'product_name', 'pct_change', 'recent_sales', 'prior_sales']].sort_values(by='pct_change').to_dict(orient='records')
    }

    # 13. Promotion Impact — average units sold on vs. off promotion days, per category
    # plus an overall "All Categories" summary row.
    promo_rows = []
    for cat, group in list(df_clean.groupby('category')) + [("All Categories", df_clean)]:
        on_promo = group[group['promotion_flag'] == 1]['units_sold']
        off_promo = group[group['promotion_flag'] == 0]['units_sold']
        avg_on = float(on_promo.mean()) if len(on_promo) else 0.0
        avg_off = float(off_promo.mean()) if len(off_promo) else 0.0
        lift_pct = round(((avg_on - avg_off) / avg_off) * 100, 1) if avg_off > 0 else 0.0
        promo_rows.append({
            "category": cat,
            "avg_units_on_promo": round(avg_on, 2),
            "avg_units_off_promo": round(avg_off, 2),
            "lift_pct": lift_pct
        })
    # "All Categories" first for a quick top-line read, then categories by promo lift
    overall_row = [r for r in promo_rows if r["category"] == "All Categories"]
    category_rows = sorted([r for r in promo_rows if r["category"] != "All Categories"], key=lambda r: r["lift_pct"], reverse=True)
    report["promotion_impact"] = overall_row + category_rows

    df_clean.drop(columns=['_revenue', '_year_month'], inplace=True)

    return report
