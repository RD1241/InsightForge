import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import preprocessor functions
from core.forecasting.preprocessor import clean_dataset, aggregate_to_product_level
from core.forecasting.registry import get_best_model_metadata, get_product_models, load_training_report
from core.forecasting.train_pipeline import generate_future_forecast

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIVE_DATASET_PATH = os.path.join(BASE_DIR, "data", "active_dataset.csv")

def get_loaded_df():
    """
    Helper to load and clean the active dataset.
    """
    if not os.path.exists(ACTIVE_DATASET_PATH):
        raise FileNotFoundError("No active dataset loaded. Please upload a dataset or load the demo first.")
    df = pd.read_csv(ACTIVE_DATASET_PATH)
    return clean_dataset(df)

# --- Core Analyst Tools ---

def top_selling_products(limit: int = 5) -> dict:
    """
    Returns the top selling products by total sales volume and total revenue.
    """
    df = get_loaded_df()
    
    # Group by product
    product_totals = df.groupby(['product_id', 'product_name', 'category']).agg(
        total_sales=('units_sold', 'sum'),
        avg_price=('price', 'mean')
    ).reset_index()
    
    product_totals['total_revenue'] = round(product_totals['total_sales'] * product_totals['avg_price'], 2)
    product_totals['total_sales'] = product_totals['total_sales'].astype(int)
    
    top_products = product_totals.sort_values(by='total_sales', ascending=False).head(limit)
    
    return {
        "status": "success",
        "metric_type": "top_selling_products",
        "data": top_products.to_dict(orient="records")
    }

def low_stock_products(threshold_days: int = 10) -> dict:
    """
    Identifies products running low on stock based on their daily sales rate
    over the last 30 days compared to current stock.
    """
    df = get_loaded_df()
    
    # Get current stock levels (on maximum date in dataset)
    max_date = df['date'].max()
    current_stock_df = df[df['date'] == max_date].groupby(['product_id', 'product_name'])['stock_on_hand'].sum().reset_index()
    
    # Calculate average daily sales rate (last 30 days of history)
    cutoff_date = max_date - timedelta(days=30)
    recent_sales = df[df['date'] > cutoff_date].groupby('product_id')['units_sold'].mean().reset_index()
    recent_sales = recent_sales.rename(columns={'units_sold': 'avg_daily_sales'})
    
    # Merge stock and sales
    merged = pd.merge(current_stock_df, recent_sales, on='product_id', how='left')
    merged['avg_daily_sales'] = merged['avg_daily_sales'].fillna(1.0) # avoid division by zero
    
    # Days of stock left = Current Stock / Daily Sales Rate
    merged['days_of_stock_remaining'] = round(merged['stock_on_hand'] / merged['avg_daily_sales'], 1)
    
    # Filter for products below threshold
    low_stock = merged[merged['days_of_stock_remaining'] <= threshold_days].sort_values(by='days_of_stock_remaining')
    
    return {
        "status": "success",
        "metric_type": "low_stock_products",
        "threshold_days": threshold_days,
        "data": low_stock.to_dict(orient="records")
    }

def inventory_health() -> dict:
    """
    Computes overall inventory health metrics (overstock, low-stock, and out-of-stock counts).
    """
    df = get_loaded_df()
    
    max_date = df['date'].max()
    current_stock = df[df['date'] == max_date].groupby(['product_id', 'product_name', 'category'])['stock_on_hand'].sum().reset_index()
    
    # Get sales rates
    cutoff_date = max_date - timedelta(days=30)
    sales_rates = df[df['date'] > cutoff_date].groupby('product_id')['units_sold'].mean().reset_index()
    sales_rates = sales_rates.rename(columns={'units_sold': 'avg_daily_sales'})
    
    merged = pd.merge(current_stock, sales_rates, on='product_id', how='left')
    merged['avg_daily_sales'] = merged['avg_daily_sales'].fillna(1.0)
    merged['days_of_stock'] = merged['stock_on_hand'] / merged['avg_daily_sales']
    
    # Categorize stock levels
    # Understock: < 5 days, Low Stock: 5-10 days, Healthy: 10-30 days, Overstock: > 30 days
    out_of_stock = len(merged[merged['stock_on_hand'] == 0])
    under_stock = len(merged[(merged['days_of_stock'] < 5) & (merged['stock_on_hand'] > 0)])
    low_stock = len(merged[(merged['days_of_stock'] >= 5) & (merged['days_of_stock'] < 10)])
    healthy_stock = len(merged[(merged['days_of_stock'] >= 10) & (merged['days_of_stock'] <= 30)])
    over_stock = len(merged[merged['days_of_stock'] > 30])
    
    return {
        "status": "success",
        "metric_type": "inventory_health",
        "total_active_products": len(merged),
        "health_summary": {
            "out_of_stock_products": out_of_stock,
            "understock_products": under_stock,
            "low_stock_products": low_stock,
            "healthy_stock_products": healthy_stock,
            "overstock_products": over_stock
        }
    }

def sales_summary() -> dict:
    """
    Generates a high-level summary of total sales, total revenue, average transaction price,
    and category breakdowns.
    """
    df = get_loaded_df()
    
    total_sales = int(df['units_sold'].sum())
    # Estimate total revenue
    df['revenue'] = df['units_sold'] * df['price']
    total_revenue = round(float(df['revenue'].sum()), 2)
    avg_price = round(float(df['price'].mean()), 2)
    
    category_summary = df.groupby('category').agg(
        sales=('units_sold', 'sum'),
        revenue=('revenue', 'sum')
    ).reset_index()
    category_summary['sales'] = category_summary['sales'].astype(int)
    category_summary['revenue'] = np.round(category_summary['revenue'].values, 2)
    
    return {
        "status": "success",
        "metric_type": "sales_summary",
        "total_sales_volume": total_sales,
        "total_revenue": total_revenue,
        "average_price": avg_price,
        "category_breakdown": category_summary.to_dict(orient="records")
    }

def compare_sales(product_id: str, period_1_days: int = 30, period_2_days: int = 30) -> dict:
    """
    Compares a product's sales volume between the last N days (period 1) 
    and the N days prior (period 2).
    """
    df = get_loaded_df()
    
    # Filter for product
    prod_df = df[df['product_id'] == product_id].copy()
    if prod_df.empty:
        return {"status": "error", "message": f"Product '{product_id}' not found."}
        
    max_date = prod_df['date'].max()
    
    # Period 1: [max_date - N days, max_date]
    p1_start = max_date - timedelta(days=period_1_days)
    df_p1 = prod_df[(prod_df['date'] > p1_start) & (prod_df['date'] <= max_date)]
    
    # Period 2: [max_date - 2N days, max_date - N days]
    p2_end = p1_start
    p2_start = p2_end - timedelta(days=period_2_days)
    df_p2 = prod_df[(prod_df['date'] > p2_start) & (prod_df['date'] <= p2_end)]
    
    p1_sales = int(df_p1['units_sold'].sum())
    p2_sales = int(df_p2['units_sold'].sum())
    
    difference = p1_sales - p2_sales
    percentage_change = round((difference / max(1, p2_sales)) * 100, 2)
    
    return {
        "status": "success",
        "metric_type": "sales_comparison",
        "product_id": product_id,
        "product_name": prod_df['product_name'].iloc[0],
        "current_period_days": period_1_days,
        "current_period_sales": p1_sales,
        "prior_period_days": period_2_days,
        "prior_period_sales": p2_sales,
        "sales_volume_difference": difference,
        "percentage_change": percentage_change
    }

def forecast_product(product_id: str) -> dict:
    """
    Retrieves the 30-day forecast and model recommendation for a specific product.
    """
    df = get_loaded_df()
    try:
        forecast_res = generate_future_forecast(df, product_id, horizon_days=30)
        # Summarize prediction variables for the LLM
        future_preds = forecast_res["forecast"]["predictions"]
        
        return {
            "status": "success",
            "metric_type": "product_forecast",
            "product_id": product_id,
            "product_name": forecast_res["product_name"],
            "category": forecast_res["category"],
            "model_used": forecast_res["model_used"],
            "validation_mae": forecast_res["metrics"].get("MAE"),
            "validation_mape": forecast_res["metrics"].get("MAPE"),
            "next_30_days_predicted_sales": int(sum(future_preds)),
            "average_daily_predicted_sales": round(float(np.mean(future_preds)), 2),
            "peak_predicted_sales": float(max(future_preds))
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def model_comparison(product_id: str) -> dict:
    """
    Retrieves metrics for all models fitted on a specific product,
    explaining why the best one is recommended.
    """
    best_meta = get_best_model_metadata(product_id)
    if not best_meta:
        return {"status": "error", "message": f"No models trained for product '{product_id}'."}
        
    all_models = get_product_models(product_id)
    
    comparison = []
    for m in all_models:
        comparison.append({
            "model_name": m["model_name"],
            "metrics": m["metrics"]
        })
        
    return {
        "status": "success",
        "metric_type": "model_comparison",
        "product_id": product_id,
        "recommended_model": best_meta["model_name"],
        "recommendation_reason": best_meta["recommendation_reason"],
        "all_models_metrics": comparison
    }

def generate_business_insights() -> dict:
    """
    Compiles automated high-level insights by auditing sales patterns,
    understock levels, and training metrics.
    """
    df = get_loaded_df()
    
    # 1. Sales stats
    summary = sales_summary()
    
    # 2. Restock alerts
    restock = low_stock_products(threshold_days=7) # critical stock out under 7 days
    
    # 3. Slow movers (products with lowest sales rate in last 30 days)
    max_date = df['date'].max()
    cutoff_date = max_date - timedelta(days=30)
    avg_sales = df[df['date'] > cutoff_date].groupby(['product_id', 'product_name'])['units_sold'].mean().reset_index()
    slow_movers = avg_sales.sort_values(by='units_sold').head(3)
    
    return {
        "status": "success",
        "metric_type": "business_insights",
        "summary": {
            "total_revenue": summary["total_revenue"],
            "total_sales_volume": summary["total_sales_volume"]
        },
        "critical_restock_alerts": restock["data"][:3], # top 3 urgent restocks
        "slow_moving_products": slow_movers.to_dict(orient="records")
    }
