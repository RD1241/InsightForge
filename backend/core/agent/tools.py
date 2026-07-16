import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import preprocessor functions
from core.forecasting.preprocessor import clean_dataset, aggregate_to_product_level
from core.forecasting.registry import get_best_model_metadata, get_product_models, load_model, load_training_report
from core.forecasting.train_pipeline import generate_future_forecast
from routers.dataset import get_clean_df

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIVE_DATASET_PATH = os.path.join(BASE_DIR, "data", "active_dataset.csv")

def get_loaded_df():
    """
    Helper to load and clean the active dataset.
    """
    return get_clean_df()

# --- Core Analyst Tools ---

def list_products() -> dict:
    """
    Returns a catalog of all unique products available in the active dataset,
    grouped by category, with their IDs and names. Help LLM guide catalog queries.
    """
    try:
        df = get_loaded_df()
        unique_prods = df[['product_id', 'product_name', 'category']].drop_duplicates()
        
        catalog = {}
        for _, row in unique_prods.iterrows():
            cat = str(row['category'])
            if cat not in catalog:
                catalog[cat] = []
            catalog[cat].append({
                "product_id": str(row['product_id']),
                "product_name": str(row['product_name'])
            })
            
        return {
            "status": "success",
            "metric_type": "product_catalog",
            "catalog": catalog
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to list product catalog: {str(e)}"}

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

def _run_ml_forecast_with_overrides(model, history_subset, horizon_days=30, override_promo=False, override_season=False) -> float:
    """
    Helper to run recursive ML forecasting with specific features muted/neutralized
    to determine mathematical feature group contributions counterfactually.
    """
    sales_buffer = history_subset['units_sold'].tolist()
    last_historic_promo = 0 if override_promo else history_subset['promotion_flag'].iloc[-1]
    future_promos = [last_historic_promo] + [0] * horizon_days
    
    avg_price = float(history_subset['price'].mean())
    feat_row = pd.DataFrame(np.zeros((1, len(model.feature_cols))), columns=model.feature_cols)
    feat_row.at[0, 'price'] = avg_price
    
    predictions_sum = 0.0
    
    last_historic_date = history_subset['date'].max()
    future_dates = [last_historic_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
    
    for step, fut_date in enumerate(future_dates, start=1):
        if override_season:
            # Set to Wednesday in June (neutral weekday, no weekend impact)
            day_of_week = 2
            month = 6
            is_weekend = 0
            day_of_year = 180
        else:
            day_of_week = fut_date.weekday()
            month = fut_date.month
            is_weekend = 1 if day_of_week in [5, 6] else 0
            day_of_year = fut_date.timetuple().tm_yday
            
        lag_1 = sales_buffer[-1]
        lag_7 = sales_buffer[-7] if len(sales_buffer) >= 7 else sales_buffer[0]
        lag_14 = sales_buffer[-14] if len(sales_buffer) >= 14 else sales_buffer[0]
        roll_7 = np.mean(sales_buffer[-7:])
        roll_30 = np.mean(sales_buffer[-30:])
        promo_lag = 0 if override_promo else future_promos[step - 1]
        
        feat_row.at[0, 'promotion_flag'] = 0
        feat_row.at[0, 'day_of_week'] = day_of_week
        feat_row.at[0, 'month'] = month
        feat_row.at[0, 'is_weekend'] = is_weekend
        feat_row.at[0, 'day_of_year'] = day_of_year
        feat_row.at[0, 'units_sold_lag_1'] = lag_1
        feat_row.at[0, 'units_sold_lag_7'] = lag_7
        feat_row.at[0, 'units_sold_lag_14'] = lag_14
        feat_row.at[0, 'units_sold_roll_mean_7'] = roll_7
        feat_row.at[0, 'units_sold_roll_mean_30'] = roll_30
        feat_row.at[0, 'promo_lag_1'] = int(promo_lag)
        
        pred_res = model.predict(feat_row, step=step)
        pred_val = float(pred_res["predictions"][0])
        predictions_sum += pred_val
        sales_buffer.append(pred_val)
        
    return predictions_sum

def explain_forecast_decomposition(product_id: str) -> dict:
    """
    Computes a causal mathematical decomposition of the future forecast values
    to explain what portion is driven by:
      1. Baseline History/Trend
      2. Calendar Seasonality (Days of Week/Is Weekend)
      3. Promotion flag impacts (inc. promotional hangover lag effects)
    Calculated counterfactually in Python to prevent LLM hallucinations.
    """
    try:
        df = get_loaded_df()
        
        # Load best model
        best_meta = get_best_model_metadata(product_id)
        if not best_meta:
            return {"status": "error", "message": f"No models trained for product '{product_id}'."}
            
        model_name = best_meta["model_name"]
        
        # Get product aggregations
        df_product = aggregate_to_product_level(df)
        prod_df = df_product[df_product['product_id'] == product_id].copy()
        
        if model_name == "Prophet":
            # Prophet returns components natively
            model = load_model(product_id, model_name)
            last_historic_date = prod_df['date'].max()
            future_dates = [last_historic_date + timedelta(days=i) for i in range(1, 31)]
            avg_price = float(prod_df.sort_values(by='date').tail(30)['price'].mean())
            
            future_df = pd.DataFrame({"date": future_dates, "price": avg_price, "promotion_flag": 0})
            res_act = model.model.predict(future_df.rename(columns={'date': 'ds'}))
            
            trend_sum = float(res_act['trend'].sum())
            weekly_sum = float(res_act['weekly'].sum()) if 'weekly' in res_act else 0.0
            yearly_sum = float(res_act['yearly'].sum()) if 'yearly' in res_act else 0.0
            total_pred = float(res_act['yhat'].sum())
            
            weekly_pct = (weekly_sum / max(1, total_pred)) * 100
            yearly_pct = (yearly_sum / max(1, total_pred)) * 100
            
            promo_pct = 0.0 # No future promos by default
            season_pct = max(0.0, min(100.0, weekly_pct + yearly_pct))
            trend_pct = 100.0 - season_pct - promo_pct
            
            return {
                "status": "success",
                "metric_type": "forecast_decomposition",
                "product_id": product_id,
                "product_name": prod_df['product_name'].iloc[0],
                "model_used": model_name,
                "decomposition": {
                    "baseline_trend_percent": round(trend_pct, 1),
                    "weekly_seasonality_percent": round(season_pct, 1),
                    "promotion_impact_percent": round(promo_pct, 1)
                },
                "total_forecasted_units": int(round(total_pred)),
                "note": "Contributions represent an analytical counterfactual approximation of forecast drivers over the 30-day horizon, not an exact mathematical cause of individual sales day spikes."
            }
            
        else:
            # ML Model Counterfactuals
            model = load_model(product_id, model_name)
            history_subset = prod_df.sort_values(by='date').tail(30).copy()
            
            sum_act = _run_ml_forecast_with_overrides(model, history_subset, override_promo=False, override_season=False)
            if sum_act <= 0:
                return {
                    "status": "success",
                    "metric_type": "forecast_decomposition",
                    "product_id": product_id,
                    "product_name": prod_df['product_name'].iloc[0],
                    "model_used": model_name,
                    "decomposition": {
                        "baseline_trend_percent": 100.0,
                        "weekly_seasonality_percent": 0.0,
                        "promotion_impact_percent": 0.0
                    },
                    "total_forecasted_units": 0
                }
                
            sum_nopromo = _run_ml_forecast_with_overrides(model, history_subset, override_promo=True, override_season=False)
            sum_noseason = _run_ml_forecast_with_overrides(model, history_subset, override_promo=False, override_season=True)
            
            promo_impact = sum_act - sum_nopromo
            season_impact = sum_act - sum_noseason
            trend_impact = sum_act - promo_impact - season_impact
            
            promo_pct = (promo_impact / sum_act) * 100
            season_pct = (season_impact / sum_act) * 100
            trend_pct = (trend_impact / sum_act) * 100
            
            # Bound and clamp to prevent mathematical edge case skewing
            promo_pct = max(-50.0, min(100.0, promo_pct))
            season_pct = max(-50.0, min(100.0, season_pct))
            
            # Make sure it adds up to 100% exactly
            trend_pct = 100.0 - promo_pct - season_pct
            if trend_pct < 0:
                trend_pct = 0.0
                season_pct = 100.0 - promo_pct
                
            return {
                "status": "success",
                "metric_type": "forecast_decomposition",
                "product_id": product_id,
                "product_name": prod_df['product_name'].iloc[0],
                "model_used": model_name,
                "decomposition": {
                    "baseline_trend_percent": round(trend_pct, 1),
                    "weekly_seasonality_percent": round(season_pct, 1),
                    "promotion_impact_percent": round(promo_pct, 1)
                },
                "total_forecasted_units": int(round(sum_act)),
                "note": "Contributions represent an analytical counterfactual approximation of forecast drivers over the 30-day horizon, not an exact mathematical cause of individual sales day spikes."
            }
    except Exception as e:
        return {"status": "error", "message": f"Decomposition failed: {str(e)}"}

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
        mae = m["metrics"].get("MAE", 0.0)
        mape = m["metrics"].get("MAPE", 0.0)
        r2 = m["metrics"].get("R2", 0.0)
        
        comparison.append({
            "model_name": m["model_name"],
            "metrics": m["metrics"],
            "human_friendly_explanation": {
                "MAE": f"MAE of {mae} indicates that on average, this model's daily predictions deviate from actual sales by {mae} units.",
                "MAPE": f"MAPE of {mape}% represents the average relative percentage deviation of predictions from actual sales volume.",
                "R2": f"R² of {r2} indicates that this model explains {r2 * 100:.1f}% of the historical variance in demand (a negative value implies it performs worse than a simple historical mean baseline)."
            }
        })
        
    return {
        "status": "success",
        "metric_type": "model_comparison",
        "product_id": product_id,
        "recommended_model": best_meta["model_name"],
        "recommendation_reason": best_meta["recommendation_reason"],
        "all_models_metrics": comparison
    }

CHART_TYPES = {"bar", "line", "donut", "area"}
CHART_METRICS = {"units_sold", "revenue", "stock", "profit"}
CHART_DIMENSIONS = {"product", "category", "date", "day_of_week", "month"}
_MONTH_LABELS = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
_DAY_LABELS = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

def generate_chart_spec(
    chart_type: str = "bar",
    metric: str = "units_sold",
    dimension: str = "category",
    product_ids: list = None,
    categories: list = None,
    start_date: str = None,
    end_date: str = None,
    recent_days: int = None,
    limit: int = 10,
) -> dict:
    """
    Builds a small, chart-ready data spec for the chat's "generate a chart" feature.
    Every number here comes from real pandas aggregation over the active dataset — the
    LLM only ever selects WHICH chart to build (chart_type/metric/dimension/filters),
    never the values, mirroring how every other tool in this module prevents
    hallucination. Reuses the same filter-then-aggregate approach as
    core/forecasting/eda.py's generate_eda_report() so a chat-requested chart and the
    equivalent Sales Insights filtered view always agree with each other.

    product_ids is expected to already be resolved (see agent.py's resolve_product_id) —
    this function does no fuzzy name matching itself, consistent with how forecast_product/
    compare_sales/etc. also take a pre-resolved product_id rather than a raw name.

    recent_days is a simpler alternative to start_date for date-relative requests (e.g.
    "last 3 months" -> recent_days=90) — safer for an LLM to produce reliably than exact
    calendar dates, mirroring compare_sales()'s existing period_days pattern.
    """
    if chart_type not in CHART_TYPES:
        chart_type = "donut" if chart_type == "pie" else "bar"
    if metric not in CHART_METRICS:
        metric = "units_sold"
    if dimension not in CHART_DIMENSIONS:
        dimension = "category"
    limit = max(1, min(int(limit or 10), 25))

    df = get_loaded_df()
    if product_ids:
        df = df[df['product_id'].isin(product_ids)]
    if categories:
        df = df[df['category'].isin(categories)]
    if recent_days:
        cutoff = df['date'].max() - timedelta(days=int(recent_days))
        df = df[df['date'] >= cutoff]
    elif start_date or end_date:
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

    if len(df) == 0:
        return {"status": "error", "message": "No data matches the requested filters — try a broader date range or different products/categories."}

    df = df.copy()
    df['revenue'] = df['units_sold'] * df['price']
    if metric == "profit":
        # Prefer a real 'profit' column from the source dataset if present (respects
        # whatever calculation the user's own data used) over deriving one — only
        # falls back to revenue - cost when the dataset actually has cost data.
        if 'profit' in df.columns:
            pass  # already a real column, nothing to compute
        elif 'cost_price' in df.columns:
            df['profit'] = df['revenue'] - (df['units_sold'] * df['cost_price'])
        else:
            return {"status": "error", "message": "This dataset doesn't include cost or profit data, so a profit/loss chart isn't possible — try revenue instead."}
    value_col = {"revenue": "revenue", "stock": "stock_on_hand", "profit": "profit"}.get(metric, "units_sold")

    # A donut/pie reads as part-of-a-whole composition across a handful of categories —
    # it breaks down for a date series (e.g. 30 daily slices is unreadable), regardless
    # of literally what the user asked for ("pie chart of profit over the last 30 days").
    # Line reads correctly as a trend over time and is the closer match to their intent.
    if dimension == 'date' and chart_type == 'donut':
        chart_type = 'line'

    if dimension == 'product':
        grouped = df.groupby('product_name')[value_col].sum().sort_values(ascending=False).head(limit)
    elif dimension == 'date':
        grouped = df.groupby(df['date'].dt.strftime('%Y-%m-%d'))[value_col].sum().sort_index()
    elif dimension == 'day_of_week':
        grouped = df.groupby(df['date'].dt.dayofweek)[value_col].sum().sort_index()
        grouped.index = grouped.index.map(_DAY_LABELS)
    elif dimension == 'month':
        grouped = df.groupby(df['date'].dt.month)[value_col].sum().sort_index()
        grouped.index = grouped.index.map(_MONTH_LABELS)
    else:  # category
        grouped = df.groupby('category')[value_col].sum().sort_values(ascending=False)

    metric_label = {"revenue": "Revenue (₹)", "stock": "Stock on Hand", "profit": "Profit (₹)"}.get(metric, "Units Sold")
    title = f"{metric_label} by {dimension.replace('_', ' ').title()}"

    return {
        "status": "success",
        "metric_type": "chart_spec",
        "chart": {
            "chart_type": chart_type,
            "title": title,
            "x": [str(v) for v in grouped.index.tolist()],
            "y": [round(float(v), 2) for v in grouped.values],
            "series_label": metric_label
        }
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
