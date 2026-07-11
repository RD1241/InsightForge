import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(output_path: str = None, num_years: int = 2, start_date_str: str = "2024-01-01"):
    """
    Generates a realistic synthetic retail demand dataset with seasonal patterns,
    weekly effects, promotions, and realistic inventory stock levels.
    """
    np.random.seed(42)
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=365 * num_years - 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Store definition
    stores = [
        {"store_id": "ST_001", "store_name": "Downtown Supermarket"},
        {"store_id": "ST_002", "store_name": "Suburban Express"}
    ]
    
    # Product definition
    products = [
        {"product_id": "PRD_01", "product_name": "Organic Milk 1L", "category": "Dairy", "base_price": 3.50, "base_demand": 60, "trend": 0.005},
        {"product_id": "PRD_02", "product_name": "Whole Wheat Bread", "category": "Bakery", "base_price": 2.80, "base_demand": 45, "trend": 0.002},
        {"product_id": "PRD_03", "product_name": "Free Range Eggs 12pk", "category": "Dairy", "base_price": 4.99, "base_demand": 30, "trend": 0.008},
        {"product_id": "PRD_04", "product_name": "Fresh Fuji Apples 1kg", "category": "Produce", "base_price": 3.99, "base_demand": 25, "trend": -0.001},
        {"product_id": "PRD_05", "product_name": "Herbal Shampoo 400ml", "category": "Personal Care", "base_price": 8.50, "base_demand": 8, "trend": 0.001}
    ]
    
    records = []
    
    for store in stores:
        store_id = store["store_id"]
        
        for prod in products:
            prod_id = prod["product_id"]
            prod_name = prod["product_name"]
            cat = prod["category"]
            base_price = prod["base_price"]
            base_demand = prod["base_demand"]
            trend_val = prod["trend"]
            
            # Setup initial stock for this product in this store
            stock_on_hand = int(base_demand * 5)
            replenishment_level = int(base_demand * 1.5)
            replenishment_qty = int(base_demand * 5)
            
            # Pre-schedule some promotions for this product/store (approx 3-week durations, 4 times a year)
            promo_weeks = np.random.choice(range(1, 52 * num_years), size=4 * num_years, replace=False)
            promo_days = set()
            for w in promo_weeks:
                start_day = (w - 1) * 7
                for d in range(7):
                    promo_days.add(start_day + d)
            
            for t_idx, current_date in enumerate(date_range):
                day_of_week = current_date.weekday()
                day_of_year = current_date.timetuple().tm_yday
                
                # 1. Base demand with linear trend
                demand = base_demand + (t_idx * trend_val)
                
                # 2. Weekly seasonality (spikes on Friday, Saturday, Sunday)
                # 0: Monday, 1: Tuesday, 2: Wednesday, 3: Thursday, 4: Friday, 5: Saturday, 6: Sunday
                if day_of_week in [4, 5]:
                    weekly_factor = 1.35  # Friday-Saturday spike
                elif day_of_week == 6:
                    weekly_factor = 1.15  # Sunday medium sales
                elif day_of_week in [0, 1]:
                    weekly_factor = 0.80  # Monday-Tuesday slow
                else:
                    weekly_factor = 0.95  # Wednesday-Thursday normal
                
                demand *= weekly_factor
                
                # 3. Yearly/Seasonal effects
                # Apples peak in Autumn (around Oct/Nov), Milk/Shampoo slightly higher in Summer, etc.
                if prod_id == "PRD_04": # Apples
                    yearly_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 270) / 365)
                elif prod_id == "PRD_05": # Shampoo
                    yearly_factor = 1.0 + 0.1 * np.cos(2 * np.pi * (day_of_year - 180) / 365)
                else: # Generic winter/holiday spike for Dairy/Bakery
                    yearly_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 350) / 365)
                
                demand *= yearly_factor
                
                # 4. Promotions (Price drops and demand surges)
                is_promo = t_idx in promo_days
                price = base_price
                if is_promo:
                    price_discount = 0.20  # 20% discount
                    price = round(base_price * (1 - price_discount), 2)
                    # Demand surge during promotion
                    demand *= 1.8
                
                # 5. Random Noise
                noise = np.random.normal(loc=0.0, scale=base_demand * 0.12)
                demand += noise
                
                # Ensure demand is non-negative
                units_demanded = max(0, int(round(demand)))
                
                # 6. Inventory depletion & stockout capping
                if stock_on_hand >= units_demanded:
                    units_sold = units_demanded
                    stock_on_hand -= units_sold
                else:
                    units_sold = stock_on_hand
                    stock_on_hand = 0  # Stockout!
                
                # 7. Inventory replenishment logic
                # If stock on hand falls below replenishment level, schedule a delivery in 1-2 days
                # For simplicity, we replenish it instantly at the end of the day if it falls below threshold
                if stock_on_hand <= replenishment_level:
                    stock_on_hand += replenishment_qty
                
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "store_id": store_id,
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "category": cat,
                    "units_sold": units_sold,
                    "price": price,
                    "stock_on_hand": stock_on_hand,
                    "promotion_flag": 1 if is_promo else 0
                })
                
    df = pd.DataFrame(records)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Synthetic dataset saved to: {output_path} (Shape: {df.shape})")
        
    return df

if __name__ == "__main__":
    # Create default data directory and generate synthetic dataset
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    output_file = os.path.join(data_dir, "synthetic_retail_data.csv")
    generate_synthetic_data(output_file)
