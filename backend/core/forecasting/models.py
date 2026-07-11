import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Graceful import of Prophet to handle compile/install failures on Windows
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    import logging
    # Suppress cmdstanpy verbose logs
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    PROPHET_AVAILABLE = True
except ImportError:
    pass

def train_test_split_chronological(df: pd.DataFrame, test_days: int = 30):
    """
    Splits a single product's features dataframe chronologically.
    The last test_days are used as the test set.
    """
    df_sorted = df.sort_values(by='date').reset_index(drop=True)
    cutoff_date = df_sorted['date'].max() - pd.Timedelta(days=test_days)
    
    train_df = df_sorted[df_sorted['date'] <= cutoff_date].reset_index(drop=True)
    test_df = df_sorted[df_sorted['date'] > cutoff_date].reset_index(drop=True)
    
    return train_df, test_df

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error.
    Handles division by zero by replacing 0s with 1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Mask to prevent division by zero
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(y_true, y_pred):
    """
    Computes standard regression metrics.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(calculate_mape(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 2)
    }

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_cols = [
            'price', 'promotion_flag', 'day_of_week', 'month', 'is_weekend', 'day_of_year',
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
            'units_sold_roll_mean_7', 'units_sold_roll_mean_30'
        ]
        
    def fit(self, train_df: pd.DataFrame):
        X = train_df[self.feature_cols]
        y = train_df['units_sold']
        self.model.fit(X, y)
        
    def predict(self, df: pd.DataFrame):
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        return np.maximum(0, preds)  # Demand cannot be negative

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.feature_cols = [
            'price', 'promotion_flag', 'day_of_week', 'month', 'is_weekend', 'day_of_year',
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
            'units_sold_roll_mean_7', 'units_sold_roll_mean_30'
        ]
        
    def fit(self, train_df: pd.DataFrame):
        X = train_df[self.feature_cols]
        y = train_df['units_sold']
        self.model.fit(X, y)
        
    def predict(self, df: pd.DataFrame):
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        return np.maximum(0, preds)

class ProphetModel:
    def __init__(self):
        self.model = None
        
    def fit(self, train_df: pd.DataFrame):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed or failed to load. Use fallback model.")
            
        # Prepare Prophet format dataframe
        prophet_df = train_df[['date', 'units_sold', 'price', 'promotion_flag']].rename(
            columns={'date': 'ds', 'units_sold': 'y'}
        )
        
        # Instantiate Prophet with standard weekly and yearly seasonality
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        # Add price and promotion as additional regressors
        self.model.add_regressor('price')
        self.model.add_regressor('promotion_flag')
        
        self.model.fit(prophet_df)
        
    def predict(self, df: pd.DataFrame):
        if not self.model:
            raise ValueError("Prophet model must be fitted before predicting.")
            
        prophet_df = df[['date', 'price', 'promotion_flag']].rename(columns={'date': 'ds'})
        forecast = self.model.predict(prophet_df)
        
        # Return yhat and uncertainty intervals
        preds = forecast['yhat'].values
        yhat_lower = forecast['yhat_lower'].values
        yhat_upper = forecast['yhat_upper'].values
        
        return {
            "predictions": np.maximum(0, preds),
            "lower_bound": np.maximum(0, yhat_lower),
            "upper_bound": np.maximum(0, yhat_upper)
        }
