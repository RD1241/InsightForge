import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Graceful import of Prophet
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    PROPHET_AVAILABLE = True
except ImportError:
    pass

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles zero values by dividing only non-zero actual sales.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(y_true, y_pred):
    """
    Evaluates predictions and returns standard metrics.
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

class BaseForecaster:
    """
    Unified base interface for all forecasting models.
    """
    def __init__(self):
        self.model_name = "Base"
        self.residual_std = 0.0
        
    def fit(self, train_df: pd.DataFrame):
        pass
        
    def predict(self, df: pd.DataFrame, step: int = 1) -> dict:
        """
        Returns a dictionary:
        {
            "predictions": np.array,
            "lower_bound": np.array,
            "upper_bound": np.array
        }
        """
        pass

class LinearRegressionModel(BaseForecaster):
    def __init__(self):
        super().__init__()
        self.model_name = "Linear Regression"
        self.model = LinearRegression()
        self.feature_cols = [
            'price', 'promotion_flag', 'day_of_week', 'month', 'is_weekend', 'day_of_year',
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
            'units_sold_roll_mean_7', 'units_sold_roll_mean_30', 'promo_lag_1'
        ]
        
    def fit(self, train_df: pd.DataFrame):
        X = train_df[self.feature_cols]
        y = train_df['units_sold']
        self.model.fit(X, y)
        
        # Calculate standard deviation of residuals for confidence intervals
        preds_train = self.model.predict(X)
        residuals = y - preds_train
        self.residual_std = float(np.std(residuals))
        
    def predict(self, df: pd.DataFrame, step: int = 1) -> dict:
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        preds = np.maximum(0, preds)
        
        # Calculate step-dependent confidence interval: uncertainty grows over time
        step_factor = np.sqrt(step)
        lower_bound = np.maximum(0, preds - 1.96 * self.residual_std * step_factor)
        upper_bound = preds + 1.96 * self.residual_std * step_factor
        
        return {
            "predictions": np.round(preds, 2),
            "lower_bound": np.round(lower_bound, 2),
            "upper_bound": np.round(upper_bound, 2)
        }

class RidgeRegressionModel(BaseForecaster):
    def __init__(self):
        super().__init__()
        self.model_name = "Ridge Regression"
        # Standardizing features is essential for L2 regularization
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0, random_state=42))
        ])
        self.feature_cols = [
            'price', 'promotion_flag', 'day_of_week', 'month', 'is_weekend', 'day_of_year',
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
            'units_sold_roll_mean_7', 'units_sold_roll_mean_30', 'promo_lag_1'
        ]
        
    def fit(self, train_df: pd.DataFrame):
        X = train_df[self.feature_cols]
        y = train_df['units_sold']
        self.model.fit(X, y)
        
        # Calculate standard deviation of residuals for confidence intervals
        preds_train = self.model.predict(X)
        residuals = y - preds_train
        self.residual_std = float(np.std(residuals))
        
    def predict(self, df: pd.DataFrame, step: int = 1) -> dict:
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        preds = np.maximum(0, preds)
        
        # Calculate step-dependent confidence interval: uncertainty grows over time
        step_factor = np.sqrt(step)
        lower_bound = np.maximum(0, preds - 1.96 * self.residual_std * step_factor)
        upper_bound = preds + 1.96 * self.residual_std * step_factor
        
        return {
            "predictions": np.round(preds, 2),
            "lower_bound": np.round(lower_bound, 2),
            "upper_bound": np.round(upper_bound, 2)
        }

class RandomForestModel(BaseForecaster):
    def __init__(self):
        super().__init__()
        self.model_name = "Random Forest"
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.feature_cols = [
            'price', 'promotion_flag', 'day_of_week', 'month', 'is_weekend', 'day_of_year',
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_14',
            'units_sold_roll_mean_7', 'units_sold_roll_mean_30', 'promo_lag_1'
        ]
        
    def fit(self, train_df: pd.DataFrame):
        X = train_df[self.feature_cols]
        y = train_df['units_sold']
        self.model.fit(X, y)
        
        # Calculate standard deviation of residuals
        preds_train = self.model.predict(X)
        residuals = y - preds_train
        self.residual_std = float(np.std(residuals))
        
    def predict(self, df: pd.DataFrame, step: int = 1) -> dict:
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        preds = np.maximum(0, preds)
        
        # Calculate step-dependent confidence interval: uncertainty grows over time
        step_factor = np.sqrt(step)
        lower_bound = np.maximum(0, preds - 1.96 * self.residual_std * step_factor)
        upper_bound = preds + 1.96 * self.residual_std * step_factor
        
        return {
            "predictions": np.round(preds, 2),
            "lower_bound": np.round(lower_bound, 2),
            "upper_bound": np.round(upper_bound, 2)
        }

class ProphetModel(BaseForecaster):
    def __init__(self):
        super().__init__()
        self.model_name = "Prophet"
        self.model = None
        
    def fit(self, train_df: pd.DataFrame):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed or failed to load.")
            
        prophet_df = train_df[['date', 'units_sold', 'price', 'promotion_flag']].rename(
            columns={'date': 'ds', 'units_sold': 'y'}
        )
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95  # 95% confidence interval
        )
        self.model.add_regressor('price')
        self.model.add_regressor('promotion_flag')
        
        self.model.fit(prophet_df)
        
    def predict(self, df: pd.DataFrame, step: int = 1) -> dict:
        if not self.model:
            raise ValueError("Prophet model must be fitted before predicting.")
            
        prophet_df = df[['date', 'price', 'promotion_flag']].rename(columns={'date': 'ds'})
        forecast = self.model.predict(prophet_df)
        
        preds = forecast['yhat'].values
        yhat_lower = forecast['yhat_lower'].values
        yhat_upper = forecast['yhat_upper'].values
        
        return {
            "predictions": np.round(np.maximum(0, preds), 2),
            "lower_bound": np.round(np.maximum(0, yhat_lower), 2),
            "upper_bound": np.round(np.maximum(0, yhat_upper), 2)
        }
