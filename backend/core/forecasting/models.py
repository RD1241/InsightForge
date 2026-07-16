import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Single source of truth for how far a What-If price scenario is allowed to move from the
# historical average — referenced by the API layer (routers/forecasting.py, validates the
# incoming price_multiplier), the recursive forecast loop (train_pipeline.py, clamps the
# multiplier before computing avg_price), and ProphetModel.predict()'s own regressor clip
# below. All three used to hardcode 0.7/1.3 independently; a change to one without the
# others silently reintroduces the "What-If slider does nothing" bug at a new threshold.
PRICE_SCENARIO_MULTIPLIER_MIN = 0.7
PRICE_SCENARIO_MULTIPLIER_MAX = 1.3

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

class GradientBoostingModel(BaseForecaster):
    """
    Histogram-based gradient boosting (scikit-learn's HistGradientBoostingRegressor —
    architecturally the same family as LightGBM/XGBoost, built into scikit-learn so no
    new dependency is required). Chosen over the previous Random Forest model because
    gradient-boosted trees are the empirically dominant approach for tabular retail demand
    forecasting: in the M5 forecasting competition (the largest public retail-forecasting
    benchmark, ~30,000 real product/store series), the top 50 submissions were all
    tree-ensemble/gradient-boosting based. It also predicts faster per call than the prior
    Random Forest, which matters here since the recursive validation loop calls .predict()
    once per test day.
    """
    def __init__(self):
        super().__init__()
        self.model_name = "Gradient Boosting"
        self.model = HistGradientBoostingRegressor(max_iter=200, random_state=42)
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
        # Raw (pre-standardization) training price bounds — Prophet z-scores
        # 'price' internally during fit(), so self.model.history['price'] is NOT
        # usable for this; these must be captured from the raw input separately.
        self.train_price_min = None
        self.train_price_max = None

    def fit(self, train_df: pd.DataFrame):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed or failed to load.")

        prophet_df = train_df[['date', 'units_sold', 'price', 'promotion_flag']].rename(
            columns={'date': 'ds', 'units_sold': 'y'}
        )

        self.train_price_min = float(prophet_df['price'].min())
        self.train_price_max = float(prophet_df['price'].max())

        # Only fit yearly seasonality if there's at least a full year of training
        # history. Forcing it on (the old hardcoded True) makes Prophet fit a yearly
        # cycle it has never actually observed on a short dataset — confirmed directly:
        # on an 89-day real user upload, one product's validation R2 went from -1312
        # (forced True) to -0.09 (disabled) from this switch alone.
        #
        # Prophet's own yearly_seasonality='auto' isn't a safe substitute here — it
        # requires >=730 days (2 full years) before enabling it, which is calibrated
        # for long production time series, not this app's realistic dataset sizes. The
        # demo dataset (699 days) has real, learnable yearly seasonality: forcing it on
        # scores R2=0.93; 'auto' disables it (span < 730) and drops to R2=0.48. A
        # self-computed 365-day (one full cycle observed) threshold gets both cases
        # right — verified directly against both the demo dataset and a real 89-day
        # user upload.
        train_span_days = (train_df['date'].max() - train_df['date'].min()).days
        use_yearly_seasonality = train_span_days >= 365

        self.model = Prophet(
            yearly_seasonality=use_yearly_seasonality,
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

        # Clip the price regressor to a margin around the range Prophet actually trained
        # on. Prophet's price coefficient is fit against whatever variance the training
        # window happened to contain — for products with a near-constant historical
        # price, that coefficient is poorly identified, and feeding a serve-time price far
        # outside that range causes the additive model to extrapolate the trend term
        # catastrophically (observed: correct-looking forecasts collapsing to a large
        # negative yhat, clipped to a flat 0 by the max(0, ...) below).
        #
        # The margin matters: clipping to the *exact* historical min/max (no slack) made
        # the What-If price slider a no-op for any product whose historical price barely
        # varied (common for fixed-MRP retail items) — every requested scenario price got
        # clamped straight back to the unchanged historical value, so the forecast never
        # moved no matter how far the user dragged the slider (confirmed directly: a +30%
        # scenario shifted a real product's prediction by <1%). Widening the clip to the
        # slider's own bound (PRICE_SCENARIO_MULTIPLIER_MIN/MAX, module-level above) lets
        # every legitimate scenario reach the model while still catching genuinely
        # out-of-bounds values.
        train_min = getattr(self, 'train_price_min', None)
        train_max = getattr(self, 'train_price_max', None)
        if train_min is not None and train_max is not None:
            prophet_df = prophet_df.copy()
            prophet_df['price'] = prophet_df['price'].clip(
                lower=train_min * PRICE_SCENARIO_MULTIPLIER_MIN,
                upper=train_max * PRICE_SCENARIO_MULTIPLIER_MAX
            )

        forecast = self.model.predict(prophet_df)

        preds = forecast['yhat'].values
        yhat_lower = forecast['yhat_lower'].values
        yhat_upper = forecast['yhat_upper'].values

        return {
            "predictions": np.round(np.maximum(0, preds), 2),
            "lower_bound": np.round(np.maximum(0, yhat_lower), 2),
            "upper_bound": np.round(np.maximum(0, yhat_upper), 2)
        }
