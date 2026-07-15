import pandas as pd


def analyze_pretrain_characteristics(df_clean: pd.DataFrame) -> dict:
    """
    Cheap, pure descriptive-stats analysis of the active dataset — no model fitting —
    run BEFORE training to give the user a plain-language "AI consultant" preview of
    which model is likely to perform best.

    This is explicitly a preview, not a promise: the authoritative answer is always
    whichever model actually wins on real chronological validation once Train is
    clicked (see train_pipeline.py's _train_single_product). Framing it that way here
    means this heuristic and the post-training result can never contradict each other
    in a way that undermines trust in the "AI recommends, you decide" experience.
    """
    daily = df_clean.groupby('date')['units_sold'].sum().reset_index()
    daily['day_of_week'] = daily['date'].dt.dayofweek
    weekly_avg = daily.groupby('day_of_week')['units_sold'].mean()
    weekly_mean = float(weekly_avg.mean())
    weekly_swing_pct = round(((weekly_avg.max() - weekly_avg.min()) / weekly_mean) * 100, 1) if weekly_mean > 0 else 0.0

    # Price variability, averaged across products — a direct callback to the lesson
    # learned fixing the Prophet price-regressor bug (see models.py): a model can only
    # learn a meaningful price effect if price actually varies during training.
    price_cv_per_product = df_clean.groupby('product_id')['price'].agg(
        lambda s: (s.std() / s.mean() * 100) if s.mean() > 0 else 0.0
    )
    avg_price_cv = round(float(price_cv_per_product.mean()), 1) if len(price_cv_per_product) else 0.0

    # Trend stability: most recent 90 days vs. the 90 days before that.
    max_date = daily['date'].max()
    recent_mask = daily['date'] > max_date - pd.Timedelta(days=90)
    prior_mask = (daily['date'] <= max_date - pd.Timedelta(days=90)) & (daily['date'] > max_date - pd.Timedelta(days=180))
    recent_mean = float(daily.loc[recent_mask, 'units_sold'].mean()) if recent_mask.any() else 0.0
    prior_mean = float(daily.loc[prior_mask, 'units_sold'].mean()) if prior_mask.any() else 0.0
    trend_shift_pct = round(((recent_mean - prior_mean) / prior_mean) * 100, 1) if prior_mean > 0 else 0.0

    if weekly_swing_pct >= 25:
        suggested_model = "Prophet"
        reason = (
            f"We detected strong weekly seasonality — demand swings {weekly_swing_pct:.0f}% "
            f"between the slowest and busiest day of the week. Prophet is built to decompose "
            f"exactly this kind of recurring pattern, so it's likely to come out on top."
        )
    elif abs(trend_shift_pct) >= 20:
        suggested_model = "Gradient Boosting"
        reason = (
            f"Demand has shifted {abs(trend_shift_pct):.0f}% between the last 90 days and the "
            f"90 days before that. Gradient Boosting's flexibility tends to adapt faster to a "
            f"changing trend than a fixed seasonal or linear model."
        )
    else:
        suggested_model = "Ridge Regression"
        reason = (
            "Demand looks relatively stable with no strong weekly pattern or recent shift, so "
            "a simple, transparent linear model is likely to perform close to the more complex options."
        )

    if avg_price_cv < 5:
        price_note = "Prices have barely moved historically, so price-based what-if scenarios may have limited effect regardless of which model is used."
    else:
        price_note = f"Prices have varied meaningfully (~{avg_price_cv:.0f}% average variation), so price-based what-if scenarios should show a real effect."

    return {
        "weekly_seasonality_swing_pct": weekly_swing_pct,
        "avg_price_variation_pct": avg_price_cv,
        "recent_trend_shift_pct": trend_shift_pct,
        "suggested_model": suggested_model,
        "reason": reason,
        "price_note": price_note
    }
