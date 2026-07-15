"""
Scale benchmark for InsightForge.

Measures the core pipeline stages (load, validate, clean, aggregate, smooth,
feature-engineer, EDA, model fit/validate) against synthetic datasets sized
at 10k / 50k / 100k / 200k rows, plus wide-column and registry-I/O stress
tests, to find real bottlenecks before optimizing anything.

Does NOT touch the real active_dataset.csv or models_store/ — everything
here runs against in-memory data or isolated temp files.

Run from the repo root:
    venv/Scripts/python.exe backend/scratch/benchmark_scale.py
"""
import os
import sys
import time
import json
import tempfile
import shutil

import numpy as np
import pandas as pd

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from core.forecasting.preprocessor import (
    validate_dataset, clean_dataset, aggregate_to_product_level,
    smooth_outliers, build_features
)
from core.forecasting.eda import generate_eda_report
from core.forecasting.models import (
    RidgeRegressionModel, GradientBoostingModel,
    ProphetModel, evaluate_predictions, PROPHET_AVAILABLE
)
from core.forecasting.train_pipeline import _recursive_validation_predict

try:
    import psutil
    PROCESS = psutil.Process(os.getpid())
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def rss_mb():
    if HAS_PSUTIL:
        return PROCESS.memory_info().rss / 1e6
    return None


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def make_dataset(n_products, n_stores, n_days, n_extra_cols=0, start="2022-01-01"):
    """Vectorized synthetic dataset generator (no per-row Python loops)."""
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    products = [f"PRD_{i:04d}" for i in range(n_products)]
    stores = [f"ST_{i:03d}" for i in range(n_stores)]
    categories = ["Dairy", "Bakery", "Produce", "Beverages", "Pantry", "Snacks", "Household"]

    idx = pd.MultiIndex.from_product([dates, stores, products], names=["date", "store_id", "product_id"])
    n = len(idx)
    df = idx.to_frame(index=False)

    rng = np.random.default_rng(42)
    prod_base_price = {p: round(1.5 + rng.random() * 10, 2) for p in products}
    prod_category = {p: categories[i % len(categories)] for i, p in enumerate(products)}

    df["product_name"] = df["product_id"].map(lambda p: f"Item {p}")
    df["category"] = df["product_id"].map(prod_category)
    df["price"] = df["product_id"].map(prod_base_price)
    df["units_sold"] = rng.poisson(lam=30, size=n).astype(int)
    df["stock_on_hand"] = rng.integers(0, 400, size=n)
    df["promotion_flag"] = rng.integers(0, 2, size=n, endpoint=False)

    if n_extra_cols > 0:
        extra = rng.random((n, n_extra_cols)).astype(np.float32)
        extra_df = pd.DataFrame(extra, columns=[f"extra_metric_{i}" for i in range(n_extra_cols)])
        df = pd.concat([df.reset_index(drop=True), extra_df], axis=1)

    return df


def bench_pipeline_stages(label, df_raw, tmpdir):
    print(f"\n=== {label}  (rows={len(df_raw):,}, cols={df_raw.shape[1]}) ===")
    mem0 = rss_mb()

    csv_path = os.path.join(tmpdir, "bench.csv")
    _, t_write = timed(df_raw.to_csv, csv_path, index=False)
    (df_loaded, t_read) = timed(pd.read_csv, csv_path)
    print(f"  CSV write:            {t_write:6.3f}s")
    print(f"  CSV read (load):      {t_read:6.3f}s")

    report, t_validate = timed(validate_dataset, df_loaded)
    print(f"  validate_dataset:     {t_validate:6.3f}s   (valid={report['is_valid']})")

    df_clean, t_clean = timed(clean_dataset, df_loaded)
    print(f"  clean_dataset:        {t_clean:6.3f}s   -> {len(df_clean):,} rows")

    df_product, t_agg = timed(aggregate_to_product_level, df_clean)
    print(f"  aggregate_to_product: {t_agg:6.3f}s   -> {len(df_product):,} rows, {df_product['product_id'].nunique()} products")

    df_smooth, t_smooth = timed(smooth_outliers, df_product)
    print(f"  smooth_outliers:      {t_smooth:6.3f}s")

    df_feat, t_feat = timed(build_features, df_smooth)
    print(f"  build_features:       {t_feat:6.3f}s   -> {len(df_feat):,} rows")

    eda, t_eda = timed(generate_eda_report, df_clean)
    print(f"  generate_eda_report:  {t_eda:6.3f}s")

    mem1 = rss_mb()
    if HAS_PSUTIL:
        print(f"  process RSS memory:   {mem0:7.1f} MB -> {mem1:7.1f} MB  (+{mem1 - mem0:.1f} MB)")
    total = t_write + t_read + t_validate + t_clean + t_agg + t_smooth + t_feat + t_eda
    print(f"  TOTAL (excl. training/predict): {total:6.3f}s")

    return {
        "label": label, "rows": len(df_raw), "cols": df_raw.shape[1],
        "t_read": t_read, "t_validate": t_validate, "t_clean": t_clean,
        "t_agg": t_agg, "t_smooth": t_smooth, "t_feat": t_feat, "t_eda": t_eda,
        "total": total, "df_feat": df_feat
    }


def bench_training(label, df_feat, n_products_to_train=5):
    products = df_feat["product_id"].unique()[:n_products_to_train]
    fit_times = {"Ridge Regression": [], "Gradient Boosting": [], "Prophet": []}

    for pid in products:
        prod_df = df_feat[df_feat["product_id"] == pid].copy()
        if len(prod_df) < 44:
            continue
        train_df = prod_df.iloc[:-30]
        test_df = prod_df.iloc[-30:]
        y_test = test_df["units_sold"].values

        for name, cls in [("Ridge Regression", RidgeRegressionModel),
                           ("Gradient Boosting", GradientBoostingModel)]:
            m = cls()
            t0 = time.perf_counter()
            m.fit(train_df)
            _ = _recursive_validation_predict(m, train_df, test_df)
            fit_times[name].append(time.perf_counter() - t0)

        if PROPHET_AVAILABLE:
            m = ProphetModel()
            t0 = time.perf_counter()
            m.fit(train_df)
            _ = m.predict(test_df)
            fit_times["Prophet"].append(time.perf_counter() - t0)

    print(f"\n  -- Per-product fit+validate time ({label}, avg over {n_products_to_train} products, "
          f"history={df_feat.groupby('product_id').size().max()} days/product) --")
    per_product_total = 0.0
    for name, times in fit_times.items():
        if times:
            avg = sum(times) / len(times)
            per_product_total += avg
            print(f"    {name:<20}: {avg:6.3f}s avg")
    n_products_total = df_feat["product_id"].nunique()
    est_wall_4workers = (per_product_total * n_products_total) / 4
    print(f"  Extrapolated full training wall time for all {n_products_total} products "
          f"@ 4 parallel workers: ~{est_wall_4workers:6.1f}s (~{est_wall_4workers/60:.1f} min)")


def bench_registry_io_growth(tmpdir, checkpoints=(10, 50, 100, 200, 300)):
    """
    Isolated simulation of registry.py's save_model() I/O pattern:
    read the whole JSON, add one entry, write the whole JSON back.
    Runs against a throwaway file — never touches the real registry.
    """
    print("\n=== Registry I/O growth (isolated temp file, mirrors save_model()'s read-modify-write) ===")
    reg_path = os.path.join(tmpdir, "fake_registry.json")
    registry = {}
    with open(reg_path, "w") as f:
        json.dump(registry, f)

    cumulative = 0.0
    next_checkpoint_idx = 0
    for i in range(1, max(checkpoints) + 1):
        t0 = time.perf_counter()
        with open(reg_path, "r") as f:
            registry = json.load(f)
        pid = f"PRD_{i:04d}"
        registry[pid] = {
            "Ridge Regression": {"model_name": "Ridge Regression", "product_id": pid,
                                   "metrics": {"MAE": 5.0, "RMSE": 6.0, "MAPE": 8.0, "R2": 0.8},
                                   "model_path": f"{pid}_ridge_regression.pkl", "trained_at": "2026-01-01 00:00:00"}
        }
        with open(reg_path, "w") as f:
            json.dump(registry, f, indent=4)
        cumulative += time.perf_counter() - t0

        if next_checkpoint_idx < len(checkpoints) and i == checkpoints[next_checkpoint_idx]:
            size_kb = os.path.getsize(reg_path) / 1024
            print(f"  after {i:4d} saved models: file={size_kb:7.1f} KB   "
                  f"cumulative save time={cumulative:6.3f}s   (this save: {cumulative:.3f}s total so far)")
            next_checkpoint_idx += 1


def main():
    tmpdir = tempfile.mkdtemp(prefix="insightforge_bench_")
    print(f"Scratch dir: {tmpdir}")
    print(f"psutil available: {HAS_PSUTIL}  |  Prophet available: {PROPHET_AVAILABLE}")

    # (n_products, n_stores, n_days) chosen so products AND history both grow with scale,
    # matching the "hundreds of products, multi-year history" target.
    scales = [
        ("~10k rows",  20,  2, 250),
        ("~50k rows",  50,  2, 500),
        ("~100k rows", 100, 2, 500),
        ("~200k rows", 150, 2, 667),
    ]

    results = []
    for label, n_prod, n_store, n_days in scales:
        df_raw = make_dataset(n_prod, n_store, n_days, n_extra_cols=0)
        res = bench_pipeline_stages(label, df_raw, tmpdir)
        bench_training(label, res["df_feat"], n_products_to_train=5)
        results.append(res)

    # Wide-column stress test at a fixed, moderate row count
    print("\n\n########## WIDE-COLUMN STRESS TEST ##########")
    df_wide = make_dataset(30, 2, 500, n_extra_cols=180)
    bench_pipeline_stages("~30k rows x 189 columns", df_wide, tmpdir)

    bench_registry_io_growth(tmpdir)

    print("\n\n========== SUMMARY (pipeline stages only, seconds) ==========")
    header = f"{'Scale':<12}{'rows':>10}{'cols':>6}{'read':>8}{'validate':>10}{'clean':>8}{'agg':>8}{'smooth':>9}{'features':>10}{'eda':>8}{'TOTAL':>9}"
    print(header)
    for r in results:
        print(f"{r['label']:<12}{r['rows']:>10,}{r['cols']:>6}{r['t_read']:>8.3f}{r['t_validate']:>10.3f}"
              f"{r['t_clean']:>8.3f}{r['t_agg']:>8.3f}{r['t_smooth']:>9.3f}{r['t_feat']:>10.3f}{r['t_eda']:>8.3f}{r['total']:>9.3f}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
