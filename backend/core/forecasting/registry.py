import os
import json
import pickle
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_STORE_DIR = os.path.join(BASE_DIR, "models_store")
REGISTRY_JSON_PATH = os.path.join(MODELS_STORE_DIR, "models_registry.json")
TRAINING_REPORT_PATH = os.path.join(MODELS_STORE_DIR, "training_report.json")

# Current supported model lineup (see models.py). save_registry_batch() only ever merges
# model entries into the registry, never removes them, so switching the lineup (as
# happened when Random Forest / Linear Regression were replaced by Ridge / Gradient
# Boosting) leaves stale entries behind indefinitely. get_product_models() filters on
# this whitelist so every reader — best-model selection, /api/forecast/compare, and the
# chat agent's model_comparison tool — is protected from ever selecting or surfacing a
# model class that no longer exists in models.py.
SUPPORTED_MODELS = {"Ridge Regression", "Gradient Boosting", "Prophet"}

# Ensure models store exists
os.makedirs(MODELS_STORE_DIR, exist_ok=True)

# Threading lock to prevent race conditions during concurrent training runs
_registry_lock = threading.Lock()

def get_recommendation_reason(model_name: str, metrics: dict) -> str:
    """
    Returns a plain-language, business-facing explanation for why a specific model is
    recommended. Deliberately omits raw metric values (MAE/R²/MAPE) — those are for
    technical reference and stay in the Advanced Technical Details table; a retail
    manager doesn't need "MAE = 6.71" to trust a recommendation, they need to know it
    was the most accurate option tested and roughly how much to trust it.
    """
    r2 = metrics.get("R2", 0.0)
    if r2 >= 0.5:
        confidence_note = "and its predictions closely track your actual historical sales"
    elif r2 >= 0:
        confidence_note = "though with a moderate margin of error given the data available — treat it as a solid starting estimate"
    else:
        confidence_note = "though with limited historical data to learn from, treat this as a rough estimate rather than a precise prediction"

    if model_name == "Prophet":
        return (
            f"Prophet is recommended because it was the most accurate option on your historical data, "
            f"{confidence_note}. It's especially good at picking up on repeating weekly patterns and "
            f"adjusting for price changes and promotions."
        )
    elif model_name == "Gradient Boosting":
        return (
            f"Gradient Boosting is recommended because it was the most accurate option on your historical data, "
            f"{confidence_note}. It's well-suited to capturing complex, non-linear relationships between "
            f"price, promotions, and past sales trends."
        )
    elif model_name == "Ridge Regression":
        return (
            f"Ridge Regression is recommended because it was the most accurate option on your historical data, "
            f"{confidence_note}. It's a simple, stable model that fits general trends without overreacting to "
            f"noisy day-to-day swings."
        )
    return "Recommended because it was the most accurate option on your historical data."

def save_model_file(product_id: str, model_name: str, model_obj) -> tuple:
    """
    Serializes a trained model to a pickle file in models_store/ only — does not
    touch the registry JSON. Returns (filename, file_path).
    """
    safe_model_name = model_name.replace(" ", "_").lower()
    filename = f"{product_id}_{safe_model_name}.pkl"
    file_path = os.path.join(MODELS_STORE_DIR, filename)

    with open(file_path, "wb") as f:
        pickle.dump(model_obj, f)

    return filename, file_path


def save_registry_batch(entries_by_product: dict):
    """
    Updates the registry index for multiple products' models in a single
    read-modify-write cycle. `entries_by_product` maps
    product_id -> {model_name: {"metrics": dict, "filename": str}}.

    This is the one place that actually reads/writes REGISTRY_JSON_PATH — save_model()
    and save_models_batch() both funnel through it. A full training run calls this once
    for every product's models combined, instead of rewriting the (growing) registry file
    once per model: at hundreds of products x 4 models, that's hundreds of full-file
    read-modify-write cycles avoided.
    """
    with _registry_lock:
        registry = {}
        if os.path.exists(REGISTRY_JSON_PATH):
            try:
                with open(REGISTRY_JSON_PATH, "r") as f:
                    registry = json.load(f)
            except Exception:
                registry = {}

        trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for product_id, entries in entries_by_product.items():
            if product_id not in registry:
                registry[product_id] = {}
            for model_name, info in entries.items():
                registry[product_id][model_name] = {
                    "model_name": model_name,
                    "product_id": product_id,
                    "metrics": info["metrics"],
                    "model_path": info["filename"],  # Relative filename only to support workspace portability
                    "trained_at": trained_at
                }

        with open(REGISTRY_JSON_PATH, "w") as f:
            json.dump(registry, f, indent=4)


def save_models_batch(product_id: str, entries: dict):
    """
    Updates the registry index for one or more models of the same product in a
    single read-modify-write cycle. `entries` maps model_name -> {"metrics": dict, "filename": str}.
    """
    save_registry_batch({product_id: entries})


def save_model(product_id: str, model_name: str, model_obj, metrics: dict):
    """
    Serializes a trained model to the models_store/ directory and updates the registry index.
    Kept for callers saving a single model in isolation. When saving several models for the
    same product (e.g. during training), use save_model_file() + save_models_batch() instead
    to avoid one full registry read-modify-write per model.
    """
    filename, file_path = save_model_file(product_id, model_name, model_obj)
    save_models_batch(product_id, {model_name: {"metrics": metrics, "filename": filename}})
    return file_path

_model_cache = {}

def clear_model_cache():
    """
    Clears the in-memory models cache.
    """
    global _model_cache
    _model_cache.clear()

def load_model(product_id: str, model_name: str):
    """
    Loads a serialized model from the models_store directory.
    Uses in-memory caching to avoid redundant disk reads and deserialization.
    """
    global _model_cache
    cache_key = (product_id, model_name)
    if cache_key in _model_cache:
        return _model_cache[cache_key]
        
    with _registry_lock:
        if not os.path.exists(REGISTRY_JSON_PATH):
            raise FileNotFoundError("Model registry not found. Train models first.")
            
        with open(REGISTRY_JSON_PATH, "r") as f:
            registry = json.load(f)
            
    if product_id not in registry or model_name not in registry[product_id]:
        raise ValueError(f"Model '{model_name}' not found for product '{product_id}'.")
        
    stored_path = registry[product_id][model_name]["model_path"]
    if os.path.isabs(stored_path):
        file_path = stored_path
    else:
        file_path = os.path.join(MODELS_STORE_DIR, stored_path)
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model binary file not found at: {file_path}")
        
    with open(file_path, "rb") as f:
        model_obj = pickle.load(f)
        
    _model_cache[cache_key] = model_obj
    return model_obj

def get_product_models(product_id: str) -> list:
    """
    Returns all trained models and their metrics for a given product.
    """
    if not os.path.exists(REGISTRY_JSON_PATH):
        return []
        
    try:
        with _registry_lock:
            with open(REGISTRY_JSON_PATH, "r") as f:
                registry = json.load(f)
        if product_id in registry:
            return [
                m for m in registry[product_id].values()
                if m.get("model_name") in SUPPORTED_MODELS
            ]
    except Exception as e:
        import logging
        logger = logging.getLogger("insightforge")
        logger.error(f"Error reading model registry for product {product_id}: {str(e)}", exc_info=True)
    return []

def get_best_model_metadata(product_id: str) -> dict:
    """
    Determines the best performing model for a product based on the lowest MAE.
    Returns the metadata dictionary.
    """
    models = get_product_models(product_id)
    if not models:
        return None
        
    # Find model with lowest MAE
    # metrics = {"MAE": float, ...}
    best_model = min(models, key=lambda x: x["metrics"].get("MAE", float('inf')))
    
    # Inject recommendation explanation
    best_model["recommendation_reason"] = get_recommendation_reason(
        best_model["model_name"], best_model["metrics"]
    )
    
    return best_model

_training_report_lock = threading.Lock()

def save_training_report(report_data: dict):
    """
    Persists a comprehensive training report for the AI Analyst to query.
    Locked for the same reason as the model registry writes in save_registry_batch():
    two overlapping /train calls (two tabs, a retry) could otherwise interleave their
    writes to this file and corrupt or silently clobber it.
    """
    with _training_report_lock:
        with open(TRAINING_REPORT_PATH, "w") as f:
            json.dump(report_data, f, indent=4)

def load_training_report() -> dict:
    """
    Loads the persistent training report summary.
    """
    if not os.path.exists(TRAINING_REPORT_PATH):
        return None
    try:
        with _training_report_lock:
            with open(TRAINING_REPORT_PATH, "r") as f:
                return json.load(f)
    except Exception as e:
        import logging
        logger = logging.getLogger("insightforge")
        logger.error(f"Failed to load training report: {str(e)}", exc_info=True)
        return None
