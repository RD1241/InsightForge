import os
import json
import pickle
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_STORE_DIR = os.path.join(BASE_DIR, "models_store")
REGISTRY_JSON_PATH = os.path.join(MODELS_STORE_DIR, "models_registry.json")
TRAINING_REPORT_PATH = os.path.join(MODELS_STORE_DIR, "training_report.json")

# Ensure models store exists
os.makedirs(MODELS_STORE_DIR, exist_ok=True)

# Threading lock to prevent race conditions during concurrent training runs
_registry_lock = threading.Lock()

def get_recommendation_reason(model_name: str, metrics: dict) -> str:
    """
    Returns an evidence-based explanation for why a specific model is recommended.
    """
    mae = metrics.get("MAE", 0.0)
    r2 = metrics.get("R2", 0.0)
    
    if model_name == "Prophet":
        return (
            f"Prophet is recommended because it achieved the lowest Mean Absolute Error (MAE = {mae}) "
            f"on the chronological validation set. Prophet is highly transparent and excels at decomposing "
            f"yearly and weekly seasonal patterns while adjusting for price and promotional surges."
        )
    elif model_name == "Random Forest":
        return (
            f"Random Forest is recommended due to its superior validation metrics (MAE = {mae}, R² = {r2}). "
            f"The ensemble of decision trees successfully captured complex, non-linear relationships and "
            f"interactions between prices, promotion flags, and historical lag trends."
        )
    elif model_name == "Linear Regression":
        return (
            f"Linear Regression is recommended as it offered the lowest validation error (MAE = {mae}) "
            f"during testing. It provides a simple, robust baseline that fits linear trends effectively "
            f"without risk of overfitting on noisy retail sales data."
        )
    return f"Recommended based on lowest MAE on the validation set."

def save_model(product_id: str, model_name: str, model_obj, metrics: dict):
    """
    Serializes a trained model to the models_store/ directory and updates the registry index.
    """
    # 1. Save pickle file
    safe_model_name = model_name.replace(" ", "_").lower()
    filename = f"{product_id}_{safe_model_name}.pkl"
    file_path = os.path.join(MODELS_STORE_DIR, filename)
    
    with open(file_path, "wb") as f:
        pickle.dump(model_obj, f)
        
    # 2. Update models_registry.json under thread lock to prevent concurrent write corruption
    with _registry_lock:
        registry = {}
        if os.path.exists(REGISTRY_JSON_PATH):
            try:
                with open(REGISTRY_JSON_PATH, "r") as f:
                    registry = json.load(f)
            except Exception:
                registry = {}
                
        if product_id not in registry:
            registry[product_id] = {}
            
        registry[product_id][model_name] = {
            "model_name": model_name,
            "product_id": product_id,
            "metrics": metrics,
            "model_path": file_path,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(REGISTRY_JSON_PATH, "w") as f:
            json.dump(registry, f, indent=4)
            
    return file_path

def load_model(product_id: str, model_name: str):
    """
    Loads a serialized model from the models_store directory.
    """
    if not os.path.exists(REGISTRY_JSON_PATH):
        raise FileNotFoundError("Model registry not found. Train models first.")
        
    with open(REGISTRY_JSON_PATH, "r") as f:
        registry = json.load(f)
        
    if product_id not in registry or model_name not in registry[product_id]:
        raise ValueError(f"Model '{model_name}' not found for product '{product_id}'.")
        
    file_path = registry[product_id][model_name]["model_path"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model binary file not found at: {file_path}")
        
    with open(file_path, "rb") as f:
        model_obj = pickle.load(f)
        
    return model_obj

def get_product_models(product_id: str) -> list:
    """
    Returns all trained models and their metrics for a given product.
    """
    if not os.path.exists(REGISTRY_JSON_PATH):
        return []
        
    try:
        with open(REGISTRY_JSON_PATH, "r") as f:
            registry = json.load(f)
        if product_id in registry:
            return list(registry[product_id].values())
    except Exception:
        pass
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

def save_training_report(report_data: dict):
    """
    Persists a comprehensive training report for the AI Analyst to query.
    """
    with open(TRAINING_REPORT_PATH, "w") as f:
        json.dump(report_data, f, indent=4)
        
def load_training_report() -> dict:
    """
    Loads the persistent training report summary.
    """
    if not os.path.exists(TRAINING_REPORT_PATH):
        return None
    try:
        with open(TRAINING_REPORT_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None
