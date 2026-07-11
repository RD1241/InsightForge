import os
import shutil
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Standardize path imports
from core.forecasting.preprocessor import validate_dataset, clean_dataset
from core.forecasting.synthetic_data import generate_synthetic_data
from core.forecasting.eda import generate_eda_report

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ACTIVE_DATASET_PATH = os.path.join(DATA_DIR, "active_dataset.csv")
DEMO_DATASET_PATH = os.path.join(DATA_DIR, "synthetic_retail_data.csv")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory status cache to avoid loading large CSVs repeatedly on status checks
_cached_status = None

def clear_status_cache():
    """
    Clears the internal cache when a new dataset is active.
    """
    global _cached_status
    _cached_status = None

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Endpoint to upload a CSV dataset. Validates the structure and,
    if valid, saves it as the active dataset.
    """
    global _cached_status
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    temp_path = os.path.join(DATA_DIR, f"temp_{file.filename}")
    try:
        # Save upload to a temporary file first
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Load and validate
        df = pd.read_csv(temp_path)
        report = validate_dataset(df)
        
        if report["is_valid"]:
            # Move temp file to active dataset path
            if os.path.exists(ACTIVE_DATASET_PATH):
                os.remove(ACTIVE_DATASET_PATH)
            shutil.move(temp_path, ACTIVE_DATASET_PATH)
            
            # Cache the status results to speed up UI transitions
            stats = report["stats"]
            profile_summary = (
                f"Active dataset contains {stats['row_count']:,} records spanning from "
                f"{stats['start_date']} to {stats['end_date']}. It profiles "
                f"{stats['unique_products']} products across {stats['unique_stores']} stores "
                f"grouped into {stats['unique_categories']} categories."
            )
            _cached_status = {
                "loaded": True,
                "profile_summary": profile_summary,
                "stats": stats,
                "warnings": report["warnings"]
            }
            
            return {
                "message": "Dataset uploaded and validated successfully.",
                "report": report
            }
        else:
            # Remove temp file if invalid
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return JSONResponse(
                status_code=422,
                content={
                    "message": "Dataset validation failed.",
                    "report": report
                }
            )
    except Exception as e:
        # Cleanup temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(
            status_code=422,
            content={
                "message": "Dataset validation failed due to file parsing error.",
                "report": {
                    "is_valid": False,
                    "errors": [f"Failed to read CSV: {str(e)}"],
                    "warnings": [],
                    "stats": {}
                }
            }
        )

@router.post("/load-demo")
async def load_demo_dataset():
    """
    Loads the pre-generated synthetic retail dataset as the active dataset.
    Generates it if it doesn't exist.
    """
    global _cached_status
    try:
        # Check if demo data exists, generate if missing
        if not os.path.exists(DEMO_DATASET_PATH):
            generate_synthetic_data(DEMO_DATASET_PATH)
            
        # Copy to active dataset
        shutil.copyfile(DEMO_DATASET_PATH, ACTIVE_DATASET_PATH)
        
        # Load to read metadata
        df = pd.read_csv(ACTIVE_DATASET_PATH)
        report = validate_dataset(df)
        
        # Cache the status
        stats = report["stats"]
        profile_summary = (
            f"Active dataset contains {stats['row_count']:,} records spanning from "
            f"{stats['start_date']} to {stats['end_date']}. It profiles "
            f"{stats['unique_products']} products across {stats['unique_stores']} stores "
            f"grouped into {stats['unique_categories']} categories."
        )
        _cached_status = {
            "loaded": True,
            "profile_summary": profile_summary,
            "stats": stats,
            "warnings": report["warnings"]
        }
        
        return {
            "message": "Demo retail dataset loaded successfully.",
            "report": report
        }
    except Exception as e:
        clear_status_cache()
        raise HTTPException(status_code=500, detail=f"Failed to load demo dataset: {str(e)}")

@router.get("/status")
async def get_dataset_status():
    """
    Returns the status and summary metrics of the currently loaded active dataset.
    Uses in-memory caching to avoid repeatedly loading heavy CSV files.
    """
    global _cached_status
    if not os.path.exists(ACTIVE_DATASET_PATH):
        clear_status_cache()
        return {
            "loaded": False,
            "message": "No dataset currently active. Please upload a dataset or load the demo."
        }
        
    # Return cached status if available
    if _cached_status is not None:
        return _cached_status
        
    try:
        # Read dataset head and stats if cache is empty
        df = pd.read_csv(ACTIVE_DATASET_PATH)
        report = validate_dataset(df)
        
        # Get a programmatic P0 Dataset Intelligence summary
        stats = report["stats"]
        profile_summary = (
            f"Active dataset contains {stats['row_count']:,} records spanning from "
            f"{stats['start_date']} to {stats['end_date']}. It profiles "
            f"{stats['unique_products']} products across {stats['unique_stores']} stores "
            f"grouped into {stats['unique_categories']} categories."
        )
        
        _cached_status = {
            "loaded": True,
            "profile_summary": profile_summary,
            "stats": stats,
            "warnings": report["warnings"]
        }
        return _cached_status
    except Exception as e:
        clear_status_cache()
        raise HTTPException(status_code=500, detail=f"Error reading dataset status: {str(e)}")

@router.get("/preview")
async def get_dataset_preview(rows: int = 50):
    """
    Returns the first N rows of the active dataset for tabular preview in the frontend.
    """
    if not os.path.exists(ACTIVE_DATASET_PATH):
        raise HTTPException(status_code=404, detail="No active dataset found.")
        
    try:
        df = pd.read_csv(ACTIVE_DATASET_PATH, nrows=rows)
        # Convert NaN values to None for clean JSON serialization
        df_clean = df.where(pd.notnull(df), None)
        return {
            "columns": df_clean.columns.tolist(),
            "data": df_clean.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")

@router.get("/eda")
async def get_eda_report():
    """
    Performs EDA on the active dataset and returns statistics,
    seasonality data, correlation values, and outlier lists.
    """
    if not os.path.exists(ACTIVE_DATASET_PATH):
        raise HTTPException(status_code=404, detail="No active dataset found.")
        
    try:
        df = pd.read_csv(ACTIVE_DATASET_PATH)
        df_clean = clean_dataset(df)
        eda_data = generate_eda_report(df_clean)
        return eda_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate EDA report: {str(e)}")
