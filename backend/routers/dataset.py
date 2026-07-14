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
_cached_df = None

def get_active_df() -> pd.DataFrame:
    """
    Returns the active dataset in-memory cache, loading it from disk if not cached.
    """
    global _cached_df
    if _cached_df is None:
        if not os.path.exists(ACTIVE_DATASET_PATH):
            raise FileNotFoundError("No active dataset found.")
        _cached_df = pd.read_csv(ACTIVE_DATASET_PATH)
    return _cached_df

_cached_clean_df = None

def get_clean_df() -> pd.DataFrame:
    """
    Returns the cleaned active dataset in-memory cache, cleaning it if not cached.
    """
    global _cached_clean_df
    if _cached_clean_df is None:
        raw_df = get_active_df()
        _cached_clean_df = clean_dataset(raw_df)
    return _cached_clean_df

def clear_status_cache():
    """
    Clears the internal cache when a new dataset is active.
    """
    global _cached_status, _cached_df, _cached_clean_df
    _cached_status = None
    _cached_df = None
    _cached_clean_df = None

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Endpoint to upload a CSV dataset. Validates the structure and,
    if valid, saves it as the active dataset.
    """
    global _cached_status
    clear_status_cache()  # Clear cache on new upload (HIGH-11)
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    # Enforce file size limit of 50 MB
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    file_content = await file.read(MAX_FILE_SIZE + 1)
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail="File size exceeds the 50 MB limit. Please upload a smaller CSV."
        )
    await file.seek(0)
    
    # Sanitize filename to prevent path traversal (HIGH-02)
    from pathlib import Path
    safe_filename = Path(file.filename).name
    temp_path = os.path.join(DATA_DIR, f"temp_{safe_filename}")
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
        import logging
        logger = logging.getLogger("insightforge")
        logger.error(f"Dataset upload failed due to file parsing error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=422,
            content={
                "message": "Dataset validation failed due to file parsing error.",
                "report": {
                    "is_valid": False,
                    "errors": ["Failed to read CSV file. Please make sure it is a valid CSV format."],
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
    clear_status_cache()  # Clear cache on demo load (HIGH-11)
    try:
        # Check if demo data exists, generate if missing
        if not os.path.exists(DEMO_DATASET_PATH):
            generate_synthetic_data(DEMO_DATASET_PATH)
            
        # Copy to active dataset
        shutil.copyfile(DEMO_DATASET_PATH, ACTIVE_DATASET_PATH)
        
        # Load to read metadata
        df = get_active_df()
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
            "message": "Demo dataset loaded successfully.",
            "report": report
        }
    except Exception as e:
        clear_status_cache()
        import logging
        logger = logging.getLogger("insightforge")
        logger.error(f"Failed to load demo dataset: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load demo dataset. Please try again.")

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
        df = get_active_df()
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
        import logging
        logger = logging.getLogger("insightforge")
        logger.error(f"Error reading dataset status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error reading dataset status. Please try again.")

@router.get("/preview")
async def get_dataset_preview(rows: int = 50):
    """
    Returns the first N rows of the active dataset for tabular preview in the frontend.
    """
    if not os.path.exists(ACTIVE_DATASET_PATH):
        raise HTTPException(status_code=404, detail="No active dataset found.")
        
    try:
        # Use cache for preview
        df = get_active_df().head(rows)
        # Convert NaN values to None for clean JSON serialization
        df_clean = df.where(pd.notnull(df), None)
        return {
            "columns": df_clean.columns.tolist(),
            "data": df_clean.to_dict(orient="records")
        }
    except Exception as e:
        import logging
        logger = logging.getLogger("insightforge")
        logger.error(f"Failed to generate preview: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate preview. Please try again.")

@router.get("/eda")
async def get_eda_report():
    """
    Performs EDA on the active dataset and returns statistics,
    seasonality data, correlation values, and outlier lists.
    """
    if not os.path.exists(ACTIVE_DATASET_PATH):
        raise HTTPException(status_code=404, detail="No active dataset found.")
        
    try:
        df = get_active_df()
        df_clean = clean_dataset(df)
        eda_data = generate_eda_report(df_clean)
        return eda_data
    except Exception as e:
        import logging
        logger = logging.getLogger("insightforge")
        logger.error(f"Failed to generate EDA report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate EDA report. Please try again.")

from fastapi.responses import StreamingResponse
import io

@router.get("/template")
async def get_csv_template():
    # Return a simple CSV template with headers date, store, item, sales
    csv_data = "date,store,item,sales\n2025-01-01,1,PRD_01,100\n2025-01-02,1,PRD_01,110\n"
    return StreamingResponse(
        io.StringIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=insightforge_template.csv"}
    )
