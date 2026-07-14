# InsightForge — Engineering Handoff & Final Architecture

> **Version:** 1.0.0 (Production-Hardened)  
> **Classification:** Internal Engineering Reference  
> **Purpose:** Complete technical handoff document for system review, academic viva, and future maintenance.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Folder Structure](#2-folder-structure)
3. [Technology Stack & Dependencies](#3-technology-stack--dependencies)
4. [Complete API Reference](#4-complete-api-reference)
5. [ML Pipeline — Deep Dive](#5-ml-pipeline--deep-dive)
6. [AI Analyst Pipeline — Deep Dive](#6-ai-analyst-pipeline--deep-dive)
7. [Frontend Architecture & Flow](#7-frontend-architecture--flow)
8. [Data Flow — End-to-End](#8-data-flow--end-to-end)
9. [Engineering & Design Decisions](#9-engineering--design-decisions)
10. [Security Hardening Summary](#10-security-hardening-summary)
11. [Trade-offs](#11-trade-offs)
12. [Known Limitations](#12-known-limitations)
13. [Future Roadmap](#13-future-roadmap)

---

## 1. System Overview

InsightForge is an **AI-powered Retail Decision Support System** that combines supervised machine learning demand forecasting with a natural language AI analyst. It is designed as a college capstone project with production-quality engineering.

```
                          ┌─────────────────────────────────────────────┐
                          │             FRONTEND  (SPA)                 │
                          │   index.html   app.js   styles.css          │
                          └──────────────────┬──────────────────────────┘
                                             │ REST API (HTTP/JSON)
                          ┌──────────────────▼──────────────────────────┐
                          │         FASTAPI BACKEND (Python)            │
                          │   /api/dataset   /api/forecast   /api/agent │
                          │                  main.py                    │
                          └──────┬────────────────────┬─────────────────┘
                                 │                    │
             ┌───────────────────▼─────┐   ┌──────────▼─────────────────────┐
             │   ML FORECASTING CORE   │   │   AI AGENT CORE                │
             │   preprocessor.py       │   │   agent.py   tools.py          │
             │   models.py             │   │   llm.py (Groq / Ollama)       │
             │   train_pipeline.py     │   └────────────────────────────────┘
             │   registry.py  eda.py   │
             └─────────────────────────┘
```

The system serves both the frontend SPA and the API from a **single local server** — Uvicorn runs FastAPI which mounts the frontend static files at `/`, so a single `uvicorn main:app` command is sufficient.

---

## 2. Folder Structure

```
d:/Project_C/
├── .env                        # LLM environment config (Groq API key, Ollama host)
├── .gitignore                  # Git exclusion list (venv, __pycache__, .pkl files, active data)
├── README.md                   # User-facing getting started guide
│
├── frontend/                   # Static SPA served by FastAPI StaticFiles mount
│   ├── index.html              # Single-page application shell (all views, modals, PDF template)
│   ├── app.js                  # All JavaScript logic (~1,200 lines; SPA state machine)
│   └── styles.css              # All CSS styling (~1,000 lines; dark mode, glassmorphism, animations)
│
├── backend/
│   ├── main.py                 # FastAPI app entry point; router mounting; CORS; static file serving
│   ├── requirements.txt        # Pinned Python dependency manifest
│   │
│   ├── routers/                # Thin HTTP layer (input validation, error masking, async dispatch)
│   │   ├── dataset.py          # /api/dataset/*  — upload, demo, status, preview, EDA
│   │   ├── forecasting.py      # /api/forecast/* — train, report, predict, compare
│   │   └── agent.py            # /api/agent/*    — chat endpoint
│   │
│   ├── core/
│   │   ├── forecasting/        # Entire ML forecasting engine
│   │   │   ├── preprocessor.py     # Schema detection, cleaning, aggregation, feature engineering
│   │   │   ├── models.py           # ML model classes (LR, Ridge, RF, Prophet)
│   │   │   ├── train_pipeline.py   # Orchestrates training, evaluation, parallel execution, forecast generation
│   │   │   ├── registry.py         # Persists trained models (pickle) and metadata (JSON)
│   │   │   ├── eda.py              # Exploratory Data Analysis report generator
│   │   │   └── synthetic_data.py   # Demo dataset generator (synthetic retail)
│   │   │
│   │   └── agent/              # AI Analyst brain
│   │       ├── llm.py              # Unified async LLM client (Groq or Ollama), session memory
│   │       ├── agent.py            # Two-stage orchestrator: classify → tool call → synthesize
│   │       └── tools.py            # 10 structured analyst tool functions
│   │
│   ├── data/                   # Runtime data directory (created on first run)
│   │   ├── active_dataset.csv  # Currently loaded dataset
│   │   ├── synthetic_retail_data.csv  # Pre-generated demo dataset
│   │   └── sessions/           # Persistent chat session JSON files (one per session_id)
│   │
│   ├── models_store/           # Trained model artifacts (created after first training)
│   │   ├── models_registry.json    # Model index: product → model → path + metrics
│   │   ├── training_report.json    # Latest training run summary report
│   │   └── *.pkl                   # Serialized scikit-learn / Prophet model objects
│   │
│   └── scratch/                # Engineering verification scripts (not shipped to production)
│       ├── verify_pipeline.py      # Tests the preprocessing pipeline end-to-end
│       ├── verify_training.py      # Tests model training and forecast generation
│       └── verify_agent.py         # Tests all 10 AI Analyst tools
│
├── data/                       # External reference datasets (Kaggle, sample CSVs)
│   └── train.csv               # Kaggle Store Item Demand dataset (used for testing)
│
└── venv/                       # Python virtual environment (not committed to git)
```

---

## 3. Technology Stack & Dependencies

### Backend (Python 3.11+)

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.111.0 | Async web framework for API routing |
| `uvicorn` | 0.30.1 | ASGI server; runs FastAPI |
| `pandas` | 2.2.2 | Data manipulation and time series processing |
| `numpy` | 1.26.4 | Numerical computation |
| `scikit-learn` | 1.5.0 | Linear Regression, Ridge, Random Forest, metrics |
| `prophet` | 1.1.5 | Facebook Prophet additive time series model |
| `httpx` | 0.27.0 | Async HTTP client for LLM API calls |
| `python-dotenv` | 1.0.1 | `.env` file loading for config |
| `python-multipart` | 0.0.9 | File upload handling (`multipart/form-data`) |
| `plotly` | 5.22.0 | (Optional) Server-side chart generation |

### Frontend (Vanilla)

| Technology | Purpose |
|---|---|
| HTML5 | SPA shell structure, semantic sections, print template |
| Vanilla CSS | Dark mode, glassmorphism, animations, responsive layout |
| Vanilla JavaScript (ES2022) | State machine, API calls, chart rendering, chat UI |
| Plotly.js (CDN) | Interactive forecast chart with historical + future overlay |
| Lucide Icons (CDN) | Icon library used for badges, buttons, and avatar icons |
| Google Fonts — Inter | Typography (loaded via CSS import) |

### LLM Providers (Configurable)

| Provider | Config Key | Notes |
|---|---|---|
| **Groq** (default) | `LLM_PROVIDER=groq` | Free tier cloud API; `llama-3.3-70b-versatile` |
| **Ollama** (local) | `LLM_PROVIDER=ollama` | Fully offline; runs any local GGUF model |

---

## 4. Complete API Reference

All endpoints are prefixed under `http://localhost:8000/api/...`

### 4.1 Dataset Router — `/api/dataset`

#### `POST /api/dataset/upload`
Upload a user CSV dataset.
- **Body**: `multipart/form-data`, field `file` (CSV)
- **Constraints**: `.csv` extension only, max 50 MB
- **Schema Detection**: Supports native schema (`date, store_id, product_id, units_sold`) and Kaggle schema (`date, store, item, sales`)
- **Response (200)**: `{ message, report: { is_valid, stats, warnings, errors } }`
- **Response (422)**: Validation failed with error details
- **Response (413)**: File too large
- **Security**: Sanitizes `file.filename` via `pathlib.Path.name` before building temp file path. Clears status cache on each call.

#### `POST /api/dataset/load-demo`
Loads the pre-generated synthetic retail dataset.
- **Response (200)**: `{ message, report }`
- **Behavior**: Generates `synthetic_retail_data.csv` if absent; copies to `active_dataset.csv`; validates and caches status.

#### `GET /api/dataset/status`
Returns the profile summary and statistics of the currently active dataset.
- **Response (200)**: `{ loaded: bool, profile_summary: str, stats: { row_count, unique_stores, unique_products, unique_categories, start_date, end_date }, warnings: [] }`
- **Caching**: Reads from in-memory `_cached_status` if available; avoids re-reading the full CSV on UI polling.

#### `GET /api/dataset/preview?rows=50`
Returns the first N rows of the active dataset for tabular preview.
- **Query Params**: `rows` (default: 50)
- **Response (200)**: `{ columns: [], data: [{}] }`

#### `GET /api/dataset/eda`
Performs full Exploratory Data Analysis.
- **Response (200)**: Structured EDA report with:
  - `dataset_overview`, `descriptive_statistics`
  - `top_products`, `category_performance`
  - `sales_trend` (daily time series)
  - `weekly_seasonality`, `monthly_seasonality`
  - `correlation_matrix`
  - `outliers` (IQR-based detection, up to 10 per product)

---

### 4.2 Forecasting Router — `/api/forecast`

#### `POST /api/forecast/train?smooth_outliers=true`
Triggers the full model training pipeline for all products.
- **Query Params**: `smooth_outliers` (bool, default: `true`) — enables rolling MAD smoothing before training
- **Execution**: Offloaded to a thread pool executor (`run_in_executor`) to avoid blocking the FastAPI event loop
- **Response (200)**: `{ message, report: { timestamp, dataset_rows, total_products_trained, average_mae, average_mape, products: { <pid>: { best_model, best_metrics, all_models, training_time_seconds } } } }`
- **Persists**: `models_store/training_report.json` and per-product `.pkl` files

#### `GET /api/forecast/report`
Returns the latest persisted training report.
- **Response (200)**: Full training report JSON
- **Response (404)**: If no training has been run yet

#### `GET /api/forecast/predict?product_id=PRD_01&model_name=Prophet&horizon_days=30&price_multiplier=1.0&promo_days=`
Generates an N-day future demand forecast, supporting scenario overrides and business decision support.
- **Query Params**:
  - `product_id` (required)
  - `model_name` (optional, defaults to recommended model)
  - `horizon_days` (7–90, default 30)
  - `price_multiplier` (0.7–1.3, default 1.0)
  - `promo_days` (comma-separated days, e.g., `"5,6,12,13"`, default `""`)
- **Response (200)**: `{ product_id, product_name, category, model_used, metrics, recommendation_reason, forecast_horizon_days, price_multiplier_applied, simulated_price, history, forecast, decision_support: { current_stock, avg_daily_sales, safety_stock_threshold, status, reorder_date, recommended_reorder_qty, stockout_days_projected, revenue_at_risk, projected_stock } }`
- **Forecast Method**:
  - **Prophet**: Direct `predict(future_df)` call with extra price/promo regressor vectors.
  - **ML Models (LR, Ridge, RF)**: Autoregressive recursive forecast loop with price scaling and lag propagation.
  - **Performance**: Asynchronously offloaded to worker threads via `asyncio.to_thread` to keep the event loop non-blocking.

#### `GET /api/forecast/compare?product_id=PRD_01`
Returns validation metrics for all models trained on a product.
- **Response (200)**: Array of `{ model_name, product_id, metrics: { MAE, RMSE, MAPE, R2 }, model_path, trained_at }`

---

### 4.3 Agent Router — `/api/agent`

#### `POST /api/agent/chat`
Sends a natural language message to the AI Retail Analyst.
- **Body**: `{ message: str, session_id: str }` (JSON)
- **Session**: `session_id` is used for persistent per-session memory (stored as `data/sessions/{id}.json`)
- **Response (200)**: `{ response: str }` — natural language response
- **Internals**: Runs the two-stage agent pipeline (classify → tool call → synthesize)

#### `GET /api/health`
Simple health check endpoint.
- **Response (200)**: `{ status: "healthy", service: "InsightForge API", version: "1.0.0" }`

---

## 5. ML Pipeline — Deep Dive

### 5.1 Data Ingestion & Schema Detection (`preprocessor.py`)

The pipeline accepts two dataset schema formats:

| Column | Native InsightForge | Kaggle Alternative |
|---|---|---|
| Date | `date` | `date` |
| Store | `store_id` | `store` |
| Product | `product_id` | `item` |
| Sales | `units_sold` | `sales` |
| Optional | `product_name, category, price, stock_on_hand, promotion_flag` | (auto-generated) |

**Column mapping** (`standardize_dataframe`):  
If schema is Kaggle-style, columns are renamed. Missing enrichment columns (`product_name`, `category`, `price`, `stock_on_hand`, `promotion_flag`) are auto-generated from a hardcoded `PRODUCT_CATALOG` of 50 retail products. Products not in the catalog get deterministic fallback values using hash-based pricing.

**Validation** (`validate_dataset`):  
Checks for required columns, parses date formats, counts nulls, flags negatives in `units_sold`/`price`, and warns if the dataset covers fewer than 90 days.

**Cleaning** (`clean_dataset`):  
- Drops duplicates
- Parses dates (coerce invalid to NaT → drop)
- Forward-fills price and stock per product
- Fills `units_sold` NaN as 0
- Sorts chronologically

**Aggregation** (`aggregate_to_product_level`):  
Folds store-level rows into product-level daily time series:
- `units_sold`: sum across stores
- `price`: mean across stores
- `stock_on_hand`: sum across stores
- `promotion_flag`: max across stores (if any store promoted → promoted)

**Outlier Smoothing** (`smooth_outliers`):  
Optional pre-training step (toggle via `smooth_outliers` query param). Applies rolling Median Absolute Deviation (MAD) with window=14 and threshold=3.0. Sales spikes outside `median ± 3*MAD` are clipped to the threshold bounds. Benchmarked to reduce average MAE by ~4.9%.

**Grid Alignment** (`ensure_regular_daily_grid`):  
Before feature engineering, every product's time series is reindexed to the complete global date range. Retail datasets often omit dates with zero sales (no transaction recorded). Without this step, `shift(N)` produces lag features that correspond to N *rows* ago rather than N *calendar days* ago — silently corrupting every feature.

**Feature Engineering** (`build_features`):  

| Feature | Description |
|---|---|
| `day_of_week` | 0 (Mon) – 6 (Sun) |
| `month` | 1 – 12 |
| `is_weekend` | Binary: 1 if Sat/Sun |
| `day_of_year` | 1 – 365 |
| `units_sold_lag_1` | Sales 1 calendar day ago |
| `units_sold_lag_7` | Sales 7 calendar days ago |
| `units_sold_lag_14` | Sales 14 calendar days ago |
| `units_sold_roll_mean_7` | 7-day rolling mean (shift-1 applied to prevent leakage) |
| `units_sold_roll_mean_30` | 30-day rolling mean (shift-1 applied to prevent leakage) |
| `promo_lag_1` | Was yesterday a promotion day? (captures hangover effect) |

**Leakage Prevention**: Rolling means use `shift(1)` inside the `transform` before rolling. This ensures the rolling window never includes the current day's target value. Remaining NaNs at the start of each product's history are forward-filled, then zero-filled — never backward-filled.

---

### 5.2 Model Training (`train_pipeline.py`)

**Training Split**: Last 30 days of data held out as test set. All earlier data is used for training. This is a strict chronological split — no shuffling.

**Parallel Execution**: Products are trained in parallel using `ThreadPoolExecutor(max_workers=min(n_products, 4))`. ThreadPoolExecutor (not ProcessPoolExecutor) is used because:
- Prophet's Stan/C++ backend is incompatible with `multiprocessing` pickling on Windows
- Threads share memory space — no serialization overhead

**Thread-Safe Registry**: `registry.py` uses a `threading.Lock()` (`_registry_lock`) around all read-modify-write operations on `models_registry.json` to prevent concurrent write corruption during parallel training.

**Models Trained**:

1. **Linear Regression** — `sklearn.LinearRegression`. No regularization. Fast baseline. Confidence interval: `±1.96 * residual_std * √step`.

2. **Ridge Regression** — `sklearn.Ridge(alpha=1.0)` inside a `Pipeline([StandardScaler, Ridge])`. L2 regularization prevents coefficient explosion on correlated lag features. Confidence interval: same method as LR.

3. **Random Forest** — `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`. Ensemble of 100 decision trees. Captures non-linear interactions. Confidence interval: residual std from training set.

4. **Prophet** — `prophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, interval_width=0.95)`. Additional regressors: `price`, `promotion_flag`. Native 95% credible intervals from Bayesian posterior.

**Evaluation Protocol**:  

- **Linear Regression, Ridge, Random Forest**: Recursive walk-forward validation. The model never sees future actuals during testing. Instead, each test-step prediction is appended to the lag buffer and used as input for the next step. This replicates the exact production forecasting strategy.

- **Prophet**: Direct `model.predict(test_df)` on the test date range. Prophet is an additive regression model fitted on the datetime `ds` field — it does not use autoregressive lag features and therefore does not require recursive multi-step simulation. This methodological difference is explicitly documented in code.

**Metrics Computed** (`evaluate_predictions`):
- **MAE** (Mean Absolute Error): Average absolute deviation in units
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily
- **MAPE** (Mean Absolute Percentage Error): Relative error; zero-sales days excluded
- **R²** (Coefficient of Determination): Proportion of variance explained

**Model Recommendation Heuristics** (`_train_single_product`):  
1. Reject models with negative R² (worse than a simple mean predictor), unless all models fail.
2. Among remaining valid models, select the one with lowest MAE.
3. Persist a human-readable `recommendation_reason` alongside the recommendation.

---

### 5.3 Model Registry (`registry.py`)

Models are persisted as Python `pickle` files in `models_store/`. A JSON index (`models_registry.json`) maps each product+model combination to its filename and validation metrics.

**Key design**: Only the **filename** (e.g., `PRD_01_prophet.pkl`) is stored in the registry, not absolute paths. This makes the workspace fully portable across machines. When loading, the filename is resolved relative to `MODELS_STORE_DIR` at runtime. Backwards compatibility for old absolute paths is also handled.

---

### 5.4 Future Forecast Generation (`train_pipeline.py` → `generate_future_forecast`)

1. Loads best model from registry (or user-specified model)
2. Seeds a `sales_buffer` with the last 30 days of actual historical sales
3. For each future day (1 → horizon):
   - Constructs lag/rolling features from the buffer
   - Predicts with the model
   - Appends the prediction back to the buffer for the next step
4. Confidence intervals expand proportionally with `√step` to model compounding uncertainty
5. Returns historical actuals (last 30 days) alongside future predictions for chart overlay

---

## 6. AI Analyst Pipeline — Deep Dive

### 6.1 Two-Stage Architecture (`agent.py`)

Every user chat message goes through a two-stage LLM pipeline:

```
User Message
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1: CLASSIFIER LLM CALL (JSON mode)           │
│  Input: system prompt with tool list + product     │
│         lookup dictionary + user message           │
│  Output: { "tool_name": "...", "arguments": {...} } │
│  Temperature: 0.1 (near-deterministic routing)     │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
              Tool Execution (Python)
                  tools.py functions
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: ANALYST LLM CALL (natural language)       │
│  Input: system prompt + session memory (5 turns)   │
│         + structured tool output JSON as context   │
│  Output: professional natural language response    │
│  Temperature: 0.1                                   │
└─────────────────────────────────────────────────────┘
```

This two-stage design ensures the analyst **never hallucinates numbers**. All statistics, forecasts, and metrics come from Python computation and are injected into the LLM context as ground truth. The LLM only handles natural language generation.

### 6.2 Product Resolution (`agent.py` → `resolve_product_id`)

The classifier extracts product identifiers from free-text queries. These may be imprecise (e.g., "milk", "PRD01", "organic milk"). The resolver applies three matching strategies in order:
1. Exact case-insensitive `product_id` match
2. Substring match on `product_name` (e.g., "milk" matches "Organic Milk 1L")
3. Partial `product_id` substring match

### 6.3 Session Memory (`llm.py`)

Each session maintains a sliding window of the last 10 messages (5 user, 5 assistant turns), persisted as a JSON file at `data/sessions/{session_id}.json`. Session IDs are sanitized (alphanumeric + `-_` only) before constructing file paths.

### 6.4 The 10 Analyst Tools (`tools.py`)

| Tool | Description |
|---|---|
| `list_products()` | Returns full product catalog grouped by category |
| `top_selling_products(limit)` | Ranks products by total sales volume and revenue |
| `low_stock_products(threshold_days)` | Products with fewer than N days of stock remaining |
| `inventory_health()` | Categorizes all products: out-of-stock / understock / low / healthy / overstock |
| `sales_summary()` | Total volume, total revenue, average price, category breakdown |
| `compare_sales(product_id)` | Current 30 days vs prior 30 days sales change |
| `forecast_product(product_id)` | 30-day predicted demand via best model |
| `explain_forecast_decomposition(product_id)` | Causal decomposition into trend %, seasonality %, promotion % |
| `model_comparison(product_id)` | All model metrics with human-friendly MAE/MAPE/R² explanations |
| `generate_business_insights()` | Slow movers, critical restock alerts, revenue summary |

### 6.5 Forecast Decomposition — How It Works

The decomposition is entirely Python-computed (not LLM-generated) to prevent hallucination.

**For ML models**: Three counterfactual forecasts are run:
1. Full forecast (actuals)
2. Forecast with all promotions zeroed out
3. Forecast with seasonality features set to a neutral Wednesday in June

The difference between (1) and (2) gives the promotion impact. The difference between (1) and (3) gives the seasonality impact. The remainder is the baseline trend. Percentages are clamped to prevent edge-case arithmetic blowouts.

**For Prophet**: The Prophet model natively decomposes forecasts into `trend`, `weekly`, and `yearly` components. These are summed over the 30-day horizon and expressed as percentages.

### 6.6 LLM Client (`llm.py`)

`call_llm` is a unified async function that routes to either Groq or Ollama based on `LLM_PROVIDER` environment variable. Uses `httpx.AsyncClient` to avoid blocking the FastAPI async event loop.

- **Classifier calls** use `response_format: {"type": "json_object"}` to guarantee JSON output
- **Analyst calls** use default text mode
- Both use `temperature: 0.1` for consistency

---

## 7. Frontend Architecture & Flow

### 7.1 Single Page Application Structure

The frontend is a single HTML file (`index.html`) with all CSS in `styles.css` and all logic in `app.js`. There is no build step, no bundler, and no framework. This was an intentional choice for college deployment simplicity.

**View Sections** (toggled via `data-section` attributes):
- **Home** — Landing page with dataset upload and demo load
- **Dashboard** — EDA charts, sales trends, outliers table, correlation matrix
- **Forecast Workspace** — Product selector, model selector, Plotly chart, metrics comparison table
- **AI Analyst** — Chat panel with session memory
- **Settings** — LLM configuration information

### 7.2 State Management (`app.js`)

All UI state lives in a single `state` object:

```javascript
const state = {
    activeSection: "home",
    activeProduct: null,       // Currently selected product_id
    activeModel: null,         // null = best recommender
    trainedReport: null,       // Full training report from /api/forecast/report
    forecastData: null,        // Current /api/forecast/predict response
    datasetStats: null         // Dataset stats from /api/dataset/status
};
```

### 7.3 Key Frontend Flows

**Dataset Upload Flow**:
1. User drags/drops or selects a CSV
2. Client-side 50 MB size check before upload
3. `POST /api/dataset/upload`
4. If valid: status cache updates, dataset stats shown, user navigated to Dashboard
5. If invalid: validation errors listed in a warnings panel

**Training Flow**:
1. User clicks Train button (with optional outlier smoothing toggle)
2. `POST /api/forecast/train` (server processes in background thread pool)
3. On 200: training report stored in `state.trainedReport`
4. `updateWorkspaceForProduct` called to populate product dropdown, metrics table, model selector
5. `GET /api/forecast/predict` fetches and renders the Plotly forecast chart

**Forecast Chart (`Plotly.js`)**: Renders two traces:
- Historical actuals (last 30 days of real data, solid line)
- Future predictions with ±95% confidence interval shading, dashed line

**Chat Flow**:
1. User types message → Enter or Send button
2. `POST /api/agent/chat` with `session_id = "default_session"`
3. Typing indicator displayed during await
4. Response rendered via `renderMarkdownSafely(text)` (XSS-safe)

**PDF Export Flow**:
1. Builds a hidden print-specific template (`#print-template`) with product stats, model comparison table, AI summary
2. Captures the Plotly forecast chart as a `data:image/png` URL via `Plotly.toImage`
3. Injects the image into the print template
4. Calls `window.print()` (browser handles PDF download)

### 7.4 Security Helpers in app.js

```javascript
// Escapes HTML special characters before DOM insertion
function escapeHtml(text) { ... }

// Safe markdown parser: escape first, then render bold/italic/code/links
function renderMarkdownSafely(text) { ... }
```

`renderMarkdownSafely` first runs `escapeHtml` on the raw text to neutralize any injected HTML, then applies regex-based markdown patterns (`**bold**`, `*italic*`, `` `code` ``, `[link](url)`) to produce safe markup.

### 7.5 Offline Detection

`app.js` polls `/api/health` every 5 seconds. If the server goes offline, a full-screen blurred overlay (`#offline-overlay`) is shown. It auto-hides when the server returns.

---

## 8. Data Flow — End-to-End

### 8.1 Dataset Lifecycle

```
CSV File (user / Kaggle / synthetic)
     │
     ▼
POST /api/dataset/upload or /load-demo
     │
     ▼
standardize_dataframe()     ← Column renaming, enrichment, catalog lookup
     │
     ▼
validate_dataset()          ← Required columns, date parsing, null checks, negative value checks
     │ (if valid)
     ▼
Saved as active_dataset.csv ← Single active dataset per server instance
     │
     ▼
_cached_status updated      ← In-memory stats cache for fast /status polling
```

### 8.2 Training Lifecycle

```
active_dataset.csv
     │
     ▼
clean_dataset()             ← Drop dupes, parse dates, ffill missing, sort chronologically
     │
     ▼
aggregate_to_product_level() ← Fold stores → product daily totals
     │
     ▼
smooth_outliers()           ← (Optional) MAD clipping on extreme sales spikes
     │
     ▼
ensure_regular_daily_grid() ← Reindex each product to full calendar (prevent lag corruption)
     │
     ▼
build_features()            ← Calendar + lag + rolling features (no leakage)
     │
     ▼
ThreadPoolExecutor          ← Parallel training per product (max 4 workers)
     │
     ├── Linear Regression → recursive walk-forward eval → save_model()
     ├── Ridge Regression  → recursive walk-forward eval → save_model()
     ├── Random Forest     → recursive walk-forward eval → save_model()
     └── Prophet           → native predict() eval → save_model()
     │
     ▼
Best model selected (lowest MAE, positive R² filter)
     │
     ▼
models_registry.json updated (thread-safe lock)
training_report.json saved
```

### 8.3 Prediction Lifecycle (ML models)

```
GET /api/forecast/predict?product_id=PRD_01&horizon_days=30
     │
     ▼
load_model(product_id, model_name)   ← Loads .pkl via relative path from registry
     │
     ▼
Seed sales_buffer = last 30 days of actuals
     │
     ▼
For step 1 → 30:
  lag_1, lag_7, lag_14 = sales_buffer[-1], [-7], [-14]
  roll_7, roll_30 = mean(buffer[-7:]), mean(buffer[-30:])
  prediction = model.predict(feat_row)
  confidence = ±1.96 * residual_std * √step
  sales_buffer.append(prediction)   ← RECURSIVE!
     │
     ▼
Return predictions, lower_bound, upper_bound arrays
```

### 8.4 AI Agent Lifecycle

```
POST /api/agent/chat { message, session_id }
     │
     ▼
get_product_lookup_str()            ← Build product → ID lookup for classifier context
     │
     ▼
call_llm(classifier_messages, require_json=True)  ← Groq/Ollama API call (async)
     │
     ▼
Parse JSON: { tool_name, arguments }
     │
     ▼
Dispatch to tools.py function       ← Pure Python computation, no hallucination
     │
     ▼
get_session_history(session_id)     ← Load last 10 messages from disk
     │
     ▼
call_llm(analyst_messages)          ← Final NL generation with tool output as context (async)
     │
     ▼
add_message_to_history()            ← Persist updated session to disk
     │
     ▼
Return final_response               ← Back to /api/agent router → client
```

---

## 9. Engineering & Design Decisions

### 9.1 Single-Port Deployment
FastAPI's `StaticFiles` mount serves the frontend SPA at `/`. This means only one process, one port, and zero CORS complexity in production. For college demos, this is ideal.

### 9.2 Thread Pool for Training (not Process Pool)
Prophet's Stan/C++ MCMC backend cannot be pickled across processes on Windows (`multiprocessing` fails). `ThreadPoolExecutor` bypasses this while still achieving real parallelism for I/O-bound operations and pseudo-parallelism for CPU-bound training (limited by GIL, but Stan bypasses the GIL via C extensions).

### 9.3 Recursive Walk-Forward Evaluation (not batch predict)
A simple `model.predict(test_set)` would leak future actuals into lag features on every test row. The recursive approach replicates the real production workflow — lag features are constructed from the model's own predictions at each step. This correctly measures how the model would actually perform in deployment.

### 9.4 Prophet Evaluated Differently by Design
Prophet is not an auto-regressive model. It fits a global additive function on time directly. Putting it inside a recursive lag-update loop would be both incorrect and meaningless. Evaluating it natively and documenting this difference is the academically sound decision.

### 9.5 Groq + Ollama Dual Provider
The `.env` switch between Groq (free-tier cloud) and Ollama (fully local) allows the application to run without any external dependency. For deployment where internet connectivity is uncertain (such as a college demo), flipping to Ollama with a local model ensures the AI Analyst always works.

### 9.6 JSON Registry Instead of SQL
A lightweight `models_registry.json` was chosen over SQLite or PostgreSQL because:
- No database setup required for college deployment
- Human-readable and inspectable
- Sufficient performance for dozens of models
- Protected by `threading.Lock()` for concurrent writes

### 9.7 Confidence Intervals via Residual Std
Prophet provides native Bayesian credible intervals. For LR, Ridge, and RF, confidence intervals are approximated as `±1.96 * residual_std * √step`. The `√step` factor grows uncertainty proportionally with forecast horizon — a standard approximation for AR-type models.

### 9.8 Session Memory as Flat JSON Files
Chat sessions are persisted as JSON files per `session_id`. This is simpler than Redis or a database and survives server restarts. The 10-message sliding window prevents unbounded growth.

### 9.9 Two-Stage Agent Architecture
Separating classification from synthesis prevents the analyst LLM from both routing and generating simultaneously. A classifier LLM focused purely on JSON routing (with near-zero temperature) achieves reliable tool selection. The analyst LLM then receives grounded, structured data and only needs to produce natural language.

---

## 10. Security Hardening Summary

| Issue | Fix Applied | Location |
|---|---|---|
| XSS — chat messages | `renderMarkdownSafely()`: escape then safe-parse markdown | `app.js` |
| XSS — tables, dropdowns | `escapeHtml()` on all server-generated values injected into `innerHTML` | `app.js` |
| Path Traversal — upload | `pathlib.Path(file.filename).name` strips directory components | `dataset.py` |
| Internal Error Leakage | All `str(e)` in HTTP responses replaced with generic messages; tracebacks logged server-side only | `dataset.py`, `forecasting.py`, `agent.py` |
| CORS Credentials Spec Violation | `allow_credentials=False`; wildcard origin + credentials is spec-invalid | `main.py` |
| Session ID Traversal | `session_id` sanitized to alphanumeric + `[-_]` before file path construction | `llm.py` |
| Race Condition — Registry | `threading.Lock()` guards all registry read-modify-write cycles | `registry.py` |
| Unknown Agent Tool | Explicit `else` branch returns structured error instead of `None` | `agent.py` |
| Status Cache Staleness | Cache cleared at the start of every new upload or demo-load operation | `dataset.py` |

---

## 11. Trade-offs

| Decision | Benefit | Cost |
|---|---|---|
| Vanilla JS SPA (no React/Vue) | Zero build tooling, simple college demo | Imperative DOM manipulation; harder to scale |
| Single `active_dataset.csv` | Simple state, no multi-user complexity | Only one dataset can be active at a time; no concurrent users |
| ThreadPool instead of async training | Works around Windows pickling limits | Python GIL limits true CPU parallelism for LR/RF |
| Pickle for model serialization | Simple, built-in, no extra dependencies | Pickle files are version-sensitive; not safe for untrusted sources |
| Flat JSON registry | No database setup, human-readable | Not suitable for thousands of models or concurrent multi-user writes |
| 30-day test split | Standard retail evaluation period | Shorter datasets may have a disproportionately large test set |
| Prophet evaluated natively | Correct evaluation methodology | Cannot compare head-to-head using identical evaluation loop |
| Session memory capped at 10 messages | Prevents context window overflow in LLM | Loses older conversation context in long sessions |
| `temperature=0.1` for LLM calls | Highly deterministic, repeatable responses | Reduces creativity and variability in analyst prose |

---

## 12. Known Limitations

1. **Single active dataset**: The system supports one active dataset at a time. Switching datasets requires retraining from scratch. There is no multi-user or multi-dataset isolation.

2. **No authentication**: The API has no authentication layer. Any process on the same network can call all endpoints. Not suitable for production multi-tenant deployment without adding JWT or API key validation.

3. **Prophet training time**: Prophet trains Stan's MCMC sampler in C++. Training 5 products takes ~5–15 seconds. Training 50+ products could take several minutes. The async executor prevents server freezing but the user must wait.

4. **Confidence interval approximation**: LR/Ridge/RF confidence intervals are estimated from training set residual standard deviation, not from a proper predictive posterior. The `√step` growth factor is a practical approximation, not a theoretically derived quantity.

5. **Pickle serialization**: scikit-learn and Prophet models are pickled. Pickle files are version-sensitive — if the Python or library version changes, old `.pkl` files may not load. No version metadata is stored in the registry.

6. **Future promotions & price overrides**: Now fully supported. Planners can configure future price adjustments and toggle daily promotions inside the What-If Simulator panel, and these overrides propagate dynamically to recursive lags and future Prophet regressors.

7. **Stock simulation**: If the uploaded dataset doesn't include `stock_on_hand`, it is synthetically simulated using a cyclic formula. This simulation is approximate and should not be used for actual replenishment planning without real stock data.

8. **No streaming for LLM responses**: The AI Analyst sends one `await call_llm(...)` blocking call. There is no streaming/chunked response — users see a typing indicator until the full response is ready.

9. **Single session ID**: The UI hardcodes `session_id = "default_session"`. Multiple simultaneous browser tabs would share the same conversation history.

10. **Plotly renders on main thread**: For very large datasets or many products, Plotly chart rendering may briefly block the browser's main thread. No Web Worker offloading is implemented.

---

## 13. Future Roadmap

### Immediate Improvements (Low Effort, High Value)

- **Streaming LLM Responses**: Replace single `await call_llm` with a server-sent event (SSE) stream to progressively render analyst responses and eliminate typing wait time.
- **Multi-Session Support**: Pass a user-generated `session_id` from the UI (UUID) instead of hardcoding `"default_session"`. Each browser tab would maintain its own conversation.
- **Future Promotions UI**: Allow users to mark planned promotion dates before generating forecasts, so the recursive forecast seeds the correct `promotion_flag` for future days.

### Medium-Term Improvements (Engineering Investment)

- **ONNX Model Export**: Export scikit-learn models to ONNX format for version-agnostic, cross-platform inference. Eliminates pickle version fragility.
- **SQLite Registry**: Replace `models_registry.json` with an SQLite database using SQLAlchemy for proper concurrent access, query capabilities, and model versioning.
- **Model Versioning**: Track multiple training runs per product. Allow reverting to a previous model version without retraining.
- **Multi-Tenant Dataset Isolation**: Per-user dataset namespacing using session tokens. Each user uploads and trains independently on the same server instance.
- **Confidence Interval Bootstrapping**: Replace the residual std approximation with a proper bootstrapped prediction interval for LR/Ridge/RF models.
- **Automated Retraining Schedule**: Trigger retraining automatically when the dataset is updated, using a background scheduler (e.g., APScheduler or Celery).

### Long-Term Vision (Architecture Evolution)

- **XGBoost / LightGBM Models**: Add gradient boosting models as additional competitors. XGBoost typically outperforms Random Forest on tabular retail data and is highly competitive with Prophet for non-seasonal products.
- **DeepAR / LSTM Integration**: Add deep learning sequence models (Amazon DeepAR, LSTM) via PyTorch for products with complex, non-linear demand patterns.
- **Multi-Product Cross-Learning**: Train a single global model on all products simultaneously (with product embeddings as features) instead of independent per-product models. Useful for products with insufficient individual history.
- **Real-Time Data Integration**: Replace CSV upload with a live database or data warehouse connector (BigQuery, Snowflake, PostgreSQL) for continuous ingestion of point-of-sale transactions.
- **Competitor Benchmarking API**: Integrate external APIs (e.g., Azure AutoML, Amazon Forecast) for automatic comparison of InsightForge models against cloud-native forecasting services.
- **Explainability Dashboard**: Integrate SHAP (SHapley Additive exPlanations) values to render feature importance plots directly in the dashboard — showing exactly how much each feature (lag, promotion, day of week) contributed to each specific forecast.

## 14. Version 2.0 Hardening & Optimization Upgrade Summary

InsightForge was significantly expanded and hardened in Version 2. Below is an engineering log of the upgrades and changes implemented.

### 14.1 Business Intelligence & Decision Support Engine (Backend)
The inventory planning logic is centralized in Python (`train_pipeline.py`):
- **Safety Stock Threshold**: Calculated dynamically as \(2.0 \times \text{Average Daily Sales}\) (representing a 2-day supply buffer).
- **Target Stock Level**: Calculated as \(7.0 \times \text{Average Daily Sales}\) (representing a 7-day target replenishment buffer).
- **Replenishment Triggers**: Scans the future forecast and identifies the first day when projected stock levels fall below safety stock, recommending an order date and a restorative order quantity (`target_stock_threshold - current_stock`).
- **Revenue at Risk**: Accumulates unmet demand units (lost sales) and multiplies them by the average price to project financial stockout risk.

### 14.2 What-If Simulator Panel (Frontend)
- **Controls**: Includes a custom price multiplier slider (`[0.7, 1.3]`) and a calendar day grid for toggling promotions on days 1–30.
- **Quick Scenarios**: Integrated preset profiles (Holiday Sale, Weekend Promo, Supplier Delay, Discount, Demand Surge, and Baseline) to speed up operational simulation.
- **Scenario Save & Compare**: Logs multiple simulated scenarios under user-defined names. Computes a performance scorecard including estimated revenue, volume delta, stockout days, and highlights the optimal scenario with a recommended winner badge using the formula:
  \[\text{Score} = \text{Projected Revenue} - \text{Revenue at Risk}\]

### 14.3 Learning Center drawer
- **Glossary & Explanation**: Built a sliding help drawer that provides context-aware model descriptions (e.g. why Prophet was selected over Random Forest based on validation performance) and uses LaTeX math formatting (rendered via KaTeX CDN) to explain MAE, MAPE, and R² metrics in clear business terms.

### 14.4 Performance & Stability Audits
- **Async Event Loop Offloading**: Transformed predictive endpoints to run CPU-bound ML recursive predictions in worker threads using `asyncio.to_thread`.
- **In-Memory Caches**: Added an active dataset cache in `routers/dataset.py` and a binary model object cache in `registry.py` to prevent redundant file system reads and CPU parsing.
- **Registry Locking**: Protected concurrent reads on `models_registry.json` using the existing thread-safe reentrant lock `_registry_lock`.
- **Plotly Memory Cleanup**: Purges old Plotly graph instances with `Plotly.purge()` to prevent DOM memory leaks over long browser sessions.

## 15. Version 3.0 Executive Experience & AI Decision Intelligence

Version 3.0 redesigns the complete user experience around the **30-second rule**—allowing a non-technical retail manager to immediately make decisions—and applies **progressive disclosure** to keep technical details available for academic examination.

### 15.1 Phase 3.1: Data Hub & Business Overview (EDA) Redesign
- **Dataset Health Check**: Replaced raw technical descriptions with a direct health badge (`✓ Ready for Planning`) and next-steps guidance.
- **Key Sales Drivers & Influences**: Renamed the correlation matrix card and implemented a dynamic **AI Driver Analysis** text card that parses price elasticity, promotional peaks, and weekend traffic correlations into plain language.
- **Inline AI Observation Cards**: Added observations below all EDA trendlines and seasonality charts answering: *What happened?*, *Why?*, and *Recommended Action*.

### 15.2 Phase 3.2: Forecast Page & Executive Summaries
- **Business Health Executive Summary Board**: Shows expected revenue, stockout days, revenue risk, prediction accuracy ($(100 - \text{MAPE})\%$), and qualitative forecast reliability (derived from $R^2$) before any charts are drawn.
- **AI Action Plan Alert**: Prominently highlights simulated replenishment reorder recommendations at the top of the forecasting page.
- **Progressive Disclosure Details**: Encapsulated standard validation tables (MAE, MAPE, R²) inside collapsible HTML `<details>` blocks.
- **Model Engine Branding**: Relabeled models with intuitive descriptive prefixes (e.g. `★ Forecast Engine (Facebook Prophet)`).

---

*This document was generated as the final engineering handoff for InsightForge v3.0.0 following the Version 3.0 development and redesign phases.*
