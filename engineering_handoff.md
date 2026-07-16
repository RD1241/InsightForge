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
14. [Version 2.0 Hardening & Optimization Upgrade Summary](#14-version-20-hardening--optimization-upgrade-summary)
15. [Version 3.0 Executive Experience & AI Decision Intelligence](#15-version-30-executive-experience--ai-decision-intelligence)
16. [Version 3.2 — Manager-First Redesign, Backend Scale Optimization, and Model Lineup Upgrade](#16-version-32--manager-first-redesign-backend-scale-optimization-and-model-lineup-upgrade)
17. [Version 4.0 — Final 5-Phase Development Cycle](#17-version-40--final-5-phase-development-cycle)
18. [Real-World Bug-Fix Round](#18-real-world-bug-fix-round)
19. [UX Honesty & Jargon-Removal Pass](#19-ux-honesty--jargon-removal-pass)

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
│   │   │   ├── models.py           # ML model classes (Ridge, Gradient Boosting, Prophet)
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
| `scikit-learn` | 1.5.0 | Ridge, HistGradientBoostingRegressor, metrics |
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
  - **ML Models (Ridge, Gradient Boosting)**: Autoregressive recursive forecast loop with price scaling and lag propagation.
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

**Models Trained** (revised in §16 — see rationale there):

1. **Ridge Regression** — `sklearn.Ridge(alpha=1.0)` inside a `Pipeline([StandardScaler, Ridge])`. L2 regularization prevents coefficient explosion on correlated lag features. Confidence interval: `±1.96 * residual_std * √step`. Kept as the interpretable linear baseline (plain Linear Regression was dropped — Ridge dominates it, so keeping both taught nothing extra).

2. **Gradient Boosting** — `sklearn.ensemble.HistGradientBoostingRegressor(max_iter=200, random_state=42)`. Histogram-based gradient boosting — the same algorithmic family as LightGBM, built into scikit-learn (no new dependency). Replaces Random Forest: gradient boosting is the empirically dominant approach for tabular retail forecasting (see §16.3), and it also trains ~4-5x faster per product than the Random Forest it replaced, since HistGradientBoostingRegressor predicts faster per call than a 100-tree Random Forest inside the recursive validation loop. Confidence interval: residual std from training set.

3. **Prophet** — `prophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, interval_width=0.95)`. Additional regressors: `price`, `promotion_flag`. Native 95% credible intervals from Bayesian posterior.

**Evaluation Protocol**:  

- **Ridge, Gradient Boosting**: Recursive walk-forward validation. The model never sees future actuals during testing. Instead, each test-step prediction is appended to the lag buffer and used as input for the next step. This replicates the exact production forecasting strategy.

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
     ├── Ridge Regression   → recursive walk-forward eval → save_model()
     ├── Gradient Boosting  → recursive walk-forward eval → save_model()
     └── Prophet            → native predict() eval → save_model()
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
Prophet provides native Bayesian credible intervals. For Ridge and Gradient Boosting, confidence intervals are approximated as `±1.96 * residual_std * √step`. The `√step` factor grows uncertainty proportionally with forecast horizon — a standard approximation for AR-type models.

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

4. **Confidence interval approximation**: Ridge/Gradient Boosting confidence intervals are estimated from training set residual standard deviation, not from a proper predictive posterior. The `√step` growth factor is a practical approximation, not a theoretically derived quantity.

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
- ~~**Future Promotions UI**: Allow users to mark planned promotion dates before generating forecasts.~~ **Done** — the What-If Simulator's promotion-day checkbox grid (§16 era) lets users toggle future promo days per scenario, seeding the correct `promotion_flag` for the recursive forecast and Prophet's regressor alike.

### Medium-Term Improvements (Engineering Investment)

- **ONNX Model Export**: Export scikit-learn models to ONNX format for version-agnostic, cross-platform inference. Eliminates pickle version fragility.
- **SQLite Registry**: Replace `models_registry.json` with an SQLite database using SQLAlchemy for proper concurrent access, query capabilities, and model versioning.
- **Model Versioning**: Track multiple training runs per product. Allow reverting to a previous model version without retraining.
- **Multi-Tenant Dataset Isolation**: Per-user dataset namespacing using session tokens. Each user uploads and trains independently on the same server instance.
- **Confidence Interval Bootstrapping**: Replace the residual std approximation with a proper bootstrapped prediction interval for the Ridge/Gradient Boosting models.
- **Automated Retraining Schedule**: Trigger retraining automatically when the dataset is updated, using a background scheduler (e.g., APScheduler or Celery).

### Long-Term Vision (Architecture Evolution)

- ~~**XGBoost / LightGBM Models**: Add gradient boosting models as additional competitors.~~ **Done in §16** — Random Forest was replaced with scikit-learn's `HistGradientBoostingRegressor` (LightGBM's algorithmic family, zero new dependency).
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
- **Glossary & Explanation**: Built a sliding help drawer that provides context-aware model descriptions (e.g. why Prophet was selected over Gradient Boosting based on validation performance) and uses LaTeX math formatting (rendered via KaTeX CDN) to explain MAE, MAPE, and R² metrics in clear business terms.

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

### 15.3 Phase 3.3: Retail AI Assistant Upgrade
- **Analyst Renaming**: Repurposed "AI Retail Analyst" references to "Retail AI Assistant" for user friendliness.
- **Quick Action Prompts**: Added a grid of 7 emoji-prefixed quick action buttons at the bottom of the chat drawer to trigger immediate business checks.

### 15.4 Phase 3.4: Styling, Animations & Micro-interactions
- **Collapsible Disclosure Styling**: Custom CSS hides summary indicators and underlines gear triggers for cleaner, more premium layouts.
- **Glassmorphic Observations**: Added soft indigo borders and gradients to AI Observation boxes.
- **Responsive Layout Breakpoints**: Designed the grid layout to wrap seamlessly from 6 columns on desktops to 3 columns on tablets, and 2 columns on mobile.

### 15.5 Cycle 6: Stability & Concurrency Polish
- **Request Token Guards**: Integrated async query matching token sequences to filter out stale data updates on rapid dropdown switches.
- **Mutual Drawer Toggles**: Coordinated open/close actions on chat and learning side panels to eliminate layering conflicts.
- **State Cleanup & Resets**: Added What-If configuration cleanup upon product switches and persisted background model fitting state.
- **PDF Export safety**: Placed defensive blocks to avoid JavaScript thread errors if statistics are absent.

### 15.6 Cycle 7: Visual Polish & UX Upgrades
- **Unified INR Currency**: Converted average price and sales stats from USD (`$`) to Rupee (`₹`).
- **Sidebar & Breadcrumb Renames**: Refactored Exploratory EDA ➔ *Sales Insights* and Forecast Engine ➔ *AI Recommendation & Forecasts*.
- **CSV Download Route**: Added the `/api/dataset/template` endpoint on the backend and placed a direct template download button on the Data Hub.
- **Glassmorphic & Mobile CSS**: Lowered slide-out drawer backgrounds to `0.8` opacity to activate the blur filter, and added media queries to hide the sidebar, collapse padding, and stack stats cards.
- **Shimmering Loading Skeletons**: Replaced spinners with shimmering card background gradients and clip-path bar-chart placeholders. Pulsing skeleton blocks are applied to KPI values during load periods.

---

## 16. Version 3.2 — Manager-First Redesign, Backend Scale Optimization, and Model Lineup Upgrade

### 16.1 Manager-first presentation layer (frontend-only, no forecasting logic touched)
Repositioned the product from "ML dashboard" to "Retail Business Intelligence SaaS" using an explicit information hierarchy — **Executive Decision → Business Insight → Supporting Evidence → Technical Details** — applied consistently instead of interleaving technical and business content:
- **Forecast page** restructured into four labeled tiers: *Today's Recommendation* (a 5-line plain-English checklist replacing the old one-sentence AI Action banner), *Business Overview* (the existing 6-tile KPI grid, demoted below the recommendation), *Forecast* (chart + What-If simulator + AI recommendation), and *Technical Details* (metrics table + model override control, both now behind a single `<details class="technical-details-disclosure">`).
- **Sales Insights** correlation section flipped: a plain-language "Sales are strongly influenced by ✓ Promotions ✓ Weekend shopping..." checklist is now the primary view; the raw correlation heatmap moved into a "View Detailed Analysis" disclosure, lazy-drawn on open (Plotly cannot size itself inside a closed `<details>` at draw time).
- **Learning Center** Glossary cards now lead with a business question ("How far off are my forecasts, on average?") instead of the raw metric name; formulas and technical definitions moved into a nested Advanced Technical Details section per card.
- **Retail AI Assistant → Retail Business Advisor**: renamed throughout (chat header, welcome message, typing indicator); quick actions were already business-framed and needed no change.
- Fixed two real bugs found during this pass: `getModelFriendlyLabel()` compared against stale string literals (`"Ridge"`/`"LinearRegression"`) that never matched the real values (`"Ridge Regression"`/`"Linear Regression"`), so friendly model names silently fell back to raw text for those two models; and the Learning Center's Glossary tab button had no click handler at all (a pre-existing latent bug, unrelated to this redesign, found while verifying it).

### 16.2 Backend scale optimization
Benchmarked the pipeline against synthetic datasets at 10k/50k/100k/200k rows (`backend/scratch/benchmark_scale.py`) before optimizing anything, per the target of comfortably handling ~100k-200k row / 50-200+ column retail datasets on typical developer hardware. Findings and fixes:

| Bottleneck found | Fix | Measured impact |
|---|---|---|
| `/api/dataset/eda` recomputed the full EDA report from scratch on every page visit | In-memory cache mirroring the existing `_cached_status` pattern, invalidated on upload/demo-load | 6.46s → ~0ms on repeat loads (200k rows) |
| `agent.py` and `/predict` each re-ran `clean_dataset()` independently instead of reusing the existing cache | Routed through the existing `get_clean_df()` cache | Removes ~0.66s of redundant work per chat message / forecast call at 200k rows |
| `registry.py`'s `save_model()` did a full read-modify-write of `models_registry.json` per model (4x per product) | Batched to one registry write per full training run (`save_registry_batch()`) | Isolated simulation: 300 sequential per-model writes cost 1.99s cumulative and scaled worse than linearly (O(n²)) — full-run batching eliminates this entirely |
| `smooth_outliers()` looped per-product with manual `.loc` writes | Replaced with `groupby(...).transform()`; output verified byte-identical to the original across 6 test cases including edge conditions | Removes per-product loop/indexing overhead; core MAD computation untouched |
| Training report writes (`save_training_report`) were unlocked while model registry writes already were | Added a lock, mirroring the existing `_registry_lock` | Closes a race between overlapping `/train` calls |
| Chat session history read-modify-write had no lock | Added an `RLock` around the full read-append-write cycle | Closes a race between concurrent messages on the same `session_id` |

All five changes are in the support/caching layer; no forecasting logic, feature engineering, or model implementation was touched. `verify_pipeline.py`, `verify_training.py`, and `verify_agent.py` all pass.

### 16.3 Model lineup: Random Forest + Linear Regression → Gradient Boosting
Replaced the 4-model lineup (Linear Regression, Ridge, Random Forest, Prophet) with a research-backed 3-model lineup:

1. **Ridge Regression** — kept as the interpretable linear baseline. Plain Linear Regression was dropped: Ridge is always equal-or-better due to L2 regularization, so training both never demonstrated anything Ridge alone didn't.
2. **Gradient Boosting** (`sklearn.ensemble.HistGradientBoostingRegressor`) — replaces Random Forest. This is evidence-based, not a preference: in the [M5 forecasting competition](https://medium.com/artefact-engineering-and-data-science/sales-forecasting-in-retail-what-we-learned-from-the-m5-competition-445c5911e2f6) — the largest public retail-demand-forecasting benchmark (~30,000 real Walmart product/store series) — the **top 50 submissions were all tree-ensemble/gradient-boosting based**. Multiple 2025 comparative studies ([arXiv:2311.00993](https://arxiv.org/pdf/2311.00993), [arXiv:2506.05941](https://arxiv.org/html/2506.05941v2)) confirm gradient boosting consistently outperforms bagged ensembles like Random Forest on tabular retail sales data, while Prophet remains the right tool specifically for seasonality/holiday effects. `HistGradientBoostingRegressor` was chosen over LightGBM/XGBoost specifically because it's histogram-based gradient boosting from the same algorithmic family, already included via the existing `scikit-learn` dependency — zero new packages, same "keep it understandable, no unnecessary complexity" constraint the rest of the project follows.
3. **Prophet** — unchanged; already the strongest performer on real validation data in this project (e.g. PRD_01: MAE 6.42–6.71 across runs) and the literature-backed choice for seasonality.

Measured effect on the existing demo dataset: Gradient Boosting fits and recursively validates a product in **~0.33-0.35s** (after one-time warmup), versus Random Forest's previously-measured ~1.4-1.7s per product — roughly 4-5x faster, which also meaningfully improves total training wall time as product count grows.

---

*This document was updated to reflect the Version 3.2 manager-first redesign, backend scale optimization, and model lineup upgrade. Prior sections describing Random Forest/Linear Regression or the 4-model lineup have been revised in place where they conflicted; §16 is the authoritative summary of what changed and why.*

---

## 17. Version 4.0 — Final 5-Phase Development Cycle

Phase 0 of this cycle (manager-first redesign, model lineup upgrade, dataset/forecast correctness fixes) is `596d18b`, already documented as §16. This section covers Phases 1–4, all committed in a single continuous session on **2026-07-16, 01:57–12:16**.

### 17.1 Phase 1: Sales Insights BI Redesign & Full Filtering (`75579c5`, 01:57)

**Backend (`eda.py`, `dataset.py`):**
- 5 new curated aggregations: revenue by category, monthly revenue trend, inventory health distribution, fast/slow movers (30-day vs. prior 30-day), and promotion impact per category. Inventory health reuses the exact same stock-status thresholds as the Forecast page's decision support so the two pages never disagree.
- `GET /api/dataset/eda` now accepts `product_ids`/`categories`/`start_date`/`end_date` query params, applied as a pandas mask before aggregation. The existing `_cached_eda` slot is only ever read/written by the exact no-filter request; filtered requests always recompute fresh.
- New `GET /api/dataset/products` endpoint (deduped catalog) to power filter pickers without bloating the EDA payload.
- **Bug found while testing filters**: narrowing the date range to a single month makes the `month` column constant, which makes Pearson correlation mathematically undefined (NaN) — and NaN isn't valid JSON, so any narrow date filter 500'd. Fixed by treating undefined correlation as 0.
- Benchmarked the filtered path at 10k–200k rows: 33–72ms, well under the unfiltered path since filtering shrinks the working set before the per-product outlier loop.

**Frontend (`app.js`, `index.html`, `styles.css`):**
- New filter bar: searchable product multi-select, category toggle chips, date range inputs + quick presets (30 days / quarter / year / all time).
- 5 new chart components reusing existing conventions (`horizontalRankingLayout` for ranking bars, the same stale-response-token + debounce pattern already used for EDA/forecast calls).
- Filters and filter options reset alongside everything else in `resetAllAppState()` on a new dataset load.

Also: `.claude/launch.json` now runs the backend with `--reload` for faster iteration — safe now that dataset freshness is a frontend `sessionStorage` concern (Phase 0) rather than a backend startup hook.

### 17.2 Phase 2: AI-Generated Charts from Chat (`17dac40`, 02:07)

**Backend (`tools.py`, `agent.py`, `routers/agent.py`):**
- New `generate_chart_spec()` tool: constrained vocabulary only (`chart_type` in `{bar, line, donut, area}`, `metric` in `{units_sold, revenue, stock}`, `dimension` in `{product, category, date, day_of_week, month}`), reusing the same filter-then-aggregate approach as `eda.py`'s `generate_eda_report()` so a chat-requested chart and the equivalent Sales Insights filtered view always agree. Every number comes from real pandas aggregation — the LLM only ever picks *which* chart to build, never the values, the same anti-hallucination pattern as the other 10 tools.
- Added as an 11th classifier tool. Product names (free text) are resolved through the existing `resolve_product_id()` fuzzy matcher first, same as `compare_sales`/`forecast_product`/etc.
- **Bug found while testing**: "show top 10 products" initially routed to the existing `top_selling_products` tool instead of generating a chart, since both tool descriptions plausibly matched. Sharpened both descriptions so visual-intent phrasing ("show", "plot", "chart", "graph", "compare X and Y") prefers `generate_chart_spec`.
- `POST /api/agent/chat` response extended additively to `{"response": str, "chart": ChartSpec|null}` — the chart is passed straight through from the tool's output, never round-tripped through the synthesis LLM call, which only ever writes a short caption.

**Frontend (`app.js`, `styles.css`):**
- New `renderChatChart()` draws a Plotly chart inside the bot's message bubble (horizontal bar for rankings, line/area for time series, donut for category composition) — reuses the same single-hue-for-magnitude convention as Sales Insights' `horizontalRankingLayout()`, sized for a chat bubble rather than a dashboard card.
- Bot messages carrying a chart get a widened bubble (`.message-has-chart`) since the default 85% max-width is sized for text only.

Verified live: all 4 of the plan's example prompts ("show top 10 products", "compare Milk and Bread sales", "pie chart of category revenue", "show revenue for the last 3 months") produce correct real-data charts; a non-chart question still routes to the existing tools unaffected.

### 17.3 Phase 3: Pre-Train AI Preview + Real Training Progress (`f1011c2`, 02:17)

> **Superseded**: the pre-train AI preview card described below was removed in its entirety during the 2026-07-16 UX honesty pass — see §19.6. It's kept here as history since it did ship and work correctly for a time; the real training-progress bar (the other half of this phase) is unaffected and remains in place today.

**Model lineup re-evaluation (no code change — findings only)**: checked real per-product metrics across all 5 demo products now that Phase 0's price fix was in place. Ridge never won outright but stayed legitimately useful (cheap, transparent, correctly excluded from "best model" whenever its R² goes negative); Gradient Boosting won on 1/5; Prophet dominated on 4/5, consistent with the dataset's strong designed weekly seasonality. Matches the M5-competition-backed expectation from the original lineup swap (§16.3) — no lineup change made.

**Pre-training AI preview** (`core/forecasting/analysis.py`, new at the time, since deleted): `analyze_pretrain_characteristics()` computed cheap, model-free descriptive stats (weekly seasonality swing, average price variation, 90-day trend shift) *before* any model was fit, producing a plain-language suggested model + reason, shown in a card above the Train button via `GET /api/forecast/pre-train-analysis`. Explicitly framed as a preview, not a promise. Verified at the time: the heuristic correctly flagged the demo dataset's 52% weekly swing and suggested Prophet, matching the actual training outcome.

**Real training progress** (`train_pipeline.py`, `routers/forecasting.py`) — still in place:
- Module-level progress dict + lock, updated from inside `_train_single_product()` (the function that runs in each worker thread) via a `try`/`finally` so it advances on every code path — success, the too-little-history skip, or a mid-fit exception — not just the happy path. Denominator is products (not products × 3 models), so Prophet's existing silent per-product failure never leaves the bar stuck short of 100%.
- New `GET /api/forecast/train/progress` endpoint, polled every 750ms from the frontend while `POST /train` is in flight. Replaces the old fully-fake `setInterval` random-increment bar with real current-product/model/percent and a client-side ETA.

### 17.4 Phase 4: Visual Polish, Chat Resize, Forecast Page Decluttering (`f61a9bf`, 12:16)

**Design tokens & inline-style cleanup** (`styles.css`, `index.html`):
- New `--space-1..8` and `--radius-sm/md/lg` custom properties in `:root`, matching the existing color/font token pattern.
- Reconciled the worst duplicated inline styles the Phase 1–3 work surfaced: 4 copies of the same `ai-observation-box` style block, the 6-tile Executive Summary Board (each tile had ~200 chars of duplicated inline style; consolidated into a single `.grid-6 .stat-box` modifier — later resized to `.grid-5` when the Forecast Reliability tile was removed, §19.5), the CSV template download link, and the What-If simulator's day-checkbox cells.

**Forecast page decluttering**: merged the "AI Recommendation" and "What should you do next?" cards into one card with two sections and a divider — both answer the same underlying question and had identical `border-pulse` styling as two separate cards. All existing element IDs preserved untouched.

**What-If simulator sizing**: larger price slider (custom-styled thumb) and larger promo-day checkboxes (18px, was 13px) — CSS only, no functional change.

**Chat panel**: draggable resize handle on the left edge (clamped 380–720px, width preference persisted per-browser in `localStorage`), wider default (440px, was 380px), and `renderMarkdownSafely()` extended to handle headings and bulleted/numbered lists.

**Chat tone**: added an explicit rule to `analyst_system_prompt` (`agent.py`) preferring plain business language over ML/statistics jargon (MAE, R²) unless the user's own question is technical.

One open item noted at the time: the chat panel's exact on-screen positioning couldn't get a clean geometric read via the automated browser tooling despite the CSS being verified correct at the source/CSSOM level. **Resolved** — later confirmed via the user's own screenshots that the panel renders and opens correctly; this was a tooling artifact of the automated browser environment, not a real bug.

---

## 18. Real-World Bug-Fix Round

Found by testing against the user's own uploaded retail CSV rather than the synthetic demo dataset — all three commits land on **2026-07-16, 15:00–16:00**, immediately after the Phase 1–4 cycle above.

### 18.1 CSV Upload: Non-UTF-8 Encoding (`2faae25`, 15:00)

Real-world CSV exports, especially from Excel on Windows, are frequently not plain UTF-8 — a byte-order mark, or the Windows-1252 codepage (triggered by currency symbols or other special characters), are both common and previously caused an unhandled `UnicodeDecodeError` → 422 on upload. Found via a live user-reported failure.

New `read_csv_robust()` tries `utf-8-sig` (covers plain UTF-8 and UTF-8-with-BOM) then falls back to `cp1252` (a safe last resort — single-byte, so it can decode any byte sequence). Applied at both call sites that read a user's file: the upload validation *and* `get_active_df()` (every later read of the saved file) — fixing only the first would have made upload appear to succeed while every subsequent page load failed the same way.

### 18.2 CSV Upload: Excel Files Mislabeled as `.csv` (`02b683f`, 15:12)

Root cause of the upload failure that survived the encoding fix: the user's file wasn't actually CSV text at all — it was a genuine `.xlsx` file (confirmed via magic bytes: `PK\x03\x04`, the ZIP signature every `.xlsx` is built on) that had been named/saved with a `.csv` extension. No encoding fallback can parse binary Excel data as text, so this needed a different fix: detect real file content instead of trusting the extension.

- New `_looks_like_excel()` checks magic bytes (ZIP for `.xlsx`/`.xlsm`, OLE2 for legacy `.xls`) rather than the filename, since the filename is exactly what's unreliable here.
- `read_tabular_upload()` dispatches to `pd.read_excel()` or the existing CSV encoding chain accordingly. Upload now also genuinely accepts `.xlsx`/`.xls` files directly, not just mislabeled ones.
- Storage normalized: whatever format comes in, the parsed DataFrame is written back out as clean CSV, so every downstream reader stays simple.
- A third real-world price-column alias (`unit_price`, alongside `selling_price`/`base_price`) found via the user's actual file — added to `preprocessor.py`'s price-alias chain.
- Added `openpyxl` to `backend/requirements.txt` (required by `pd.read_excel()` for `.xlsx`).

### 18.3 Prophet Yearly-Seasonality Bug, ML Jargon Cleanup Round 1, Profit Charts, Chat Maximize (`b39d2c6`, 16:00)

**Prophet accuracy investigation** (`models.py`) — triggered by real user feedback that forecast accuracy was under 60% on their own dataset:
- Root cause: `yearly_seasonality` was hardcoded `True`, forcing Prophet to fit a yearly cycle on datasets that had never actually completed one. On the user's real 89-day upload this was catastrophic (one product's R² was **-1860**; MAPE over 1000%).
- Prophet's own `yearly_seasonality='auto'` is not a safe substitute — it requires ≥730 days (2 full years), which incorrectly disabled real, learnable yearly seasonality on the 699-day demo dataset (R² dropped from 0.93 to 0.48 under `'auto'`). Settled on a self-computed **365-day** (one full cycle observed) threshold instead, verified correct on both datasets: demo back to R²=0.92, the real dataset's worst product from -1860 to -0.11.
- Evaluated the user's suggestion to replace the 3-model comparison with a single "best in the world" model — declined, with evidence: post-fix, Prophet/Ridge/Gradient Boosting won 10/6/4 of 20 real products respectively, so no single model dominates. A bigger single model would also need *more* data than 89 days, not less.

**ML jargon cleanup — round 1**: `get_recommendation_reason()` (`registry.py`) rewritten from raw metric text ("MAE = 6.71, R² = 0.92") to plain language with a qualitative confidence note; exact numbers stayed in the already-collapsed Advanced Technical Details table only. *(This was later found to be incomplete — the exported PDF report had its own separate, unfixed code path. See §19.1.)*

**Profit metric**: AI chart generation didn't support "profit" — a real "profit and loss" chat request silently fell back to revenue. Added `profit` metric to `generate_chart_spec` (prefers a real `profit` column, else derives from `revenue - cost_price`, else declines gracefully). Also fixed a chart-type bug found while verifying: a "pie chart... over the last 30 days" request produced an unreadable 30-slice donut — date-dimension requests now force `line` regardless of literal wording.

**Chat maximize**: the Phase 4 drag handle alone wasn't discoverable in real testing; added a visible maximize/restore button in the chat header.

---

## 19. UX Honesty & Jargon-Removal Pass

Two commits (`bed56c9`, `14586e5`, 2026-07-16 16:49–18:22) plus a longer follow-on session addressing feedback relayed from a second AI the user consulted, verified against the actual repo rather than taken at face value, per the standing "verify before trusting a pasted proposal" practice. **As of this section, §19.3 onward reflects the current working tree and has not yet been committed** — check `git log`/`git status` for the latest commit state before relying on commit hashes for that portion.

### 19.1 Export Report & Dashboard Jargon Cleanup, AI Summary Bug Fix, INR Currency (`bed56c9`, 16:49)

The round-1 jargon cleanup (§18.3) only touched the on-screen recommendation card — the exported PDF report still dumped raw MAE/MAPE/R² and named all 3 candidate models directly. Fixed properly this time:
- Recommendation text and always-visible UI labels no longer name the underlying algorithm anywhere (dashboard badge, chart subtitle, `get_recommendation_reason()`). Real model names/metrics stay available in the existing collapsed Advanced Technical Details section and a new **"Appendix A: Technical Model Comparison (Reference Only)"** section at the end of the exported report, for the user's own viva/thesis reference — a deliberate product decision (see the two-audience discussion below).
- Fixed the report's "AI Analyst Summary", found to be almost always blank ("No custom audit logs generated yet.") — it read `prodData.recommendation_reason`, a field that only exists on the per-request `/api/forecast/predict` response (`state.forecastData`), never on the static training report object it was actually reading from.
- A related bug in the same code: the "has the user chatted" check counted `<p>` tags rather than message bubbles, and the welcome message alone has 2 paragraphs — so it always thought the user had chatted, and even when they had, it grabbed only the *last paragraph* of the final reply rather than the whole thing. Fixed to count bubbles and use the full last bubble's content.
- Chat agent had no currency instruction anywhere in its system prompt and defaulted to `$` instead of ₹ for an Indian retail app — added an explicit instruction to `analyst_system_prompt`.

**Product framing established this round** (in response to a second AI's relayed proposal, itself independently reaching a very similar conclusion): InsightForge serves two audiences with one interface — a retail manager who should never need to know Prophet/MAE/R² exist, and the student/examiner who needs exactly those details for a viva. Resolution: hide model names, counts, and raw metrics from every default-visible surface; keep them fully available, clearly labeled as reference/technical, behind the existing collapsed disclosures and the new report appendix. The 3-model internal comparison itself was kept (not consolidated to one model) — evidence-backed, per §18.3's win-distribution finding.

### 19.2 Forecast Accuracy Tile: Raw Percentage Removed (`14586e5`, 18:22)

The Business Overview "Forecast Accuracy" tile already led with a qualitative label (Poor/Good/Excellent) but still showed a raw "56.0% accuracy" sub-badge underneath — the last number-shaped leftover of the same jargon-removal effort. Removed the number entirely (qualitative label only); deleted the now-unused DOM element and its population code rather than leaving it dead.

### 19.3 Prophet What-If Price-Clip Bug

`ProphetModel.predict()` clipped its `price` regressor to the *exact* historical `[min, max]` range it trained on. For any product whose historical price barely varied (common for fixed-MRP retail items), this made the What-If Simulator's price slider a complete no-op — confirmed directly: a +30% price scenario shifted a real product's prediction by <1%, since the requested price was clamped straight back to the unchanged historical value regardless of the multiplier.

Fixed by widening the clip to `[train_min × 0.7, train_max × 1.3]` — matching the slider's own allowed range (`price_multiplier` ∈ [0.7, 1.3]) — so every legitimate scenario reaches the model while the clip still catches genuinely out-of-bounds values. The 0.7/1.3 bound is now a single named constant (`PRICE_SCENARIO_MULTIPLIER_MIN/MAX` in `models.py`) imported by `train_pipeline.py`'s clamp and `routers/forecasting.py`'s `Query` validation — previously all three hardcoded the same literal independently, found during this session's code-review pass (§19.7).

**Important caveat, found while diagnosing a specific user report**: on the user's own uploaded dataset, this fix didn't visibly change the What-If slider's effect, because the deeper issue is upstream of any code fix — every one of that dataset's 20 products has a `unit_price` column ranging ~₹15–450 essentially independent of which product it is, with near-zero correlation to units sold (-0.26 to +0.24 across products). That pattern is consistent with price having been randomly generated per row rather than tied to the actual product, most likely in whatever script generated that "real" dataset for the capstone. No model can honestly find a price effect in that data — the slider doing little for that specific file is the model being correct, not broken.

### 19.4 Forecast Chart Redesign

Per user feedback that the chart still "looked machine generated": ran the existing `drawForecastChart()` colors through the project's `dataviz` skill validator rather than eyeballing them (indigo `#6366f1` / amber `#f59e0b`, the app's own `--accent`/`--warning` tokens) — both pass colorblind-safety (CVD ΔE 140.9/81.2/148.2, well above the ≥12 target) and contrast-vs-surface; amber alone fails the dark-mode "lightness band" aesthetic check, but wasn't changed since it's a pre-existing app-wide brand token, not something to fork just for one chart.

- **Simplified legend labels**: "95% Confidence Interval" → "Likely Range"; "Baseline Forecast (Normal Price)" → "Without This Change".
- **Reduced clutter**: the uncertainty band and the scenario-comparison line no longer render simultaneously — only one shows at a time depending on whether a What-If scenario is active, instead of 4 overlapping visual elements at once.
- **Visual polish**: removed per-day markers on the forecast line (was cluttering a 30–90 point series); replaced with a single 9–10px end-of-line dot via a per-point `marker.size` array. 2px line widths throughout (was 2.5–3px).
- Also fixed: a toast (`"Forecast generated using ${model}."`) that leaked the raw model name on every forecast/scenario run — now a generic `"Forecast updated."`.

### 19.5 Forecast Reliability Tile Removed; Export Report Confidence De-Starred

Every one of the user's real dataset's 20 products scored in the worst "Poor accuracy / Low reliability" band (R² -0.48 to 0.16, MAPE 31–95%) — confirmed as a genuine dataset characteristic (short ~4-month history plus the near-random price column, §19.3), not a display bug. Per direct user request:
- Removed the "Forecast Reliability" star-rating stat tile from the Business Overview grid entirely (kept "Forecast Accuracy" as the one confidence signal on the dashboard). Grid resized from 6 columns (`.grid-6`) to 5 (`.grid-5`, base rule + all 3 responsive breakpoints + child-selector rules renamed together).
- The R² computation itself was kept (feeds a new low-confidence explanatory caption shown only when a product's confidence is genuinely low, rather than removing the honest signal outright — a manager acting on an order-quantity recommendation still needs to know when the underlying forecast has no demonstrated skill).
- "Poor"/"Low" qualitative wording reworded to "Limited" throughout (dashboard tile, export report), on the reasoning that the same honest signal reads as "here's a data limitation to be aware of" rather than "this app doesn't work" — while still not fabricating a fake accuracy number.
- Stars removed from the exported report's "Forecast Confidence" field per explicit request (was the last remaining place `getConfidenceLabel()` returned star-emoji strings; now plain text — "Excellent"/"Good"/"Fair"/"Limited" — matching the dashboard tile's convention).

### 19.6 Pre-Train AI Preview Removed Entirely

Per direct user request, removed the "Before You Train — AI Preview" feature shipped in Phase 3 (§17.3) completely rather than leaving it disabled: the HTML card, its CSS classes, the `loadPreTrainAnalysis()` JS function and its element references, the `GET /api/forecast/pre-train-analysis` backend route, and the `core/forecasting/analysis.py` module it depended on (deleted outright — `analyze_pretrain_characteristics()` was its only export). Confirmed via repo-wide grep that no other code references any of it. The real training-progress bar (the other half of Phase 3) is unaffected.

### 19.7 Final Code-Review Pass

Ran an 8-angle, 7-agent parallel code review (per the project's `code-review` skill, high effort) against every change made since the last push, covering §19.3–19.6 above. 10 findings confirmed and fixed:

| # | Bug | Fix |
|---|---|---|
| 1–2 | `if (modelData && modelData.MAPE)` / `.R2` treated a genuinely trained value of exactly `0` as absent (falsy-zero), silently keeping a stale fallback value | Changed to explicit `!== undefined` checks |
| 3 | The new low-confidence note wasn't cleared on a forecast error or product switch — a stale note from a previous product could stay visible under a different one's blank/errored summary | Added `setLowConfidenceNote(null)` to the error path |
| 4 | The note said "for this product" but computed a dataset-wide date span, identical regardless of which product was selected | Reworded to "in your dataset" — an accurate description of what the number actually measures |
| 5 | `.text-2xs` CSS class used in 6 places (including the new note) was never defined anywhere in `styles.css` — a pre-existing gap, not new to this session | Added the missing class definition (`0.68rem`) |
| 6 | The `0.7`/`1.3` price-scenario bound was hardcoded independently in 3 files (`models.py`, `routers/forecasting.py`, `train_pipeline.py`) | Extracted to `PRICE_SCENARIO_MULTIPLIER_MIN/MAX` in `models.py`, imported by the other two |
| 7 | `getAccuracyTileLabel()` re-encoded the same MAPE thresholds already in `METRICS_EXPLAINER.MAPE.ratingFn` (Learning Center glossary) | Now derives its label from `ratingFn()` and remaps "Poor" → "Limited" |
| 8 | The low-confidence note recomputed a date-span via raw `Date` subtraction instead of reading `state.datasetStats.days_span`, which the backend already computes and the app already displays elsewhere | Reads the existing field directly |
| 9 | The note's logic was 4 levels of inline nested `if`s, unlike its sibling label functions | Extracted to named `getLowConfidenceNote()`/`setLowConfidenceNote()` |
| 10 | `.italic` utility class — used in 3 pre-existing places (the "Simulation Estimate" badge, both What-If baseline delta labels) — was, like `.text-2xs`, never actually defined | Added the missing class definition |

Also re-ran `verify_pipeline.py`, `verify_training.py`, and `verify_agent.py` (all pass) to confirm nothing in this session's changes broke the existing pipeline, training, or the 11 AI Analyst tools. `verify_agent.py` initially failed with a `KeyError` — traced to the script hardcoding a demo-dataset product ID (`PRD_01`) while the user's own real dataset (different ID convention, `P001`) was the currently active one on disk; not a real bug, confirmed by temporarily swapping in the demo dataset (backing up and restoring the user's real file around the test) and re-running clean.

---

*This document was last updated 2026-07-16 to add §17–§19 (the final 5-phase development cycle, the real-world bug-fix round, and the UX honesty/jargon-removal pass) and to correct stale references to the retired Linear Regression/Random Forest model pair throughout §2, §4.2, §9.7, §12, and §13. §19 documents working-tree state that was not yet committed at time of writing — check `git log` for the current commit boundary before treating §19.3 onward as shipped.*
