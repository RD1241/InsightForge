# InsightForge: AI-Powered Retail Decision Support System

InsightForge is an advanced, production-quality AI-Powered Retail Decision Support System designed to help small stores and supermarkets optimize their inventory. Built using FastAPI, Vanilla HTML/CSS/JS, Scikit-learn, and Prophet, it integrates time-series demand forecasting with an interactive AI Retail Analyst.

---

## 🌟 Key Features

1. **Dataset Ingestion & Auto-Mapping (Data Hub)**:
   - Supports CSV uploads (including non-UTF-8 encodings and genuine Excel files saved with a `.csv` extension) and detects schemas dynamically.
   - Automatically maps columns from Kaggle datasets (`date, store, item, sales`) or custom schemas, including several real-world price-column aliases (`selling_price`, `base_price`, `unit_price`).
   - Generates mock metadata (categories, prices) and simulates inventory stock cycles if missing.
   - Built-in **Retail Simulator Demo** to generate 2 years of realistic seasonal retail demand.
   - Dataset Health Check with plain-language warnings, and a Clear Dataset control that resets everything for a fresh upload.

2. **Sales Insights (Business Intelligence Dashboard)**:
   - Filterable by product, category, and date range (with quick presets), backing 10+ curated charts: sales trend, weekly/monthly seasonality, top products, revenue by category, inventory health distribution, fast/slow movers, and promotion impact.
   - Every chart's "Explain" button asks the AI Analyst to summarize it in plain language.
   - Automatic anomalous sales detection using the **Interquartile Range (IQR)** method, with an option to smooth outliers before training.

3. **Time Series Forecasting Engine**:
   - Compares **Ridge Regression**, **Gradient Boosting** (`HistGradientBoostingRegressor`), and **Prophet** — the recommended model per product is chosen automatically from real validation performance, never hardcoded.
   - **Strict Time-Based Split**: Evaluates models on a 30-day chronological split (never random shuffle) to prevent future data leakage.
   - **Residual-based 95% Confidence Intervals**: Computes residuals variance on ML models for confidence bounds matching Prophet.
   - **Recursive Multi-Step Forecasting**: Automatically feeds prior predictions back into lags and rolling averages to simulate future dates.
   - **What-If Simulator**: adjust price (±30%) and toggle future promotion days per scenario, with scenario-vs-baseline comparison and a saved scenario history.
   - Plain-language "Forecast Confidence" and business-decision output (reorder quantity, stockout risk, revenue at risk) by default; raw model names and MAE/MAPE/R² metrics are available on demand behind an "Advanced Technical Details" disclosure, for technical/academic reference without cluttering the primary view.
   - **Export Report**: generates a printable PDF-style business report (dataset summary, forecast confidence, AI recommendation) with the same technical details in a clearly labeled reference appendix.
   - Persistent Model Registry saving binary `.pkl` files and metadata index in `models_registry.json`.
   - Generates a central `training_report.json` summarizing optimization runs.

4. **AI Retail Analyst & Chat Panel**:
   - Collapsible, resizable dashboard panel serving as an intelligent forecasting copilot (Retail Business Advisor).
   - **Structured Tool Calling**: Agent decides which deterministic tool to call (`top_selling_products`, `low_stock_products`, `inventory_health`, `generate_chart_spec`, etc.) instead of executing raw python code — 11 tools in total.
   - **AI-Generated Charts from Chat**: ask for a chart in plain English ("compare Milk and Bread", "pie chart of category revenue") and the agent builds a real Plotly chart from actual data, inline in the chat.
   - **Evidence-Based, Plain-Language Explanations**: Agent summarizes structured JSON outputs in business terms by default, only using technical terms (MAE, MAPE, R²) if the user's own question is technical.
   - **Sparkles Explain Action**: Every dashboard chart has an "Explain" button that triggers the Analyst to summarize the visualization in natural language.
   - Unified support for **Groq API** (free tier cloud) and **Ollama** (local offline models) with session memory.

5. **Learning Center**:
   - A drawer of business-question-first ML explainers (e.g. "How far off are my forecasts, on average?") with the technical formula/definition available underneath for anyone who wants it — separates the plain-language reading from the exam-ready technical one.

---

## 📁 Project Architecture

```
backend/
  main.py                # FastAPI entry point, mounts routers and serves static frontend
  requirements.txt       # Unified Python dependencies (fastapi, pandas, scikit-learn, prophet)
  data/                  # Storage for active datasets & simulated retail data
  models_store/          # Binary pickle models, registry index, and training run reports
  routers/
    dataset.py           # Upload, loading, preview, and automatic EDA routes
    forecasting.py       # Model training, prediction, and registry comparison routes
    agent.py             # Chat completions route for the AI Analyst
  core/
    forecasting/
      preprocessor.py    # Dataset-agnostic validation, cleaning, and lag/rolling feature engineering
      models.py          # Forecasting wrappers (Ridge Regression, Gradient Boosting, Prophet)
      registry.py        # Model serialization, best-model logic, and recommendation reasons
      synthetic_data.py  # Realistic 2-year daily retail demand generator
    agent/
      agent.py           # Request classification, tool execution, and LLM text formatting
      tools.py           # Structured tools (sales rates, stockouts, ratios)
      llm.py             # Unified API client for Groq and local Ollama
  scratch/               # Diagnostic verification scripts (pipeline, training, toolbelt)
frontend/
  index.html             # Single Page Application HTML structure
  styles.css             # Dark-slate theme (glassmorphic cards, custom scrollbars, chat layout)
  app.js                 # View router, file uploads, Plotly.js chart renderer, and chat coordinator
```

---

## 🚀 Setup & Installation Instructions

Follow these steps to set up the virtual environment and run the project locally on Windows:

### 1. Initialize Virtual Environment & Dependencies

Open PowerShell in the project directory (`D:\Project_C`) and run:

```powershell
# 1. Create the virtual environment
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install all requirements (using a fast mirror is recommended)
pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Configure Environment Variables

Create a file named `.env` in the root directory (`D:\Project_C\.env`) with the following contents:

```env
# Provider: 'groq' (free tier cloud API) or 'ollama' (fully local offline server)
LLM_PROVIDER=groq

# Groq API Details (Get your free key at console.groq.com)
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama Local Details (Ensure Ollama is running locally if LLM_PROVIDER=ollama)
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3
```

### 3. Run Pipeline Diagnostics (Optional)

To verify the code and math are functional, run the verification scripts inside the active virtual environment:

```powershell
# Verify preprocessor and synthetic data generator
python backend/scratch/verify_pipeline.py

# Verify models training, evaluations, and registry recommendation
python backend/scratch/verify_training.py

# Verify the AI Analyst's database toolbelt metrics
python backend/scratch/verify_agent.py
```

### 4. Activate the Server

Start the backend server using `uvicorn`:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
👉 **[http://localhost:8000/](http://localhost:8000/)**

---

## 📊 Technical Presentation & Viva Guide

If asked by examiners to explain components during your BCA presentation, refer to these principles:
* **Time Series Splits**: Shuffling data randomly leaks future information to past metrics. We use a chronological cut-off (training on early dates, evaluating on the final 30 days) to match real forecasting.
* **Recursive Multi-Step Predictions**: For the ML models (Ridge, Gradient Boosting), we do not hardcode future lags. We forecast day $T+1$, append the prediction to the sales buffer, and use it as a feature to predict day $T+2$.
* **Confidence Intervals for ML**: Standard deviation of training residuals is calculated ($\sigma_e$). 95% confidence bands are computed as $\hat{y} \pm 1.96 \cdot \sigma_e$.
* **Security & Determinism**: The AI Analyst does not execute raw python/pandas code, avoiding prompt injections. The agent classifies the intent, triggers safe pre-compiled database tools, and translates the JSON results back to the user.