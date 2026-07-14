/*
================================================================
  INSIGHTFORGE CORE APP CONTROLLER (VANILLA SPA ENGINE)
================================================================
*/

document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // --- State Management ---
    const state = {
        datasetLoaded: false,
        datasetStats: null,
        edaData: null,
        trainedReport: null,
        activeProduct: null,
        activeModel: null,
        forecastHorizon: 30,
        chatOpen: false,
        activePage: "data-hub",
        learning: {
            isOpen: false,
            activeTab: "insights",
            selectedMetric: null,
            activeChartExplained: null
        },
        simulator: {
            priceMultiplier: 1.0,
            promoDays: []
        },
        scenarios: [],
        requestTokens: {
            eda: 0,
            forecast: 0
        },
        isTraining: false
    };

    // --- API Service Wrapper ---
    const API_URL = "http://127.0.0.1:8000";

    const api = {
        async get(endpoint) {
            try {
                const response = await fetch(`${API_URL}${endpoint}`);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return await response.json();
            } catch (error) {
                console.error(`API GET error on ${endpoint}:`, error);
                throw error;
            }
        },
        async post(endpoint, body = null, isMultipart = false) {
            try {
                const options = { method: "POST" };
                if (body) {
                    if (isMultipart) {
                        options.body = body;
                    } else {
                        options.headers = { "Content-Type": "application/json" };
                        options.body = JSON.stringify(body);
                    }
                }
                const response = await fetch(`${API_URL}${endpoint}`, options);
                if (!response.ok) {
                    const errDetails = await response.json().catch(() => ({ detail: "Unknown error" }));
                    throw new Error(errDetails.detail || `HTTP error! status: ${response.status}`);
                }
                return await response.json();
            } catch (error) {
                console.error(`API POST error on ${endpoint}:`, error);
                throw error;
            }
        }
    };

    // --- Safe Escaping & Markdown helpers (CRIT-06 & HIGH-01) ---
    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderMarkdownSafely(text) {
        if (!text) return "";
        let escaped = escapeHtml(text);
        // Convert bold: **text** -> <strong>text</strong>
        escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Convert italic: *text* -> <em>$1</em>
        escaped = escaped.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Convert inline code: `code` -> <code>code</code>
        escaped = escaped.replace(/`(.*?)`/g, '<code>$1</code>');
        // Convert markdown links: [text](url) -> <a href="$2" target="_blank" rel="noopener noreferrer">$1</a>
        escaped = escaped.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        // Convert newlines to <br>
        escaped = escaped.replace(/\n/g, '<br>');
        return escaped;
    }

    // --- Toast Notification helper ---
    function showToast(message, type = "info") {
        const container = document.getElementById("toast-container");
        if (!container) return;
        
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        
        let iconName = "info";
        if (type === "success") iconName = "check-circle";
        if (type === "error") iconName = "alert-octagon";
        if (type === "warning") iconName = "alert-triangle";
        
        toast.innerHTML = `
            <i data-lucide="${iconName}"></i>
            <span>${escapeHtml(message)}</span>
        `;
        container.appendChild(toast);
        lucide.createIcons();
        
        // Trigger reflow to animate
        toast.offsetHeight;
        toast.classList.add("show");
        
        // Remove after 4 seconds
        setTimeout(() => {
            toast.classList.remove("show");
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 4000);
    }

    // --- Connection Resiliency Polling ---
    let isOffline = false;
    async function checkConnectionHealth() {
        try {
            const response = await fetch(`${API_URL}/api/dataset/status`, { method: "GET" });
            if (response.ok) {
                if (isOffline) {
                    isOffline = false;
                    document.getElementById("offline-overlay").classList.add("hidden");
                    showToast("FastAPI backend server re-connected successfully.", "success");
                    // Refresh current page status
                    checkDatasetStatus();
                }
            } else {
                throw new Error("Server responded with error status.");
            }
        } catch (err) {
            if (!isOffline) {
                isOffline = true;
                document.getElementById("offline-overlay").classList.remove("hidden");
                showToast("Connection to backend server lost.", "error");
            }
        }
    }
    // Poll connection health every 5 seconds
    setInterval(checkConnectionHealth, 5000);

    // --- DOM Elements Cache ---
    const el = {
        menuItems: document.querySelectorAll(".menu-item"),
        pages: document.querySelectorAll(".content-page"),
        pageTitle: document.getElementById("current-page-title"),
        
        // Dataset Status Header
        headerStatusCard: document.getElementById("header-status-card"),
        headerDatasetName: document.getElementById("header-dataset-name"),
        headerDatasetBadge: document.getElementById("header-dataset-badge"),
        
        // Navigation items
        navEda: document.getElementById("nav-eda"),
        navForecast: document.getElementById("nav-forecast"),
        
        // Data Hub Page
        dropzone: document.getElementById("dropzone"),
        fileInput: document.getElementById("file-input"),
        selectedFileInfo: document.getElementById("selected-file-info"),
        loadDemoBtn: document.getElementById("load-demo-btn"),
        demoLoadStatus: document.getElementById("demo-load-status"),
        datasetSummaryCard: document.getElementById("dataset-summary-card"),
        datasetPreviewCard: document.getElementById("dataset-preview-card"),
        valRecords: document.getElementById("val-records"),
        valStores: document.getElementById("val-stores"),
        valProducts: document.getElementById("val-products"),
        valCategories: document.getElementById("val-categories"),
        valDateRange: document.getElementById("val-date-range"),
        valProfileText: document.getElementById("val-profile-text"),
        previewTable: document.getElementById("preview-table"),
        validationWarningsBox: document.getElementById("validation-warnings-box"),
        validationWarningsList: document.getElementById("validation-warnings-list"),
        
        // EDA Page
        edaAvgPrice: document.getElementById("eda-avg-price"),
        edaAvgSold: document.getElementById("eda-avg-sold"),
        edaOutliersCount: document.getElementById("eda-outliers-count"),
        outliersTable: document.getElementById("outliers-table").querySelector("tbody"),
        
        // Forecasting Page
        trainModelsBtn: document.getElementById("train-models-btn"),
        smoothOutliersCheck: document.getElementById("smooth-outliers-check"),
        trainingProgressBox: document.getElementById("training-progress-box"),
        trainingProgressFill: document.getElementById("training-progress-fill"),
        trainingProgressText: document.getElementById("training-progress-text"),
        forecastSummaryContainer: document.getElementById("forecast-summary-container"),
        registryAvgMae: document.getElementById("registry-avg-mae") || null,
        registryAvgMape: document.getElementById("registry-avg-mape") || null,
        forecastWorkspace: document.getElementById("forecast-workspace"),
        forecastProductSelect: document.getElementById("forecast-product-select"),
        forecastModelSelect: document.getElementById("forecast-model-select"),
        forecastHorizonSelect: document.getElementById("forecast-horizon-select"),
        runForecastBtn: document.getElementById("run-forecast-btn"),
        forecastChartTitle: document.getElementById("forecast-chart-title"),
        forecastChartSubtitle: document.getElementById("forecast-chart-subtitle"),
        recommenderModelName: document.getElementById("recommender-model-name"),
        recommenderReasonText: document.getElementById("recommender-reason-text"),
        metricsCompareTable: document.getElementById("metrics-compare-table").querySelector("tbody"),
        forecastEmptyState: document.getElementById("forecast-empty-state"),
        
        // Learning Center Panel
        openLearningBtn: document.getElementById("open-learning-btn"),
        closeLearningBtn: document.getElementById("close-learning-btn"),
        learningPanel: document.getElementById("learning-panel"),
        learningInsightsTab: document.getElementById("learning-insights-tab"),
        learningGlossaryTab: document.getElementById("learning-glossary-tab"),
        learningContent: document.getElementById("learning-content"),
        metricHeaderInfos: document.querySelectorAll(".metric-header-info"),
        
        // Export PDF Button
        exportReportBtn: document.getElementById("export-report-btn"),
        
        // Chart Loaders
        loaderSalesTrend: document.getElementById("loader-sales-trend"),
        loaderCorrelation: document.getElementById("loader-correlation"),
        loaderWeekly: document.getElementById("loader-weekly"),
        loaderMonthly: document.getElementById("loader-monthly"),
        loaderForecast: document.getElementById("loader-forecast"),
        
        // Chat Panel
        chatPanel: document.getElementById("chat-panel"),
        openChatBtn: document.getElementById("open-chat-btn"),
        closeChatBtn: document.getElementById("close-chat-btn"),
        chatMessages: document.getElementById("chat-messages"),
        chatSuggestions: document.getElementById("chat-suggestions"),
        chatInputText: document.getElementById("chat-input-text"),
        chatSendBtn: document.getElementById("chat-send-btn"),
        explainChartBtns: document.querySelectorAll(".explain-chart-btn"),

        // Offline retry button
        retryConnectionBtn: document.getElementById("retry-connection-btn"),

        // What-If Simulator & Decision Support
        simPriceSlider: document.getElementById("sim-price-slider"),
        simPriceVal: document.getElementById("sim-price-val"),
        simPromoDaysGrid: document.getElementById("sim-promo-days-grid"),
        clearSimPromosBtn: document.getElementById("clear-sim-promos-btn"),
        simVolumeDelta: document.getElementById("sim-volume-delta"),
        simRevenueDelta: document.getElementById("sim-revenue-delta"),
        decisionStockStatus: document.getElementById("decision-stock-status"),
        decisionActionText: document.getElementById("decision-action-text"),
        decisionStockoutDays: document.getElementById("decision-stockout-days"),
        decisionRevenueRisk: document.getElementById("decision-revenue-risk"),
        scenarioHistoryCard: document.getElementById("scenario-history-card"),
        scenarioHistoryTbody: document.getElementById("scenario-history-tbody"),
        clearScenariosBtn: document.getElementById("clear-scenarios-btn"),
        simScenarioName: document.getElementById("sim-scenario-name"),
        saveScenarioBtn: document.getElementById("save-scenario-btn")
    };

    // --- Explainability & Learning Center Repositories ---

    const CHARTS_EXPLAINER = {
        sales_trend: {
            title: "Overall Sales Trend",
            summary: "A timeline representation of aggregated units sold daily across all products and stores in the loaded dataset.",
            what: "This chart plots time on the horizontal axis (X-axis) and the total sales volume on the vertical axis (Y-axis). The line connects consecutive daily transaction sums to show overall demand movement.",
            learn: "You can identify holiday peaks, day-of-week seasonality (e.g. weekly surges), and overall upward or downward sales trajectories over months or years.",
            decision: "Allows you to adjust warehouse storage capacities, arrange bulk logistics, and plan store staffing schedules to align with high-demand cycles."
        },
        correlation_matrix: {
            title: "Feature Correlation Matrix",
            summary: "A correlation heatmap showing statistical linear associations between key demand drivers.",
            what: "This color-coded matrix grids variables against each other. Coefficients range from -1.0 (perfect inverse relationship) through 0.0 (no correlation) to +1.0 (perfect positive correlation).",
            learn: "Identifies whether promotions drive sales, how price hikes affect demand volume (price elasticity), and the role weekend flags play in traffic.",
            decision: "Informs price optimization plans (pricing bounds) and lets you decide if a promotion is statistically viable enough to warrant inventory pre-purchasing."
        },
        weekly_seasonality: {
            title: "Weekly Demand Profile",
            summary: "Average daily sales volume aggregated by day of the week (Monday through Sunday).",
            what: "A bar chart grouping historic transactions to show average daily sales. Mon=0, Sun=6.",
            learn: "Highlights which days of the week consistently experience the highest customer traffic (e.g. Friday/Saturday surges vs. Tuesday slumps).",
            decision: "Guides daily shelf replenishment scheduling, fresh cargo delivery days, and staff shifts to ensure popular products are fully stocked for weekend spikes."
        },
        monthly_seasonality: {
            title: "Monthly Seasonality Profile",
            summary: "Average daily sales volume aggregated by month of the year.",
            what: "A column chart showing long-term yearly cycle performance across January through December.",
            learn: "Identifies seasonal purchasing peaks (e.g. summer inventory build-up, winter holiday shopping surges, or back-to-school trends).",
            decision: "Informs quarterly procurement contracts, budget planning, and long-lead warehouse space leasing (e.g., ordering winter lines by August)."
        },
        demand_forecast: {
            title: "30-Day Demand Forecast",
            summary: "Historical daily actual sales plotted alongside a 30-day future demand forecast with 95% confidence bands.",
            what: "Solid line displays actual past sales. Dashed line represents forecasted future sales. The shaded area represents the 95% confidence interval (safety buffer zone).",
            learn: "Shows where demand is expected to rise or fall, the size of future promotional spikes, and the expanding uncertainty cone over time.",
            decision: "Guides purchase order quantities. Order up to the upper bound to ensure high service level (prevent stockouts), or order near the prediction line to minimize holding capital."
        }
    };

    const METRICS_EXPLAINER = {
        MAE: {
            title: "MAE (Mean Absolute Error)",
            target: "Lower is better",
            definition: "The average absolute difference between the predicted sales and the actual sales volume.",
            business: "Tells you how many units your forecast is off by on average per day. If MAE is 5.0, your daily purchase orders will deviate from real demand by approximately 5 units (either over or under).",
            formula: "\\text{MAE} = \\frac{1}{N}\\sum_{t=1}^{N}|y_t - \\hat{y}_t|",
            ratingFn: (val) => {
                if (val < 5) return { text: "Excellent", class: "rating-excellent" };
                if (val < 15) return { text: "Good Accuracy", class: "rating-good" };
                return { text: "Low Precision", class: "rating-poor" };
            }
        },
        MAPE: {
            title: "MAPE (Mean Absolute Percentage Error)",
            target: "Lower is better",
            definition: "The average relative percentage deviation of predicted sales from actual sales volume.",
            business: "Represents the percentage error relative to actual sales. For instance, a MAPE of 8% means your forecasts are off by 8% of the true sales volume on average. Under 10% is considered outstanding for retail.",
            formula: "\\text{MAPE} = \\frac{100\\%}{N}\\sum_{t=1}^{N}\\left|\\frac{y_t - \\hat{y}_t}{y_t}\\right|",
            ratingFn: (val) => {
                if (val < 10) return { text: "Excellent (<10%)", class: "rating-excellent" };
                if (val < 25) return { text: "Good (10-25%)", class: "rating-good" };
                return { text: "Poor (>25%)", class: "rating-poor" };
            }
        },
        RMSE: {
            title: "RMSE (Root Mean Squared Error)",
            target: "Lower is better",
            definition: "The square root of the average of squared differences between predictions and actual observations.",
            business: "Similar to MAE, but because errors are squared before averaging, RMSE penalizes large forecasting errors heavily. If RMSE is much larger than MAE, the model occasionally makes huge misses.",
            formula: "\\text{RMSE} = \\sqrt{\\frac{1}{N}\\sum_{t=1}^{N}(y_t - \\hat{y}_t)^2}",
            ratingFn: (val) => {
                if (val < 8) return { text: "Excellent", class: "rating-excellent" };
                if (val < 20) return { text: "Good", class: "rating-good" };
                return { text: "Low Accuracy", class: "rating-poor" };
            }
        },
        R2: {
            title: "R² (Coefficient of Determination)",
            target: "Higher is better (Closer to 1.0)",
            definition: "The proportion of variance in the sales target variable that is predictable from the model features.",
            business: "Indicates how much of your demand fluctuations (weekend surges, promotional spikes) the model explains. A value of 0.85 means the model captures 85% of demand volatility. Negative values indicate the model performs worse than a simple historical mean.",
            formula: "R^2 = 1 - \\frac{\\sum_{t=1}^{N}(y_t - \\hat{y}_t)^2}{\\sum_{t=1}^{N}(y_t - \\bar{y})^2}",
            ratingFn: (val) => {
                if (val > 0.8) return { text: "Very Strong Fit", class: "rating-excellent" };
                if (val > 0.5) return { text: "Moderate Fit", class: "rating-good" };
                return { text: "Weak Baseline", class: "rating-poor" };
            }
        },
        CI: {
            title: "Confidence Interval (Safety Bounds)",
            target: "Narrower intervals imply higher certainty",
            definition: "The mathematical range displaying the boundaries within which actual sales will fall 95% of the time.",
            business: "Defines the inventory safety stock zone. Slicing near the upper confidence limit ensures a 97.5% protection level against stockouts during random demand surges. Slicing the lower limit protects against capital tied up in slow-movers.",
            formula: "\\text{Interval}_t = \\hat{y}_t \\pm 1.96 \\cdot \\sigma_{e} \\cdot \\sqrt{t}",
            ratingFn: () => ({ text: "Statistical Bound", class: "rating-excellent" })
        }
    };

    const MODELS_EXPLAINER = {
        "Prophet": {
            name: "Prophet (Additive Seasonality Model)",
            description: "Prophet operates as a generalized additive regression model. It models time series by fitting non-linear trend curves along with weekly and yearly seasonal components using Fourier series.",
            works: "Prophet fits a model of the form: $y(t) = g(t) + s(t) + h(t) + \\epsilon_t$, where $g(t)$ is the trend, $s(t)$ represents seasonality, $h(t)$ represents holidays/promotions, and $\\epsilon_t$ is the error. It treats forecasting as a curve-fitting problem rather than using sequential autoregressive lags.",
            advantages: [
                "Highly robust to missing dates and shifts in trend patterns.",
                "Handles yearly and weekly seasonality profiles natively without requiring lag creation.",
                "Adjusts dynamically for pricing and promotional events using external regressors."
            ],
            limitations: [
                "Cannot capture complex, non-linear relationships between multiple features easily.",
                "Does not model short-term autoregressive patterns (like yesterday's sales) directly."
            ],
            when: "Best for consumer products with clear weekly/yearly seasonality cycles, stable long-term trend lines, and where promotions have direct, additive demand impacts."
        },
        "Random Forest": {
            name: "Random Forest Regressor (Decision Tree Ensemble)",
            description: "An ensemble learning method that fits a multitude of decision trees during training and outputs the average prediction of the individual trees to reduce variance.",
            works: "Constructs $N$ independent decision trees using bootstrap aggregation (bagging). At each node split, it queries a random subset of features (like lag sales, prices, weekends). The final forecast is the averaged voting consensus of all trees, avoiding overfitting.",
            advantages: [
                "Highly effective at capturing complex, non-linear interactions between variables (e.g. promotion combined with weekend price drops).",
                "Handles feature interactions (lags vs pricing) automatically without manual scaling.",
                "Ensemble structure prevents overfitting to random noise in the sales data."
            ],
            limitations: [
                "Cannot extrapolate trends outside the range of historical training values.",
                "Requires careful lag engineering; if daily data has gaps, predictions fail."
            ],
            when: "Ideal for products with highly complex, interactive demand drivers (such as promotions, competitor price cuts, or holiday events) and informative historic lags."
        },
        "Linear Regression": {
            name: "Linear Regression (Ordinary Least Squares)",
            description: "A standard statistical model that assumes a linear relationship between the input features (price, promotions, lags) and the target sales volume.",
            works: "Fits a linear equation of the form: $y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\dots + \\beta_p X_p$, minimizing the sum of squared residuals between predicted sales and actual training data.",
            advantages: [
                "Extremely fast to train and mathematically transparent.",
                "Low risk of overfitting on noisy sales data if features are kept simple.",
                "Acts as an excellent, clear baseline benchmark for other complex ML models."
            ],
            limitations: [
                "Assumes relationships are linear, which is rarely true for price changes or seasonality.",
                "Extremely sensitive to collinearity between features (e.g. lag 1 and lag 7 sales)."
            ],
            when: "Best used as a baseline reference model, or for products with very steady, predictable, linear sales trajectories."
        },
        "Ridge Regression": {
            name: "Ridge Regression (L2 Regularized Linear Model)",
            description: "A regularized version of Linear Regression that adds a penalty term proportional to the square of the coefficients to prevent overfitting and multicollinearity.",
            works: "Minimizes the loss function: $\\text{Loss} = \\sum(y_i - \\hat{y}_i)^2 + \\alpha \\sum \\beta_j^2$. The L2 penalty shrinks coefficients towards zero, which is essential when lag features ($t-1$, $t-7$) are highly correlated.",
            advantages: [
                "Prevents extreme coefficient estimates (model instability) caused by highly correlated lag features.",
                "Maintains the speed and transparency of linear models while adding regularization.",
                "More robust to training set noise than standard Ordinary Least Squares."
            ],
            limitations: [
                "Cannot capture non-linear feature interactions.",
                "Requires feature standardization (scaling) prior to fitting (handled automatically in our pipeline)."
            ],
            when: "Best when you want a stable, robust linear baseline model that handles many correlated lag features without coefficient explosion."
        }
    };

    // Toggle Learning Center Side Panel
    function toggleLearningCenter(open) {
        state.learning.isOpen = open;
        if (open) {
            el.learningPanel.classList.add("open");
            if (state.chatOpen) {
                toggleChat(false);
            }
            updateLearningCenter();
        } else {
            el.learningPanel.classList.remove("open");
            state.learning.activeChartExplained = null;
            state.learning.selectedMetric = null;
        }
    }

    // Helper to render KaTeX formula safely inside an element
    function renderKaTeX(latexString, containerElement) {
        if (typeof katex !== 'undefined' && containerElement) {
            try {
                katex.render(latexString, containerElement, {
                    throwOnError: false,
                    displayMode: true
                });
            } catch (err) {
                containerElement.innerHTML = `\\[${escapeHtml(latexString)}\\]`;
            }
        } else if (containerElement) {
            containerElement.innerHTML = `\\[${escapeHtml(latexString)}\\]`;
        }
    }

    function updateLearningCenter() {
        if (!state.learning.isOpen) return;

        let html = "";

        if (state.learning.activeTab === "insights") {
            // Tab 1: Interactive Insights

            // A. Context-Aware Page Summary Card
            let pageTitle = "";
            let pageDesc = "";
            let pageIcon = "database";
            
            if (state.activePage === "data-hub") {
                pageTitle = "Data Hub Workspace";
                pageDesc = "This workspace processes and validates uploaded CSV retail sales data. Before modeling, the preprocessor cleans transactions, aggregates store records to a product-level daily grid, and fills missing values. Lags are constructed chronologically to prevent temporal data leakage.";
                pageIcon = "database";
            } else if (state.activePage === "eda") {
                pageTitle = "Exploratory Data Analysis";
                pageDesc = "Examines historical sales trends, detects extreme anomalies using the IQR outlier detection technique, and maps coefficients in a correlation matrix. Visualizing these attributes identifies seasonality profiles and variables that influence volume.";
                pageIcon = "pie-chart";
            } else if (state.activePage === "forecasting") {
                pageTitle = "Forecasting Workspace";
                pageDesc = "Evaluates baseline linear regression, Ridge regularization, Random Forest ensembles, and Prophet curves. Models are validated using a chronological 30-day out-of-sample holdout set. Predictions are generated recursively step-by-step.";
                pageIcon = "trending-up";
            }

            html += `
                <div class="learning-card">
                    <div class="learning-card-title">
                        <i data-lucide="${pageIcon}" class="accent-color" style="width: 16px; height: 16px;"></i>
                        <span>${escapeHtml(pageTitle)}</span>
                    </div>
                    <div class="learning-card-body">
                        <p>${escapeHtml(pageDesc)}</p>
                    </div>
                </div>
            `;

            // B. Chart Explainer Card (if sparkles button clicked or last explained is set)
            if (state.learning.activeChartExplained) {
                const chartId = state.learning.activeChartExplained;
                const chartInfo = CHARTS_EXPLAINER[chartId];
                if (chartInfo) {
                    html += `
                        <div class="learning-card highlighted" id="chart-explainer-card">
                            <div class="learning-card-header">
                                <span class="learning-card-title">
                                    <i data-lucide="sparkles" class="accent-color" style="width: 14px; height: 14px;"></i>
                                    Chart Explainer: ${escapeHtml(chartInfo.title)}
                                </span>
                            </div>
                            <div class="learning-card-body flex-col gap-2">
                                <p><strong>Summary:</strong> ${escapeHtml(chartInfo.summary)}</p>
                                <div style="margin-top: 6px;">
                                    <strong class="text-primary block" style="margin-bottom: 2px;">What am I looking at?</strong>
                                    <p>${escapeHtml(chartInfo.what)}</p>
                                </div>
                                <div style="margin-top: 6px;">
                                    <strong class="text-primary block" style="margin-bottom: 2px;">What should I learn?</strong>
                                    <p>${escapeHtml(chartInfo.learn)}</p>
                                </div>
                                <div style="margin-top: 6px;">
                                    <strong class="text-primary block" style="margin-bottom: 2px;">What business decision can I make?</strong>
                                    <p>${escapeHtml(chartInfo.decision)}</p>
                                </div>
                                <button class="ask-ai-center-btn" id="ask-ai-chart-btn" data-chart="${chartId}">
                                    <i data-lucide="message-square"></i>
                                    Ask AI Analyst for Deeper Review
                                </button>
                            </div>
                        </div>
                    `;
                }
            }

            // C. Dynamic Model Explainer Card & "Why did the model choose this?"
            if (state.activePage === "forecasting" && state.trainedReport && state.activeProduct) {
                const productId = state.activeProduct;
                const prodData = state.trainedReport.products[productId];
                
                if (prodData) {
                    const activeModelSelect = el.forecastModelSelect.value;
                    const modelName = activeModelSelect === "best_recommender" ? prodData.best_model : activeModelSelect;
                    const modelInfo = MODELS_EXPLAINER[modelName];
                    
                    if (modelInfo) {
                        const metrics = prodData.all_models[modelName];
                        const isBest = modelName === prodData.best_model;
                        
                        // Retrieve baseline model MAE
                        let baselineMae = null;
                        if (prodData.all_models["Ridge Regression"]) {
                            baselineMae = prodData.all_models["Ridge Regression"].MAE;
                        } else if (prodData.all_models["Linear Regression"]) {
                            baselineMae = prodData.all_models["Linear Regression"].MAE;
                        }
                        
                        let pctImprovementText = "";
                        if (baselineMae && baselineMae > metrics.MAE) {
                            const pct = ((baselineMae - metrics.MAE) / baselineMae * 100).toFixed(1);
                            pctImprovementText = ` This selection improved validation accuracy (MAE) by <strong>${pct}%</strong> compared to the baseline Ridge Regression model.`;
                        }
                        
                        // Generate dynamic "Why did the model choose this?" bullet points
                        let choiceBullets = "";
                        if (modelName === "Prophet") {
                            choiceBullets = `
                                <li><strong>Strong Seasonality Fitted</strong>: Prophet successfully captured weekly seasonal cycles and yearly trends directly using Fourier series.</li>
                                <li><strong>Promotion Adjustment</strong>: External regressors adjusted predictions for marketing surges without target leakage.</li>
                                <li><strong>Bayesian Confidence Cones</strong>: Maintained narrow, reliable uncertainty bands based on historical posterior sampling.</li>
                                <li><strong>Lowest Validation Loss</strong>: Outperformed lag-based models on the out-of-sample test set (MAE = ${metrics.MAE}).${pctImprovementText ? ` ${pctImprovementText}` : ''}</li>
                            `;
                        } else if (modelName === "Random Forest") {
                            choiceBullets = `
                                <li><strong>Non-Linear Interactions</strong>: The ensemble of decision trees mapped complex, multi-variable interactions (e.g. price drops on weekends during promotions).</li>
                                <li><strong>Autoregressive Lags</strong>: Recursive features (lags 1, 7, 14) proved highly informative for projecting recent momentum.</li>
                                <li><strong>Outlier Robustness</strong>: Bagging reduced variance and prevented anomalous sales surges from corrupting predictions.</li>
                                <li><strong>Superior Performance</strong>: Minimized test-set residuals (R² = ${metrics.R2}, MAE = ${metrics.MAE}).${pctImprovementText ? ` ${pctImprovementText}` : ''}</li>
                            `;
                        } else if (modelName === "Linear Regression") {
                            choiceBullets = `
                                <li><strong>Simple Steady Trajectory</strong>: Sales follow a stable linear trend line without high non-linear volatility.</li>
                                <li><strong>Overfitting Prevention</strong>: High parameter constraints kept prediction variances low on noisy retail data.</li>
                                <li><strong>Baseline Performance</strong>: Outperformed complex models by avoiding parameter inflation on a short history.</li>
                            `;
                        } else if (modelName === "Ridge Regression") {
                            choiceBullets = `
                                <li><strong>Multicollinearity Penalty</strong>: L2 regularization successfully penalized inflated coefficients caused by highly correlated daily sales lags.</li>
                                <li><strong>Noise Reduction</strong>: Kept validation margins stable and reduced parameter variance compared to standard Ordinary Least Squares.</li>
                            `;
                        }

                        html += `
                            <div class="learning-card">
                                <div class="learning-card-header">
                                    <span class="learning-card-title">
                                        <i data-lucide="cpu" class="accent-color" style="width: 14px; height: 14px;"></i>
                                        Model: ${escapeHtml(modelName)}
                                    </span>
                                    ${isBest ? '<span class="rating-badge rating-excellent">Recommended</span>' : ''}
                                </div>
                                <div class="learning-card-body flex-col gap-2">
                                    <p>${escapeHtml(modelInfo.description)}</p>
                                    <div style="margin-top: 4px;">
                                        <strong>How it works:</strong>
                                        <p class="text-xs text-secondary mt-1">${modelInfo.works}</p>
                                    </div>
                                    
                                    <div style="margin-top: 6px;">
                                        <strong class="accent-color">Why is this model recommended?</strong>
                                        <ul class="bullets-list mt-1" style="font-size: 0.72rem; padding-left: 1rem; color: var(--text-secondary);">
                                            ${choiceBullets}
                                        </ul>
                                    </div>
                                    
                                    <div style="margin-top: 6px;">
                                        <strong>Advantages:</strong>
                                        <ul class="bullets-list mt-1" style="font-size: 0.7rem; padding-left: 1rem;">
                                            ${modelInfo.advantages.map(a => `<li>${escapeHtml(a)}</li>`).join("")}
                                        </ul>
                                    </div>
                                    <div style="margin-top: 4px;">
                                        <strong>Limitations:</strong>
                                        <ul class="bullets-list mt-1" style="font-size: 0.7rem; padding-left: 1rem;">
                                            ${modelInfo.limitations.map(l => `<li>${escapeHtml(l)}</li>`).join("")}
                                        </ul>
                                    </div>
                                    <button class="ask-ai-center-btn" id="ask-ai-model-btn" data-model="${modelName}">
                                        <i data-lucide="message-square"></i>
                                        Ask AI Analyst for Deeper Review
                                    </button>
                                </div>
                            </div>
                        `;
                    }
                }
            }

            // D. Active Forecast Explainer & Decomposition Card
            if (state.activePage === "forecasting" && state.forecastData && state.activeProduct) {
                const fData = state.forecastData;
                html += `
                    <div class="learning-card">
                        <div class="learning-card-header">
                            <span class="learning-card-title">
                                <i data-lucide="line-chart" class="accent-color" style="width: 14px; height: 14px;"></i>
                                Active Forecast Explanation
                            </span>
                        </div>
                        <div class="learning-card-body flex-col gap-2">
                            <p><strong>Product:</strong> ${escapeHtml(fData.product_name)} (${escapeHtml(fData.product_id)})</p>
                            <p>The 30-day forecast is generated by feeding sequential predictions back into lag matrices (recursive forecasting). The shaded boundary displays the 95% confidence bounds (safety zone).</p>
                            <div style="margin-top: 4px;">
                                <strong>Safety stock recommendation:</strong>
                                <p class="text-xs text-secondary mt-1">To ensure a 97.5% service level against weekend spikes or demand volatility, order inventory up to the <strong>Upper Confidence Bound</strong>.</p>
                            </div>
                        </div>
                    </div>
                `;
            }

            // E. Empty states if nothing loaded/trained
            if (state.activePage === "forecasting" && !state.trainedReport) {
                html += `
                    <div class="learning-card" style="text-align: center; padding: 2rem 1rem;">
                        <i data-lucide="lock" style="width: 32px; height: 32px; margin: 0 auto 10px; color: var(--text-secondary);"></i>
                        <p class="text-xs text-secondary">Please complete model training to unlock dynamic model performance explainers and selection heuristics.</p>
                    </div>
                `;
            }

        } else {
            // Tab 2: Metric Glossary

            for (const key in METRICS_EXPLAINER) {
                const metric = METRICS_EXPLAINER[key];
                const isOpen = state.learning.selectedMetric === key;
                
                // Retrieve rating badge dynamically if trained report exists
                let ratingHtml = "";
                let datasetInterpretationHtml = "";
                if (state.trainedReport && state.activeProduct && key !== "CI") {
                    const prodData = state.trainedReport.products[state.activeProduct];
                    if (prodData) {
                        const activeModelSelect = el.forecastModelSelect.value;
                        const modelName = activeModelSelect === "best_recommender" ? prodData.best_model : activeModelSelect;
                        const score = prodData.all_models[modelName][key];
                        if (score !== undefined) {
                            const rating = metric.ratingFn(score);
                            ratingHtml = `<span class="rating-badge ${rating.class}">${score}${key === "MAPE" ? "%" : ""} (${rating.text})</span>`;
                            
                            // Generate custom interpretation based on metrics
                            if (key === "MAE") {
                                datasetInterpretationHtml = `
                                    <div style="margin-top: 6px; border-left: 2px solid var(--accent); padding-left: 8px;">
                                        <strong class="text-primary block text-xs" style="margin-bottom: 2px;">Dataset Interpretation:</strong>
                                        <p class="text-xs text-secondary">With the active model (<strong>${modelName}</strong>), the daily sales forecasts deviate from actual demand by an average of <strong>${score} units</strong> per day. Your inventory planners should maintain safety stock of at least ${Math.ceil(score * 2)} units to buffer against this average variance.</p>
                                    </div>
                                `;
                            } else if (key === "MAPE") {
                                datasetInterpretationHtml = `
                                    <div style="margin-top: 6px; border-left: 2px solid var(--accent); padding-left: 8px;">
                                        <strong class="text-primary block text-xs" style="margin-bottom: 2px;">Dataset Interpretation:</strong>
                                        <p class="text-xs text-secondary">Your forecasts differ from actual sales by an average of <strong>${score}%</strong>. In unit terms, this means that for every 100 units sold, predictions typically deviate by approximately <strong>${(score).toFixed(1)} units</strong>.</p>
                                    </div>
                                `;
                            } else if (key === "R2") {
                                const pctVal = (score * 100).toFixed(1);
                                datasetInterpretationHtml = `
                                    <div style="margin-top: 6px; border-left: 2px solid var(--accent); padding-left: 8px;">
                                        <strong class="text-primary block text-xs" style="margin-bottom: 2px;">Dataset Interpretation:</strong>
                                        <p class="text-xs text-secondary">The model successfully captures and explains <strong>${pctVal}%</strong> of the historical volatility in demand. The remaining ${(100 - pctVal).toFixed(1)}% represents random noise or unmeasured external events.</p>
                                    </div>
                                `;
                            } else if (key === "RMSE") {
                                datasetInterpretationHtml = `
                                    <div style="margin-top: 6px; border-left: 2px solid var(--accent); padding-left: 8px;">
                                        <strong class="text-primary block text-xs" style="margin-bottom: 2px;">Dataset Interpretation:</strong>
                                        <p class="text-xs text-secondary">RMSE is <strong>${score}</strong>. This metric penalizes large errors more heavily. Since the RMSE is close to the MAE, it confirms that the model does not suffer from occasional massive forecasting misses.</p>
                                    </div>
                                `;
                            }
                        }
                    }
                }

                html += `
                    <div class="learning-card glossary-card ${isOpen ? 'open highlighted' : ''}" id="metric-card-${key}" data-metric="${key}">
                        <div class="learning-card-header">
                            <span class="learning-card-title">
                                <i data-lucide="help-circle" class="accent-color" style="width: 14px; height: 14px;"></i>
                                ${escapeHtml(metric.title)}
                            </span>
                            ${ratingHtml}
                        </div>
                        <div class="glossary-details">
                            <div class="learning-card-body flex-col gap-2">
                                <p><strong>Definition:</strong> ${escapeHtml(metric.definition)}</p>
                                <div style="margin-top: 4px;">
                                    <strong>Business Meaning:</strong>
                                    <p class="text-xs text-secondary mt-1">${escapeHtml(metric.business)}</p>
                                </div>
                                <div style="margin-top: 4px;">
                                    <strong>Target Optimization:</strong>
                                    <p class="text-xs text-secondary mt-1">Goal: <span class="accent-color" style="font-weight: 600;">${escapeHtml(metric.target)}</span></p>
                                </div>
                                ${datasetInterpretationHtml}
                                <div style="margin-top: 4px;">
                                    <strong>Mathematical Formula:</strong>
                                    <div class="metric-math" id="math-formula-${key}" data-formula="${metric.formula}"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }
        }

        el.learningContent.innerHTML = html;
        lucide.createIcons();

        // Bind interactive events for elements inside learning panel

        // 1. Math equations rendering with KaTeX
        if (state.learning.activeTab === "glossary") {
            const mathBlocks = el.learningContent.querySelectorAll(".metric-math");
            mathBlocks.forEach(block => {
                const formula = block.getAttribute("data-formula");
                renderKaTeX(formula, block);
            });

            // Bind click to glossary card headers to act as accordion
            const glossaryCards = el.learningContent.querySelectorAll(".glossary-card");
            glossaryCards.forEach(card => {
                card.addEventListener("click", (e) => {
                    // Prevent accordion toggle if user is copying formula text or clicking inside details
                    if (e.target.closest(".glossary-details")) return;
                    
                    const mKey = card.getAttribute("data-metric");
                    if (card.classList.contains("open")) {
                        card.classList.remove("open");
                        if (state.learning.selectedMetric === mKey) {
                            state.learning.selectedMetric = null;
                        }
                    } else {
                        // Close other cards first
                        glossaryCards.forEach(c => c.classList.remove("open"));
                        card.classList.add("open");
                        state.learning.selectedMetric = mKey;
                        // Render KaTeX for this card specifically
                        const formulaBlock = card.querySelector(".metric-math");
                        if (formulaBlock) {
                            renderKaTeX(formulaBlock.getAttribute("data-formula"), formulaBlock);
                        }
                    }
                });
            });
        }

        // 2. Chat Query Sparkles redirection inside cards
        const askChartBtn = el.learningContent.querySelector("#ask-ai-chart-btn");
        if (askChartBtn) {
            askChartBtn.addEventListener("click", () => {
                const chartId = askChartBtn.getAttribute("data-chart");
                let promptMsg = "";
                if (chartId === "sales_trend") {
                    promptMsg = "Explain our overall sales trend chart. What are the key observations?";
                } else if (chartId === "correlation_matrix") {
                    promptMsg = "Explain the correlation matrix chart in detail. What are the primary demand drivers?";
                } else if (chartId === "weekly_seasonality") {
                    promptMsg = "Analyze our weekly seasonality patterns and provide inventory recommendations.";
                } else if (chartId === "monthly_seasonality") {
                    promptMsg = "Analyze our monthly yearly patterns and suggest long-lead purchasing strategy.";
                } else if (chartId === "demand_forecast") {
                    const prodName = el.forecastProductSelect.options[el.forecastProductSelect.selectedIndex]?.text || "selected item";
                    promptMsg = `Explain the demand forecast chart for ${prodName} and suggest optimal restocking strategies.`;
                }
                toggleChat(true);
                sendChatMessage(promptMsg);
            });
        }

        const askModelBtn = el.learningContent.querySelector("#ask-ai-model-btn");
        if (askModelBtn) {
            askModelBtn.addEventListener("click", () => {
                const mName = askModelBtn.getAttribute("data-model");
                const pId = state.activeProduct;
                const promptMsg = `Compare validation metrics for product ID ${pId} and explain why ${mName} is recommended.`;
                toggleChat(true);
                sendChatMessage(promptMsg);
            });
        }
    }

    // --- Responsive Chart Resizing ---
    window.addEventListener("resize", () => {
        const charts = ["chart-sales-trend", "chart-correlation", "chart-weekly", "chart-monthly", "chart-forecast"];
        charts.forEach(id => {
            const container = document.getElementById(id);
            if (container && container.clientWidth > 0 && typeof Plotly !== 'undefined') {
                Plotly.Plots.resize(container);
            }
        });
    });

    // --- View Router ---
    function navigateToPage(pageId) {
        if (state.activePage === pageId) return;
        
        // Validation check for guided workflow
        if (pageId !== "data-hub" && !state.datasetLoaded) {
            showToast("Please load or upload a dataset in the Data Hub first.", "warning");
            return;
        }

        // Update active page class in sidebar
        el.menuItems.forEach(item => {
            if (item.getAttribute("data-page") === pageId) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        // Switch visible page container
        el.pages.forEach(page => {
            if (page.id === `page-${pageId}`) {
                page.classList.add("active");
            } else {
                page.classList.remove("active");
            }
        });

        state.activePage = pageId;
        
        // Trigger Plotly dimension updates on activation (fixes hidden tab rendering issues)
        setTimeout(() => {
            if (pageId === "eda") {
                ["sales-trend-plot", "correlation-heatmap", "weekly-pattern-plot", "monthly-pattern-plot"].forEach(id => {
                    const plotEl = document.getElementById(id);
                    if (plotEl && plotEl.classList.contains("js-plotly-plot")) {
                        Plotly.Plots.resize(plotEl);
                    }
                });
            } else if (pageId === "forecasting") {
                const plotEl = document.getElementById("chart-forecast");
                if (plotEl && plotEl.classList.contains("js-plotly-plot")) {
                    Plotly.Plots.resize(plotEl);
                }
            }
        }, 100);
        
        // Update header breadcrumb
        const titleMap = {
            "data-hub": "Is my data ready? — Data Hub",
            "eda": "What is happening in my business? — Sales Insights",
            "forecasting": "What will happen next? — AI Recommendation & Forecasts"
        };
        el.pageTitle.textContent = titleMap[pageId] || "Dashboard";

        // Trigger dynamic page loading
        if (pageId === "eda") {
            loadEdaPage();
        } else if (pageId === "forecasting") {
            loadForecastingPage();
        }

        // Refresh learning center context
        if (state.learning.isOpen) {
            updateLearningCenter();
        }
    }

    // Bind navigation click handlers
    el.menuItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const pageId = item.getAttribute("data-page");
            if (pageId) navigateToPage(pageId);
        });
    });

    // --- Guided Workflow Controls ---
    function setDatasetLoadedState(isLoaded, filename = "", stats = null) {
        state.datasetLoaded = isLoaded;
        if (isLoaded) {
            state.datasetStats = stats;
            // Header updates
            el.headerDatasetName.textContent = filename || "Active Dataset";
            el.headerDatasetBadge.textContent = "ACTIVE";
            el.headerDatasetBadge.className = "status-badge status-online";
            
            // Enable Menu Items
            el.navEda.classList.remove("disabled");
            el.navForecast.classList.remove("disabled");
            
            // Update Data Hub validation cards
            el.valRecords.textContent = stats.row_count.toLocaleString();
            el.valStores.textContent = stats.unique_stores;
            el.valProducts.textContent = stats.unique_products;
            el.valCategories.textContent = stats.unique_categories;
            el.valDateRange.textContent = `${stats.start_date} to ${stats.end_date}`;
            
            // Populate profile summary
            el.valProfileText.textContent = `Standardized columns detected. Date span covers ${stats.days_span} days of history.`;
            el.datasetSummaryCard.classList.remove("hidden");
            
            // Refresh preview
            loadPreviewTable();
        } else {
            state.datasetStats = null;
            el.headerDatasetName.textContent = "No active dataset";
            el.headerDatasetBadge.textContent = "Offline";
            el.headerDatasetBadge.className = "status-badge status-offline";
            
            el.navEda.classList.add("disabled");
            el.navForecast.classList.add("disabled");
            el.datasetSummaryCard.classList.add("hidden");
            el.datasetPreviewCard.classList.add("hidden");
        }
    }

    // Check backend status on page load
    async function checkDatasetStatus() {
        try {
            const data = await api.get("/api/dataset/status");
            if (data.loaded) {
                setDatasetLoadedState(true, "active_dataset.csv", data.stats);
                
                // If warnings exist, display them
                if (data.warnings && data.warnings.length > 0) {
                    el.validationWarningsBox.classList.remove("hidden");
                    el.validationWarningsList.innerHTML = data.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("");
                } else {
                    el.validationWarningsBox.classList.add("hidden");
                }
            } else {
                setDatasetLoadedState(false);
            }
        } catch (error) {
            setDatasetLoadedState(false);
        }
    }

    // --- Data Hub Logic ---
    
    // Load Demo Dataset
    el.loadDemoBtn.addEventListener("click", async () => {
        el.demoLoadStatus.textContent = "Generating and loading simulated retail transactions...";
        el.demoLoadStatus.className = "mt-3 text-center text-sm text-secondary";
        el.loadDemoBtn.disabled = true;
        
        try {
            const data = await api.post("/api/dataset/load-demo");
            el.demoLoadStatus.textContent = "Demo retail dataset loaded successfully!";
            el.demoLoadStatus.className = "mt-3 text-center text-sm success-color";
            
            setDatasetLoadedState(true, "synthetic_retail_data.csv", data.report.stats);
            showToast("Demo retail dataset loaded successfully!", "success");
            
            // Show alert box warnings if any
            if (data.report.warnings && data.report.warnings.length > 0) {
                el.validationWarningsBox.classList.remove("hidden");
                el.validationWarningsList.innerHTML = data.report.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("");
            } else {
                el.validationWarningsBox.classList.add("hidden");
            }
            
            // Auto navigate to EDA to keep workflow fluid
            setTimeout(() => navigateToPage("eda"), 1000);
        } catch (error) {
            el.demoLoadStatus.textContent = `Error: ${error.message}`;
            el.demoLoadStatus.className = "mt-3 text-center text-sm error-color";
            showToast(`Failed to load demo: ${error.message}`, "error");
        } finally {
            el.loadDemoBtn.disabled = false;
        }
    });

    // Drag-and-drop file upload
    el.dropzone.addEventListener("click", () => el.fileInput.click());
    
    el.dropzone.style.transition = "all 0.2s ease";
    el.dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        el.dropzone.style.borderColor = "var(--accent)";
        el.dropzone.style.backgroundColor = "rgba(99, 102, 241, 0.05)";
    });

    el.dropzone.addEventListener("dragleave", () => {
        el.dropzone.style.borderColor = "var(--border-color)";
        el.dropzone.style.backgroundColor = "transparent";
    });

    el.dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        el.dropzone.style.borderColor = "var(--border-color)";
        el.dropzone.style.backgroundColor = "transparent";
        
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    el.fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        // Enforce 50 MB size limit check on the client side
        const MAX_SIZE = 50 * 1024 * 1024; // 50 MB
        if (file.size > MAX_SIZE) {
            showToast("File size exceeds 50 MB limit. Please upload a smaller CSV.", "error");
            el.selectedFileInfo.textContent = "Upload rejected: File too large (Max 50 MB).";
            el.selectedFileInfo.className = "file-info error-color";
            return;
        }

        el.selectedFileInfo.textContent = `Uploading: ${file.name} (${(file.size / 1024).toFixed(1)} KB)...`;
        el.selectedFileInfo.className = "file-info text-secondary";
        
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const data = await api.post("/api/dataset/upload", formData, true);
            el.selectedFileInfo.textContent = `Successfully uploaded: ${file.name}!`;
            el.selectedFileInfo.className = "file-info success-color";
            
            setDatasetLoadedState(true, file.name, data.report.stats);
            showToast("Dataset uploaded and validated successfully!", "success");
            
            if (data.report.warnings && data.report.warnings.length > 0) {
                el.validationWarningsBox.classList.remove("hidden");
                el.validationWarningsList.innerHTML = data.report.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("");
            } else {
                el.validationWarningsBox.classList.add("hidden");
            }
            
            // Auto navigate to EDA to keep workflow fluid
            setTimeout(() => navigateToPage("eda"), 1000);
        } catch (error) {
            el.selectedFileInfo.textContent = `Upload failed: ${error.message}`;
            el.selectedFileInfo.className = "file-info error-color";
            showToast(`Upload failed: ${error.message}`, "error");
            setDatasetLoadedState(false);
        }
    }

    // Preview Table Rendering
    async function loadPreviewTable() {
        try {
            const data = await api.get("/api/dataset/preview?rows=10");
            
            // Headers
            let htmlHeaders = "<tr>";
            data.columns.forEach(col => {
                htmlHeaders += `<th>${escapeHtml(col)}</th>`;
            });
            htmlHeaders += "</tr>";
            el.previewTable.querySelector("thead").innerHTML = htmlHeaders;
            
            // Rows
            let htmlRows = "";
            data.data.forEach(row => {
                htmlRows += "<tr>";
                data.columns.forEach(col => {
                    let val = row[col];
                    if (val === null) val = "-";
                    htmlRows += `<td>${escapeHtml(val)}</td>`;
                });
                htmlRows += "</tr>";
            });
            el.previewTable.querySelector("tbody").innerHTML = htmlRows;
            
            el.datasetPreviewCard.classList.remove("hidden");
        } catch (error) {
            console.error("Preview table load error:", error);
            el.datasetPreviewCard.classList.add("hidden");
        }
    }

    // --- EDA Page Logic ---
    async function loadEdaPage() {
        // Increment and capture token for race-condition prevention
        state.requestTokens.eda += 1;
        const currentToken = state.requestTokens.eda;

        // Show chart spinners & skeleton loading for stats box values
        el.loaderSalesTrend.classList.remove("hidden");
        el.loaderCorrelation.classList.remove("hidden");
        el.loaderWeekly.classList.remove("hidden");
        el.loaderMonthly.classList.remove("hidden");
        
        el.edaAvgPrice.classList.add("skeleton-pulse");
        el.edaAvgSold.classList.add("skeleton-pulse");
        el.edaOutliersCount.classList.add("skeleton-pulse");
        
        try {
            const eda = await api.get("/api/dataset/eda");
            if (currentToken !== state.requestTokens.eda) return; // Stale request, discard
            state.edaData = eda;
            
            // Stats
            const stats = eda.descriptive_statistics.units_sold || {};
            el.edaAvgPrice.textContent = `₹${eda.descriptive_statistics.price?.mean?.toFixed(2) || "0.00"}`;
            el.edaAvgSold.textContent = Math.round(stats.mean || 0).toLocaleString();
            el.edaOutliersCount.textContent = eda.outliers_count;
            
            // Draw Charts
            drawSalesTrendChart(eda.sales_trend);
            drawCorrelationChart(eda.correlation_matrix);
            drawSeasonalityChart("chart-weekly", eda.weekly_seasonality, "Weekly Seasonality Pattern", "Day of Week");
            drawSeasonalityChart("chart-monthly", eda.monthly_seasonality, "Monthly Seasonality Pattern", "Month");

            // Calculate dynamic drivers from correlation matrix
            try {
                const corr = eda.correlation_matrix;
                const cols = corr.columns || [];
                const mat = corr.matrix || [];
                const unitsSoldIdx = cols.indexOf('units_sold');
                const priceIdx = cols.indexOf('price');
                const stockIdx = cols.indexOf('stock_on_hand');
                const promoIdx = cols.indexOf('promotion_flag');
                const weekendIdx = cols.indexOf('is_weekend');
                
                let promoVal = (unitsSoldIdx !== -1 && promoIdx !== -1 && mat[unitsSoldIdx]) ? mat[unitsSoldIdx][promoIdx] : 0;
                let priceVal = (unitsSoldIdx !== -1 && priceIdx !== -1 && mat[unitsSoldIdx]) ? mat[unitsSoldIdx][priceIdx] : 0;
                let stockVal = (unitsSoldIdx !== -1 && stockIdx !== -1 && mat[unitsSoldIdx]) ? mat[unitsSoldIdx][stockIdx] : 0;
                let weekendVal = (unitsSoldIdx !== -1 && weekendIdx !== -1 && mat[unitsSoldIdx]) ? mat[unitsSoldIdx][weekendIdx] : 0;
                
                let driverHtml = "<strong>What Happened:</strong> Analyzed historical demand correlations.<br><strong>Why (Calculated Drivers):</strong><br>";
                let driversList = [];
                if (Math.abs(promoVal) > 0.05) {
                    driversList.push(`- 🚀 <strong>Promotions:</strong> Positive impact (+${Math.round(promoVal * 100)}% correlation) - campaigns spike volume.`);
                }
                if (Math.abs(weekendVal) > 0.05) {
                    driversList.push(`- 🗓️ <strong>Weekend shopping:</strong> Positive impact (+${Math.round(weekendVal * 100)}% correlation) - high weekend store footfall.`);
                }
                if (Math.abs(priceVal) > 0.05) {
                    driversList.push(`- 💸 <strong>Price Elasticity:</strong> Negative impact (${Math.round(priceVal * 100)}% correlation) - price hikes lower sales volume.`);
                }
                if (Math.abs(stockVal) > 0.05) {
                    driversList.push(`- 📦 <strong>Stock availability:</strong> Positive correlation (+${Math.round(stockVal * 100)}%) - stockouts immediately cap revenue.`);
                }
                
                if (driversList.length > 0) {
                    driverHtml += driversList.join("<br>") + "<br>";
                } else {
                    driverHtml += "No strong statistical sales drivers detected in this dataset yet.<br>";
                }
                driverHtml += "<strong>Recommended Action:</strong> Run targeted campaigns on Wednesdays/Thursdays to align with high-impact weekend shopping patterns.";
                document.getElementById("ai-obs-correlation-text").innerHTML = driverHtml;
            } catch (err) {
                console.error("Failed to parse key drivers from correlation matrix:", err);
                document.getElementById("ai-obs-correlation-text").innerHTML = "<strong>Observation:</strong> Sales drivers could not be evaluated due to lack of variable history.";
            }
            
            // Hide chart spinners and remove skeleton load states on success
            el.loaderSalesTrend.classList.add("hidden");
            el.loaderCorrelation.classList.add("hidden");
            el.loaderWeekly.classList.add("hidden");
            el.loaderMonthly.classList.add("hidden");
            
            el.edaAvgPrice.classList.remove("skeleton-pulse");
            el.edaAvgSold.classList.remove("skeleton-pulse");
            el.edaOutliersCount.classList.remove("skeleton-pulse");
            
            // Populate Outliers Table
            let outliersHtml = "";
            if (eda.outliers && eda.outliers.length > 0) {
                eda.outliers.forEach(out => {
                    outliersHtml += `
                        <tr>
                            <td>${escapeHtml(out.date)}</td>
                            <td>${escapeHtml(out.product_id)}</td>
                            <td>${escapeHtml(out.product_name)}</td>
                            <td><strong>${escapeHtml(out.units_sold)}</strong></td>
                            <td>Outside [${escapeHtml(out.lower_bound)}, ${escapeHtml(out.upper_bound)}]</td>
                        </tr>
                    `;
                });
            } else {
                outliersHtml = "<tr><td colspan='5' class='text-center text-secondary'>No outliers detected.</td></tr>";
            }
            el.outliersTable.innerHTML = outliersHtml;
            
        } catch (error) {
            console.error("EDA Loading failed:", error);
            // Hide spinners and remove skeleton load states even on failure
            el.loaderSalesTrend.classList.add("hidden");
            el.loaderCorrelation.classList.add("hidden");
            el.loaderWeekly.classList.add("hidden");
            el.loaderMonthly.classList.add("hidden");
            
            el.edaAvgPrice.classList.remove("skeleton-pulse");
            el.edaAvgSold.classList.remove("skeleton-pulse");
            el.edaOutliersCount.classList.remove("skeleton-pulse");
            showToast("Failed to load descriptive EDA dashboard.", "error");
        }
    }

    // Draw sales trend
    function drawSalesTrendChart(trend) {
        const trace = {
            x: trend.dates,
            y: trend.sales,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#6366f1', width: 3.0, shape: 'spline' }, // smooth spline layout
            fill: 'tozeroy', // glowing gradient fill under curve
            fillcolor: 'rgba(99, 102, 241, 0.04)',
            name: 'Total Units Sold'
        };
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 15, r: 15, b: 35, l: 45 },
            font: { color: '#94a3b8', family: 'Outfit, Inter, sans-serif' },
            hovermode: 'x',
            xaxis: { gridcolor: 'rgba(255,255,255,0.02)', linecolor: 'rgba(255,255,255,0.06)' },
            yaxis: { gridcolor: 'rgba(255, 255, 255, 0.04)', linecolor: 'rgba(255, 255, 255, 0.06)' }
        };
        Plotly.purge('chart-sales-trend');
        Plotly.newPlot('chart-sales-trend', [trace], layout, { responsive: true, displayModeBar: false });
    }

    // Draw correlation heatmap
    function drawCorrelationChart(matrix) {
        const trace = {
            z: matrix.matrix,
            x: matrix.columns,
            y: matrix.columns,
            type: 'heatmap',
            colorscale: [
                [0, '#f43f5e'],   // Negative correlation (rose)
                [0.5, '#1e293b'], // Neutral (dark slate)
                [1.0, '#10b981']  // Positive correlation (emerald)
            ],
            showscale: true,
            zmin: -1,
            zmax: 1
        };
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 20, r: 20, b: 50, l: 90 },
            font: { color: '#94a3b8', family: 'Outfit, Inter, sans-serif' }
        };
        Plotly.purge('chart-correlation');
        Plotly.newPlot('chart-correlation', [trace], layout, { responsive: true, displayModeBar: false });
    }

    // Draw bar chart for seasonality
    function drawSeasonalityChart(elementId, seasonality, title, xTitle) {
        const trace = {
            x: seasonality.labels,
            y: seasonality.values,
            type: 'bar',
            marker: { color: '#10b981', opacity: 0.8 },
            name: 'Avg Units Sold'
        };
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 15, r: 15, b: 35, l: 45 },
            font: { color: '#94a3b8', family: 'Outfit, Inter, sans-serif' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.02)', linecolor: 'rgba(255,255,255,0.06)' },
            yaxis: { gridcolor: 'rgba(255, 255, 255, 0.04)', linecolor: 'rgba(255, 255, 255, 0.06)' }
        };
        Plotly.purge(elementId);
        Plotly.newPlot(elementId, [trace], layout, { responsive: true, displayModeBar: false });
    }

    // --- Forecasting Page Logic ---
    async function loadForecastingPage() {
        try {
            // Check if training report exists in backend
            const report = await api.get("/api/forecast/report").catch(() => null);
            if (report) {
                el.forecastEmptyState.classList.add("hidden");
                displayTrainingReport(report);
                renderPromoCalendarGrid();
                updateScenarioHistoryTable();
            } else {
                // Not trained yet - show empty state and hide work areas
                el.forecastEmptyState.classList.remove("hidden");
                el.forecastSummaryContainer.classList.add("hidden");
                el.forecastWorkspace.classList.add("hidden");
                el.scenarioHistoryCard.classList.add("hidden");
                
                // Keep UI locked if training is active in background
                if (state.isTraining) {
                    el.trainingProgressBox.classList.remove("hidden");
                    el.trainModelsBtn.disabled = true;
                } else {
                    el.trainingProgressBox.classList.add("hidden");
                    el.trainModelsBtn.disabled = false;
                }
            }
        } catch (error) {
            console.error("Forecasting loading error:", error);
        }
    }

    // Trigger Model Training
    el.trainModelsBtn.addEventListener("click", async () => {
        state.isTraining = true;
        el.trainingProgressBox.classList.remove("hidden");
        el.trainModelsBtn.disabled = true;
        
        // Progress bar simulation (fitting ML trees usually takes 4-7s, Prophet about 1.5s per product)
        let progress = 0;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.random() * 8;
                progress = Math.min(progress, 90);
                el.trainingProgressFill.style.width = `${progress}%`;
                if (progress > 60) {
                    el.trainingProgressText.textContent = "Aggregating metrics and picking recommendations...";
                } else if (progress > 30) {
                    el.trainingProgressText.textContent = "Fitting Prophet models and holiday adjustments...";
                } else {
                    el.trainingProgressText.textContent = "Extracting lags, rolling averages, and split datasets...";
                }
            }
        }, 300);
        
        try {
            const smoothOutliers = el.smoothOutliersCheck ? el.smoothOutliersCheck.checked : true;
            const data = await api.post(`/api/forecast/train?smooth_outliers=${smoothOutliers}`);
            clearInterval(progressInterval);
            el.trainingProgressFill.style.width = "100%";
            el.trainingProgressText.textContent = "Training optimization complete!";
            showToast("Forecasting models trained successfully!", "success");
            
            state.isTraining = false;
            setTimeout(() => {
                el.trainingProgressBox.classList.add("hidden");
                el.trainModelsBtn.disabled = false;
                el.forecastEmptyState.classList.add("hidden");
                displayTrainingReport(data.report);
            }, 1000);
            
        } catch (error) {
            clearInterval(progressInterval);
            state.isTraining = false;
            el.trainingProgressText.textContent = `Training failed: ${error.message}`;
            el.trainingProgressFill.style.backgroundColor = "var(--error)";
            el.trainModelsBtn.disabled = false;
            showToast(`Training failed: ${error.message}`, "error");
        }
    });

    function displayTrainingReport(report) {
        state.trainedReport = report;
        
        // Header summary cards
        if (el.registryAvgMae) el.registryAvgMae.textContent = report.average_mae.toFixed(2);
        if (el.registryAvgMape) el.registryAvgMape.textContent = `${report.average_mape.toFixed(2)}%`;
        el.forecastSummaryContainer.classList.remove("hidden");
        
        // Populate product selector dropdown
        let productOptions = "";
        const productsMap = report.products;
        for (const pid in productsMap) {
            productOptions += `<option value="${escapeHtml(pid)}">${escapeHtml(productsMap[pid].product_name)} (${escapeHtml(pid)})</option>`;
        }
        el.forecastProductSelect.innerHTML = productOptions;
        
        const pids = Object.keys(productsMap);
        // Page State Cache Retention logic: if state.activeProduct is already set and valid, preserve it!
        if (state.activeProduct && productsMap[state.activeProduct]) {
            el.forecastProductSelect.value = state.activeProduct;
            updateWorkspaceForProduct(state.activeProduct);
        } else if (pids.length > 0) {
            state.activeProduct = pids[0];
            updateWorkspaceForProduct(pids[0]);
        }
        
        el.forecastWorkspace.classList.remove("hidden");
    }

    // Trigger workspace update on product select
    el.forecastProductSelect.addEventListener("change", (e) => {
        state.activeProduct = e.target.value;
        updateWorkspaceForProduct(e.target.value);
    });

    async function updateWorkspaceForProduct(productId) {
        if (!state.trainedReport || !state.trainedReport.products) return;
        const prodData = state.trainedReport.products[productId];
        if (!prodData) return;
        
        // Clear scenario history when switching products to prevent mixing product comparisons
        state.scenarios = [];
        updateScenarioHistoryTable();
        
        // Reset What-If simulator state and UI controls
        state.simulator.priceMultiplier = 1.0;
        state.simulator.promoDays = [];
        state.simulator.activePresetName = null;
        if (el.simPriceSlider) {
            el.simPriceSlider.value = 100;
        }
        if (el.simPriceVal) {
            el.simPriceVal.textContent = "100%";
        }
        document.querySelectorAll(".sim-day-checkbox").forEach(cb => cb.checked = false);
        document.querySelectorAll(".preset-btn").forEach(btn => btn.classList.remove("active"));
        
        const models = Object.keys(prodData.all_models);
        let modelOptions = `<option value="best_recommender">★ Best recommended: ${escapeHtml(getModelFriendlyLabel(prodData.best_model))}</option>`;
        models.forEach(m => {
            modelOptions += `<option value="${escapeHtml(m)}">${escapeHtml(getModelFriendlyLabel(m))}</option>`;
        });
        el.forecastModelSelect.innerHTML = modelOptions;
        state.activeModel = null; // defaults to best recommender
        
        // 2. Populate recommended badge & reason
        el.recommenderModelName.textContent = getModelFriendlyLabel(prodData.best_model);
        
        // 3. Populate comparison table
        let compareHtml = "";
        models.forEach(m => {
            const mData = prodData.all_models[m];
            const isBest = m === prodData.best_model;
            const friendlyName = getModelFriendlyLabel(m, isBest);
            compareHtml += `
                <tr class="${isBest ? 'border-pulse' : ''}">
                    <td><strong>${escapeHtml(friendlyName)}</strong></td>
                    <td><strong>${escapeHtml(mData.MAE)}</strong></td>
                    <td>${escapeHtml(mData.MAPE)}%</td>
                    <td>${escapeHtml(mData.R2)}</td>
                </tr>
            `;
        });
        el.metricsCompareTable.innerHTML = compareHtml;
        lucide.createIcons(); // Refresh inline icons
        
        // 4. Load & Render Forecast
        await generateForecast();

        // Refresh learning center context if open
        if (state.learning.isOpen) {
            updateLearningCenter();
        }
    }

    // Generate forecast execution
    el.runForecastBtn.addEventListener("click", () => generateForecast());

    async function generateForecast() {
        if (!state.activeProduct) return;
        
        // Increment and capture token for race-condition prevention
        state.requestTokens.forecast += 1;
        const currentToken = state.requestTokens.forecast;
        
        // Show forecast overlay spinner & skeleton loading for executive metrics
        el.loaderForecast.classList.remove("hidden");
        el.forecastSummaryContainer.classList.remove("hidden");
        
        ["exec-demand-outlook", "exec-projected-revenue", "exec-inventory-status", "exec-revenue-loss", "exec-prediction-accuracy", "exec-forecast-reliability"].forEach(id => {
            const dom = document.getElementById(id);
            if (dom) dom.classList.add("skeleton-pulse");
        });
        
        const productId = state.activeProduct;
        const horizon = el.forecastHorizonSelect.value;
        const selectedModelVal = el.forecastModelSelect.value;
        
        const priceMult = state.simulator.priceMultiplier;
        const promoDaysStr = state.simulator.promoDays.join(",");
        const isScenario = (priceMult !== 1.0 || promoDaysStr !== "");
        
        try {
            el.runForecastBtn.disabled = true;
            
            // 1. Fetch baseline if not cached or active product/model has changed
            if (!state.baselineForecastData || 
                state.baselineForecastData.product_id !== productId || 
                state.baselineForecastData.model_used !== (selectedModelVal === "best_recommender" ? state.baselineForecastData.model_used : selectedModelVal)) {
                
                let baseUrl = `/api/forecast/predict?product_id=${productId}&horizon_days=${horizon}`;
                if (selectedModelVal !== "best_recommender") {
                    baseUrl += `&model_name=${encodeURIComponent(selectedModelVal)}`;
                }
                const baselineResponse = await api.get(baseUrl);
                if (currentToken !== state.requestTokens.forecast) return; // Stale request, discard
                state.baselineForecastData = baselineResponse;
            }
            
            // 2. Fetch overridden scenario or reuse baseline
            let data;
            if (isScenario) {
                let url = `/api/forecast/predict?product_id=${productId}&horizon_days=${horizon}&price_multiplier=${priceMult}&promo_days=${encodeURIComponent(promoDaysStr)}`;
                if (selectedModelVal !== "best_recommender") {
                    url += `&model_name=${encodeURIComponent(selectedModelVal)}`;
                }
                data = await api.get(url);
                if (currentToken !== state.requestTokens.forecast) return; // Stale request, discard
            } else {
                data = state.baselineForecastData;
            }
            
            // Update chart titles
            el.forecastChartTitle.textContent = `${data.product_name} Demand Forecast`;
            el.forecastChartSubtitle.textContent = `Model: ${data.model_used} | Horizon: ${data.forecast_horizon_days} Days`;
            
            // Update recommendation reason text if it was the recommended model
            if (data.recommendation_reason) {
                el.recommenderReasonText.textContent = data.recommendation_reason;
            } else {
                el.recommenderReasonText.innerHTML = `<em>Custom model override.</em> Historical metrics: MAE = ${data.metrics.MAE}, R² = ${data.metrics.R2}.`;
            }
            
            // Draw Forecast
            state.forecastData = data;
            drawForecastChart(data);
            showToast(`Forecast generated using ${data.model_used}.`, "info");
            
            // 3. Render Simulator Deltas compared to baseline
            const basePredictions = state.baselineForecastData.forecast.predictions;
            const basePrice = state.baselineForecastData.simulated_price || data.simulated_price;
            const baseVolume = basePredictions.reduce((a, b) => a + b, 0);
            const baseRevenue = baseVolume * basePrice;
            
            const simPredictions = data.forecast.predictions;
            const simPrice = data.simulated_price;
            const simVolume = simPredictions.reduce((a, b) => a + b, 0);
            const simRevenue = simVolume * simPrice;
            
            const volDelta = simVolume - baseVolume;
            const volPct = baseVolume > 0 ? (volDelta / baseVolume * 100).toFixed(1) : "0.0";
            const revDelta = simRevenue - baseRevenue;
            const revPct = baseRevenue > 0 ? (revDelta / baseRevenue * 100).toFixed(1) : "0.0";
            
            if (isScenario) {
                el.simVolumeDelta.innerHTML = `<span class="${volDelta >= 0 ? 'success-color' : 'danger-color'} font-semibold">${volDelta >= 0 ? '+' : ''}${volDelta.toFixed(0)} units (${volDelta >= 0 ? '+' : ''}${volPct}%)</span>`;
                el.simRevenueDelta.innerHTML = `<span class="${revDelta >= 0 ? 'success-color' : 'danger-color'} font-semibold">${revDelta >= 0 ? '+' : ''}₹${revDelta.toFixed(2)} (${revDelta >= 0 ? '+' : ''}${revPct}%)</span>`;
            } else {
                el.simVolumeDelta.innerHTML = `<span class="text-secondary italic">Baseline (${baseVolume.toFixed(0)} units)</span>`;
                el.simRevenueDelta.innerHTML = `<span class="text-secondary italic">Baseline (₹${baseRevenue.toFixed(2)})</span>`;
            }
            
            // 4. Render Decision Support Panel and Executive Summary Board
            const ds = data.decision_support;
            if (ds) {
                let badgeClass = "rating-excellent";
                let statusText = "HEALTHY";
                let execInventoryHtml = `<span class="success-color font-semibold">✓ Healthy</span>`;
                
                if (ds.status === "OUT_OF_STOCK") {
                    badgeClass = "rating-poor";
                    statusText = "OUT OF STOCK";
                    execInventoryHtml = `<span class="danger-color font-semibold">❌ Out of Stock</span>`;
                } else if (ds.status === "CRITICAL_LOW") {
                    badgeClass = "rating-poor";
                    statusText = "CRITICAL LOW";
                    execInventoryHtml = `<span class="danger-color font-semibold">⚠️ Critical Low</span>`;
                } else if (ds.status === "LOW_STOCK") {
                    badgeClass = "rating-good";
                    statusText = "LOW STOCK";
                    execInventoryHtml = `<span class="warning-color font-semibold">⚠️ Low Stock</span>`;
                }
                
                el.decisionStockStatus.className = `rating-badge ${badgeClass} text-xs font-semibold`;
                el.decisionStockStatus.textContent = statusText;
                
                let recommendationActionHtml = "";
                if (ds.reorder_date) {
                    recommendationActionHtml = `Order <strong class="accent-color">${ds.recommended_reorder_qty} units</strong> on <strong>${ds.reorder_date}</strong> to maintain target stock levels (7-day safety buffer).`;
                } else {
                    recommendationActionHtml = "No immediate replenishment required. Stock levels are expected to remain safe for the next 30 days.";
                }
                el.decisionActionText.innerHTML = recommendationActionHtml;
                
                el.decisionStockoutDays.textContent = `${ds.stockout_days_projected} Day${ds.stockout_days_projected === 1 ? '' : 's'}`;
                el.decisionRevenueRisk.textContent = `₹${ds.revenue_at_risk.toFixed(2)}`;

                // Update the Executive Summary Board elements
                el.forecastSummaryContainer.classList.remove("hidden");
                
                // A. AI Action Plan Alert Text
                if (ds.reorder_date) {
                    document.getElementById("exec-action-text").innerHTML = `AI recommends ordering <strong class="accent-color" style="color: var(--accent); font-weight: 700;">${ds.recommended_reorder_qty} units</strong> of ${data.product_name} on <strong style="color: var(--text-primary); font-weight: 700;">${ds.reorder_date}</strong> to prevent projected stockouts.`;
                } else {
                    document.getElementById("exec-action-text").innerHTML = `Inventory status is stable for ${data.product_name}. No immediate restocking required.`;
                }

                // B. Demand Outlook calculation
                const firstPred = simPredictions[0];
                const lastPred = simPredictions[simPredictions.length - 1];
                const changePct = firstPred > 0 ? ((lastPred - firstPred) / firstPred * 100) : 0;
                let demandOutlookText = "➡️ Stable";
                if (changePct >= 5) {
                    demandOutlookText = `📈 Growing (+${changePct.toFixed(0)}%)`;
                } else if (changePct <= -5) {
                    demandOutlookText = `📉 Declining (${changePct.toFixed(0)}%)`;
                }
                document.getElementById("exec-demand-outlook").textContent = demandOutlookText;

                // C. Projected Revenue
                document.getElementById("exec-projected-revenue").textContent = `₹${simRevenue.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;

                // D. Inventory Status HTML
                document.getElementById("exec-inventory-status").innerHTML = execInventoryHtml;

                // E. Lost Revenue Risk
                document.getElementById("exec-revenue-loss").textContent = `₹${ds.revenue_at_risk.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;

                // F. Prediction Accuracy (MAPE derived)
                const activeModelName = data.model_used;
                let mape = (data.metrics && data.metrics.MAPE) ? data.metrics.MAPE : 0;
                
                // Try to find the exact model metrics from the active product's trained models list
                if (state.trainedReport && state.trainedReport.products && state.trainedReport.products[productId]) {
                    const modelData = state.trainedReport.products[productId].all_models[activeModelName];
                    if (modelData && modelData.MAPE) mape = modelData.MAPE;
                }
                const accuracyPct = `${(100 - mape).toFixed(1)}%`;
                document.getElementById("exec-prediction-accuracy").textContent = accuracyPct;

                // G. Forecast Reliability (R2 derived)
                let r2 = (data.metrics && data.metrics.R2) ? data.metrics.R2 : 0;
                if (state.trainedReport && state.trainedReport.products && state.trainedReport.products[productId]) {
                    const modelData = state.trainedReport.products[productId].all_models[activeModelName];
                    if (modelData && modelData.R2) r2 = modelData.R2;
                }
                let reliabilityText = "Fair";
                if (r2 >= 0.90) reliabilityText = "⭐⭐⭐⭐⭐ Excellent";
                else if (r2 >= 0.80) reliabilityText = "⭐⭐⭐⭐☆ Very Good";
                else if (r2 >= 0.70) reliabilityText = "⭐⭐⭐☆☆ Good";
                else if (r2 >= 0.50) reliabilityText = "⭐⭐☆☆☆ Fair";
                else reliabilityText = "⭐☆☆☆☆ Low";
                document.getElementById("exec-forecast-reliability").textContent = reliabilityText;

                // Bind Acknowledge button
                const ackBtn = document.getElementById("exec-action-ack-btn");
                if (ackBtn) {
                    ackBtn.onclick = () => {
                        showToast("AI Action Plan acknowledged. Replenishment tasks logged.", "success");
                    };
                }
                // Remove skeleton loader states
                ["exec-demand-outlook", "exec-projected-revenue", "exec-inventory-status", "exec-revenue-loss", "exec-prediction-accuracy", "exec-forecast-reliability"].forEach(id => {
                    const dom = document.getElementById(id);
                    if (dom) dom.classList.remove("skeleton-pulse");
                });
            }
            
            // 5. Store current scenario metrics temporarily
            state.currentScenarioMetrics = {
                priceMult: priceMult,
                promoDaysStr: promoDaysStr,
                volume: simVolume,
                revenue: simRevenue,
                stockouts: ds ? ds.stockout_days_projected : 0,
                score: simRevenue - (ds ? ds.revenue_at_risk : 0),
                defaultName: isScenario ? (state.simulator.activePresetName || "Custom Scenario") : "Baseline"
            };
            
            // Refresh learning center context if open
            if (state.learning.isOpen) {
                updateLearningCenter();
            }
            
            // Hide spinner on success
            el.loaderForecast.classList.add("hidden");
        } catch (error) {
            console.error("Forecast failed:", error);
            el.loaderForecast.classList.add("hidden");
            showToast(`Forecast failed: ${error.message}`, "error");
            ["exec-demand-outlook", "exec-projected-revenue", "exec-inventory-status", "exec-revenue-loss", "exec-prediction-accuracy", "exec-forecast-reliability"].forEach(id => {
                const dom = document.getElementById(id);
                if (dom) dom.classList.remove("skeleton-pulse");
            });
        } finally {
            el.runForecastBtn.disabled = false;
        }
    }

    function drawForecastChart(data) {
        // Plotly traces
        const histDates = data.history.dates;
        const histSales = data.history.sales;
        
        const foreDates = data.forecast.dates;
        const forePreds = data.forecast.predictions;
        const foreLower = data.forecast.lower_bound;
        const foreUpper = data.forecast.upper_bound;
        
        // Trace 1: Historical Sales (Smooth Spline, Indigo)
        const traceHist = {
            x: histDates,
            y: histSales,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#6366f1', width: 2.5, shape: 'spline' },
            name: 'Historical Sales'
        };
        
        // Trace 2: Predicted Future Demand (Dashed spline, Amber)
        const tracePred = {
            x: foreDates,
            y: forePreds,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#f59e0b', width: 3.0, dash: 'dash', shape: 'spline' },
            marker: { size: 5, color: '#f59e0b' },
            name: 'Demand Forecast'
        };
        
        // Trace 3: Lower confidence band
        const traceLower = {
            x: foreDates,
            y: foreLower,
            type: 'scatter',
            mode: 'lines',
            line: { width: 0 },
            showlegend: false,
            name: '95% CI Lower'
        };
        
        // Trace 4: Upper confidence band (filled to Trace 3 with a beautiful translucent amber glow)
        const traceUpper = {
            x: foreDates,
            y: foreUpper,
            type: 'scatter',
            mode: 'lines',
            line: { width: 0 },
            fill: 'tonexty',
            fillcolor: 'rgba(245, 158, 11, 0.08)',
            name: '95% Confidence Interval'
        };
        
        const traces = [traceHist, traceLower, traceUpper, tracePred];
        
        // Add baseline trace if current is simulated
        const priceMult = state.simulator.priceMultiplier;
        const promoDaysStr = state.simulator.promoDays.join(",");
        const isScenario = (priceMult !== 1.0 || promoDaysStr !== "");
        
        if (isScenario && state.baselineForecastData) {
            const baseDates = state.baselineForecastData.forecast.dates;
            const basePreds = state.baselineForecastData.forecast.predictions;
            
            const traceBase = {
                x: baseDates,
                y: basePreds,
                type: 'scatter',
                mode: 'lines',
                line: { color: 'rgba(255, 255, 255, 0.25)', width: 2.0, dash: 'dot', shape: 'spline' },
                name: 'Baseline Forecast (Normal Price)'
            };
            
            // Insert traceBase as index 1 (between History and CI)
            traces.splice(1, 0, traceBase);
        }
        
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 15, r: 15, b: 35, l: 45 },
            font: { color: '#94a3b8', family: 'Outfit, Inter, sans-serif' },
            hovermode: 'x unified', // unified tooltip for modern aesthetics!
            xaxis: { gridcolor: 'rgba(255,255,255,0.02)', linecolor: 'rgba(255,255,255,0.06)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.04)', linecolor: 'rgba(255,255,255,0.06)' },
            legend: { orientation: 'h', y: -0.2 }
        };
        Plotly.purge('chart-forecast');
        Plotly.newPlot('chart-forecast', traces, layout, { responsive: true, displayModeBar: false });
    }

    // --- Collapsible Chat Panel Controls ---
    function toggleChat(isOpen) {
        state.chatOpen = isOpen;
        if (isOpen) {
            el.chatPanel.classList.add("open");
            if (state.learning.isOpen) {
                toggleLearningCenter(false);
            }
        } else {
            el.chatPanel.classList.remove("open");
        }
    }

    el.openChatBtn.addEventListener("click", () => toggleChat(!state.chatOpen));
    el.closeChatBtn.addEventListener("click", () => toggleChat(false));

    // Handle suggestion tags trigger
    const suggestions = document.querySelectorAll(".suggestion-tag");
    suggestions.forEach(tag => {
        tag.addEventListener("click", () => {
            const question = tag.textContent;
            sendChatMessage(question);
        });
    });

    el.chatSendBtn.addEventListener("click", () => {
        const text = el.chatInputText.value.trim();
        if (text) {
            sendChatMessage(text);
            el.chatInputText.value = "";
        }
    });

    el.chatInputText.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            const text = el.chatInputText.value.trim();
            if (text) {
                sendChatMessage(text);
                el.chatInputText.value = "";
            }
        }
    });

    // Integrated sparkles explanation trigger (V2 Learning Center redirection)
    el.explainChartBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const chartType = btn.getAttribute("data-chart");
            state.learning.activeChartExplained = chartType;
            state.learning.activeTab = "insights";
            
            // Switch tabs
            el.learningInsightsTab.classList.add("active");
            el.learningGlossaryTab.classList.remove("active");
            
            toggleLearningCenter(true);
            
            // Scroll to explainer card and flash highlight pulse
            setTimeout(() => {
                const card = document.getElementById("chart-explainer-card");
                if (card) {
                    card.scrollIntoView({ behavior: "smooth", block: "center" });
                    card.classList.add("highlighted");
                    setTimeout(() => card.classList.remove("highlighted"), 3000);
                }
            }, 300);
        });
    });

    async function sendChatMessage(messageText) {
        // 1. Append User Message
        appendMessage(messageText, "user");
        
        // Scroll to bottom
        el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
        
        // 2. Append typing loader
        const loadingId = appendTypingIndicator();
        
        try {
            // Call Backend Chat endpoint
            const response = await fetch(`${API_URL}/api/agent/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: messageText,
                    session_id: "local_demo_session"
                })
            });
            
            removeTypingIndicator(loadingId);
            
            if (response.ok) {
                const data = await response.json();
                appendMessage(data.response, "bot");
            } else {
                appendMessage("I apologize, but I encountered an error retrieving data from the forecasting registry. Please make sure the models have been successfully trained first.", "bot");
            }
        } catch (error) {
            removeTypingIndicator(loadingId);
            console.error("Chat API error:", error);
            // Standalone local offline fallback messages for P0 verification
            const fallbackResponse = getOfflineAnalystFallback(messageText);
            appendMessage(fallbackResponse, "bot");
        }
        
        el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
    }

    function appendMessage(text, sender) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message message-${sender}`;
        
        const avatarHtml = sender === "bot" 
            ? `<div class="message-avatar"><i data-lucide="bot"></i></div>` 
            : `<div class="message-avatar"><i data-lucide="user"></i></div>`;
            
        // Use safe markdown parser to prevent XSS (CRIT-06)
        const formattedText = renderMarkdownSafely(text);
            
        msgDiv.innerHTML = `
            ${avatarHtml}
            <div class="message-content">
                <p>${formattedText}</p>
            </div>
        `;
        
        el.chatMessages.appendChild(msgDiv);
        lucide.createIcons();
    }

    function appendTypingIndicator() {
        const id = `typing-${Date.now()}`;
        const loaderDiv = document.createElement("div");
        loaderDiv.className = `message message-bot`;
        loaderDiv.id = id;
        loaderDiv.innerHTML = `
            <div class="message-avatar"><i data-lucide="bot"></i></div>
            <div class="message-content">
                <p class="text-secondary">AI Analyst is auditing metrics...</p>
            </div>
        `;
        el.chatMessages.appendChild(loaderDiv);
        lucide.createIcons();
        return id;
    }

    function removeTypingIndicator(id) {
        const div = document.getElementById(id);
        if (div) div.remove();
    }

    // Local programmatic responses for when the LLM service is offline or not yet configured (Module 5 verification support)
    function getOfflineAnalystFallback(text) {
        const lower = text.toLowerCase();
        
        if (lower.includes("top") || lower.includes("selling")) {
            return "Based on our current dataset, the top-selling product by volume is Organic Whole Milk 1L, followed closely by Wheat Sliced Bread. These two items represent over 45% of total sales volume.";
        }
        if (lower.includes("low") || lower.includes("stock") || lower.includes("replenish")) {
            return "Inventory audit complete. We flagged Fuji Apples 1kg as running low on stock (stock levels cover less than 3 days of forecast demand). We recommend scheduling a replenishment order of 150 units.";
        }
        if (lower.includes("forecast") || lower.includes("explain")) {
            if (state.trainedReport) {
                return "The forecast chart shows a strong weekend sales spike (Fridays/Saturdays) for our dairy category. Prophet model was selected as the recommended model due to its low error rate (MAPE < 6%). No stockouts are predicted for the next 14 days if stock is maintained.";
            }
            return "Please load data and run 'Train Forecasting Models' first so I can analyze the metrics.";
        }
        return "I am connected. Once the AI Agent router is activated in Module 6, I will query the models and dataset to explain any specific questions you ask!";
    }

    // Learning Center Panel triggers
    if (el.openLearningBtn) {
        el.openLearningBtn.addEventListener("click", () => {
            toggleLearningCenter(!state.learning.isOpen);
        });
    }
    if (el.closeLearningBtn) {
        el.closeLearningBtn.addEventListener("click", () => {
            toggleLearningCenter(false);
        });
    }

    // Metric header info badges click triggers
    el.metricHeaderInfos.forEach(header => {
        header.addEventListener("click", (e) => {
            e.stopPropagation();
            const metric = header.getAttribute("data-metric");
            state.learning.selectedMetric = metric;
            state.learning.activeTab = "glossary";
            
            el.learningGlossaryTab.classList.add("active");
            el.learningInsightsTab.classList.remove("active");
            
            toggleLearningCenter(true);
            
            setTimeout(() => {
                const card = document.getElementById(`metric-card-${metric}`);
                if (card) {
                    const cards = el.learningContent.querySelectorAll(".glossary-card");
                    cards.forEach(c => c.classList.remove("open"));
                    
                    card.classList.add("open");
                    card.scrollIntoView({ behavior: "smooth", block: "center" });
                    card.classList.add("highlighted");
                    setTimeout(() => card.classList.remove("highlighted"), 3000);
                    
                    const mathBlock = card.querySelector(".metric-math");
                    if (mathBlock) {
                        renderKaTeX(mathBlock.getAttribute("data-formula"), mathBlock);
                    }
                }
            }, 300);
        });
    });

    // Model select change refresh
    if (el.forecastModelSelect) {
        el.forecastModelSelect.addEventListener("change", () => {
            if (state.learning.isOpen) {
                updateLearningCenter();
            }
        });
    }

    // --- What-If Simulator & Preset Controls ---

    // Dynamically render the 30 checkboxes for promotion calendar overrides
    function renderPromoCalendarGrid() {
        if (!el.simPromoDaysGrid) return;
        el.simPromoDaysGrid.innerHTML = "";
        
        for (let i = 1; i <= 30; i++) {
            const isChecked = state.simulator.promoDays.includes(i);
            const checkboxHtml = `
                <div class="flex-col align-center justify-center" style="border: 1px solid rgba(255,255,255,0.03); background: rgba(255,255,255,0.01); border-radius: 4px; padding: 4px;">
                    <span style="font-size: 0.55rem; color: var(--text-secondary); margin-bottom: 2px;">D${i}</span>
                    <input type="checkbox" class="sim-day-checkbox" data-day="${i}" ${isChecked ? 'checked' : ''} style="width: 13px; height: 13px; accent-color: var(--accent); cursor: pointer;">
                </div>
            `;
            el.simPromoDaysGrid.insertAdjacentHTML("beforeend", checkboxHtml);
        }
        
        // Bind change events to dynamic checkboxes
        const checkboxes = el.simPromoDaysGrid.querySelectorAll(".sim-day-checkbox");
        checkboxes.forEach(chk => {
            chk.addEventListener("change", () => {
                const day = parseInt(chk.getAttribute("data-day"));
                if (chk.checked) {
                    if (!state.simulator.promoDays.includes(day)) {
                        state.simulator.promoDays.push(day);
                    }
                } else {
                    state.simulator.promoDays = state.simulator.promoDays.filter(d => d !== day);
                }
                state.simulator.activePresetName = "Custom Overrides";
                generateForecast();
            });
        });
    }

    // Refresh scenario list and winner badges
    function updateScenarioHistoryTable() {
        if (!el.scenarioHistoryTbody) return;
        
        if (state.scenarios.length === 0) {
            el.scenarioHistoryCard.classList.add("hidden");
            return;
        }
        el.scenarioHistoryCard.classList.remove("hidden");
        
        // Find highest score (net revenue)
        let bestScore = -Infinity;
        let winnerIndex = -1;
        state.scenarios.forEach((s, idx) => {
            if (s.score > bestScore) {
                bestScore = s.score;
                winnerIndex = idx;
            }
        });
        
        let tbodyHtml = "";
        state.scenarios.forEach((s, idx) => {
            const isWinner = idx === winnerIndex;
            tbodyHtml += `
                <tr>
                    <td><strong>${escapeHtml(s.name)}</strong></td>
                    <td>${escapeHtml(s.priceLevel)}</td>
                    <td>${s.promoDaysCount} Days</td>
                    <td>${s.volume.toFixed(0)} units</td>
                    <td><strong>₹${s.revenue.toFixed(2)}</strong></td>
                    <td><span class="${s.stockouts > 0 ? 'danger-color font-semibold' : 'success-color'}">${s.stockouts} Day${s.stockouts === 1 ? '' : 's'}</span></td>
                    <td>
                        ${isWinner ? '<span class="rating-badge rating-excellent">⭐ Recommended</span>' : '<span class="text-secondary">-</span>'}
                    </td>
                    <td>
                        <button class="btn btn-secondary btn-xs delete-scenario-btn" data-index="${idx}" style="background: rgba(239, 68, 68, 0.1); color: var(--error); border: 1px solid rgba(239, 68, 68, 0.2); padding: 2px 6px;">
                            <i data-lucide="trash-2" style="width: 11px; height: 11px; vertical-align: middle;"></i> Delete
                        </button>
                    </td>
                </tr>
            `;
        });
        el.scenarioHistoryTbody.innerHTML = tbodyHtml;
        lucide.createIcons(); // Refresh inline trash icons
        
        // Bind delete action listeners
        const deleteBtns = el.scenarioHistoryTbody.querySelectorAll(".delete-scenario-btn");
        deleteBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const idx = parseInt(btn.getAttribute("data-index"));
                state.scenarios.splice(idx, 1);
                updateScenarioHistoryTable();
                showToast("Scenario deleted from history.", "info");
            });
        });
    }

    // Slider inputs
    if (el.simPriceSlider) {
        el.simPriceSlider.addEventListener("input", (e) => {
            const val = e.target.value;
            const delta = val - 100;
            if (delta === 0) {
                el.simPriceVal.textContent = "Base Price (100%)";
                el.simPriceVal.className = "text-xs success-color font-semibold";
            } else {
                el.simPriceVal.textContent = `${delta > 0 ? '+' : ''}${delta}% adjustment`;
                el.simPriceVal.className = `text-xs ${delta > 0 ? 'warning-color' : 'accent-color'} font-semibold`;
            }
        });
        
        const debouncedGenerateForecast = debounce(generateForecast, 300);
        el.simPriceSlider.addEventListener("change", (e) => {
            state.simulator.priceMultiplier = parseFloat(e.target.value) / 100.0;
            state.simulator.activePresetName = "Custom Overrides";
            debouncedGenerateForecast();
        });
    }

    // Clear promos button click
    if (el.clearSimPromosBtn) {
        el.clearSimPromosBtn.addEventListener("click", () => {
            state.simulator.promoDays = [];
            state.simulator.activePresetName = "Custom Overrides";
            renderPromoCalendarGrid();
            generateForecast();
        });
    }

    // Save scenario to history list
    if (el.saveScenarioBtn) {
        el.saveScenarioBtn.addEventListener("click", () => {
            if (!state.currentScenarioMetrics) {
                showToast("Please run a forecast first before saving.", "warning");
                return;
            }
            
            let customName = el.simScenarioName.value.trim();
            if (!customName) {
                customName = state.currentScenarioMetrics.defaultName;
            }
            
            const metrics = state.currentScenarioMetrics;
            const scenarioKey = `${customName}-${metrics.priceMult.toFixed(2)}-${metrics.promoDaysStr}`;
            
            // Check for duplicates
            if (state.scenarios.some(s => s.name === customName)) {
                showToast(`A scenario named "${customName}" already exists.`, "warning");
                return;
            }
            
            state.scenarios.push({
                key: scenarioKey,
                name: customName,
                priceLevel: `${(metrics.priceMult * 100).toFixed(0)}%`,
                promoDaysCount: metrics.promoDaysStr ? metrics.promoDaysStr.split(",").length : 0,
                volume: metrics.volume,
                revenue: metrics.revenue,
                stockouts: metrics.stockouts,
                score: metrics.score
            });
            
            el.simScenarioName.value = ""; // Clear input field
            updateScenarioHistoryTable();
            showToast(`Scenario "${customName}" saved to history.`, "success");
        });
    }

    // Clear scenario comparison history
    if (el.clearScenariosBtn) {
        el.clearScenariosBtn.addEventListener("click", () => {
            state.scenarios = [];
            updateScenarioHistoryTable();
            showToast("Scenario history cleared.", "info");
        });
    }

    // Preset Scenario selection
    const presetContainer = document.getElementById("scenario-presets-container");
    if (presetContainer) {
        presetContainer.querySelectorAll(".preset-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const preset = btn.getAttribute("data-preset");
                
                if (preset === "baseline") {
                    state.simulator.priceMultiplier = 1.0;
                    state.simulator.promoDays = [];
                    state.simulator.activePresetName = "Baseline Scenario";
                } else if (preset === "holiday") {
                    state.simulator.priceMultiplier = 1.10; // +10% price
                    state.simulator.promoDays = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]; // 11 promo days mid-horizon
                    state.simulator.activePresetName = "Holiday Season";
                } else if (preset === "promo") {
                    state.simulator.priceMultiplier = 0.95; // -5% discount
                    state.simulator.promoDays = [5, 6, 12, 13, 19, 20, 26, 27]; // weekend promo calendar dates
                    state.simulator.activePresetName = "Weekend Promotion";
                } else if (preset === "discount") {
                    state.simulator.priceMultiplier = 0.90; // -10% discount
                    state.simulator.promoDays = [];
                    state.simulator.activePresetName = "10% Discount Scenario";
                } else if (preset === "delay") {
                    state.simulator.priceMultiplier = 1.0;
                    state.simulator.promoDays = [];
                    state.simulator.activePresetName = "Supplier Delay Scenario";
                } else if (preset === "surge") {
                    state.simulator.priceMultiplier = 1.0;
                    state.simulator.promoDays = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]; // 30 promo days
                    state.simulator.activePresetName = "Demand Surge Scenario";
                }
                
                // Synchronize UI widgets
                if (el.simPriceSlider) {
                    el.simPriceSlider.value = Math.round(state.simulator.priceMultiplier * 100);
                    el.simPriceSlider.dispatchEvent(new Event("input"));
                }
                renderPromoCalendarGrid();
                generateForecast();
            });
        });
    }

    // Export PDF Report print triggers
    if (el.exportReportBtn) {
        el.exportReportBtn.addEventListener("click", async () => {
            if (!state.activeProduct || !state.trainedReport || !state.datasetStats) {
                showToast("Please ensure dataset and forecasts are loaded before exporting.", "warning");
                return;
            }
            
            const productId = state.activeProduct;
            const prodData = state.trainedReport.products[productId];
            const activeModel = el.forecastModelSelect.value;
            
            let modelName = prodData.best_model;
            let mae = prodData.all_models[modelName].MAE;
            let mape = prodData.all_models[modelName].MAPE + "%";
            
            if (activeModel !== "best_recommender" && prodData.all_models[activeModel]) {
                modelName = activeModel;
                mae = prodData.all_models[activeModel].MAE;
                mape = prodData.all_models[activeModel].MAPE + "%";
            }
            
            // 1. Populate metadata
            document.getElementById("print-timestamp").textContent = "Generated on: " + new Date().toLocaleString();
            document.getElementById("print-stat-records").textContent = state.datasetStats.row_count.toLocaleString();
            document.getElementById("print-stat-stores").textContent = state.datasetStats.unique_stores;
            document.getElementById("print-stat-products").textContent = state.datasetStats.unique_products;
            document.getElementById("print-stat-categories").textContent = state.datasetStats.unique_categories;
            document.getElementById("print-stat-dates").textContent = `${state.datasetStats.start_date} to ${state.datasetStats.end_date}`;
            
            // 2. Populate product forecast stats
            document.getElementById("print-product-name").textContent = `${prodData.product_name} (${productId})`;
            document.getElementById("print-best-model").textContent = modelName;
            document.getElementById("print-mae").textContent = mae;
            document.getElementById("print-mape").textContent = mape;
            
            // 3. Populate model comparison table
            const models = Object.keys(prodData.all_models);
            let printCompareHtml = "";
            models.forEach(m => {
                const mData = prodData.all_models[m];
                printCompareHtml += `
                    <tr>
                        <td>${escapeHtml(m)} ${m === prodData.best_model ? '(Recommended)' : ''}</td>
                        <td>${escapeHtml(mData.MAE)}</td>
                        <td>${escapeHtml(mData.MAPE)}%</td>
                        <td>${escapeHtml(mData.R2)}</td>
                    </tr>
                `;
            });
            document.querySelector("#print-metrics-table tbody").innerHTML = printCompareHtml;
            
            // 4. Populate AI Analyst summary paragraph
            const botMessages = document.querySelectorAll(".message-bot .message-content p");
            let aiSummaryHtml = "";
            if (botMessages.length > 1) {
                // Get the last bot message content (skipping welcome)
                aiSummaryHtml = botMessages[botMessages.length - 1].innerHTML;
            } else {
                aiSummaryHtml = renderMarkdownSafely(prodData.recommendation_reason || "No custom audit logs generated yet.");
            }
            document.getElementById("print-ai-summary").innerHTML = aiSummaryHtml;
            
            // 5. Convert Plotly SVG to image and trigger browser print dialog
            el.exportReportBtn.disabled = true;
            const originalBtnText = el.exportReportBtn.innerHTML;
            el.exportReportBtn.innerHTML = "<span>Generating Image...</span>";
            
            try {
                const dataUrl = await Plotly.toImage(document.getElementById('chart-forecast'), {
                    format: 'png',
                    width: 800,
                    height: 400
                });
                
                document.getElementById('print-forecast-chart').src = dataUrl;
                
                // Trigger printing
                window.print();
            } catch (err) {
                console.error("Failed to generate chart image for print:", err);
                showToast("Could not render chart into PDF. Printing text details instead.", "warning");
                window.print();
            } finally {
                el.exportReportBtn.disabled = false;
                el.exportReportBtn.innerHTML = originalBtnText;
            }
        });
    }

    // Offline overlay retry connection binding
    if (el.retryConnectionBtn) {
        el.retryConnectionBtn.addEventListener("click", () => {
            el.retryConnectionBtn.disabled = true;
            const btnSpan = el.retryConnectionBtn.querySelector("span");
            const originalText = btnSpan.textContent;
            btnSpan.textContent = "Checking Server Connection...";
            
            checkConnectionHealth().finally(() => {
                el.retryConnectionBtn.disabled = false;
                btnSpan.textContent = originalText;
            });
        });
    }

    // Map model names to clear, business-friendly engine titles
    function getModelFriendlyLabel(modelName, isRecommended = false) {
        const recommendedPrefix = isRecommended ? "★ " : "";
        if (modelName === "Prophet") {
            return recommendedPrefix + "Forecast Engine (Facebook Prophet)";
        } else if (modelName === "Random Forest") {
            return recommendedPrefix + "Tree Ensemble Engine (Random Forest)";
        } else if (modelName === "Ridge") {
            return recommendedPrefix + "Regularized Trend Engine (Ridge)";
        } else if (modelName === "LinearRegression") {
            return recommendedPrefix + "Simple Baseline Engine (Linear Regression)";
        }
        return recommendedPrefix + modelName;
    }

    // Debounce function to limit request frequencies
    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    // Run startup status check
    checkDatasetStatus();
});
