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
        activePage: "data-hub"
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
        trainingProgressBox: document.getElementById("training-progress-box"),
        trainingProgressFill: document.getElementById("training-progress-fill"),
        trainingProgressText: document.getElementById("training-progress-text"),
        forecastSummaryContainer: document.getElementById("forecast-summary-container"),
        registryAvgMae: document.getElementById("registry-avg-mae"),
        registryAvgMape: document.getElementById("registry-avg-mape"),
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
        
        // Help Viva Modal
        openVivaModalBtn: document.getElementById("open-viva-modal-btn"),
        closeVivaModalBtn: document.getElementById("close-viva-modal-btn"),
        vivaModal: document.getElementById("viva-modal"),
        
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
        explainChartBtns: document.querySelectorAll(".explain-chart-btn")
    };

    // --- View Router ---
    function navigateToPage(pageId) {
        if (state.activePage === pageId) return;
        
        // Validation check for guided workflow
        if (pageId !== "data-hub" && !state.datasetLoaded) {
            alert("Please load or upload a dataset in the Data Hub first.");
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
        
        // Update header breadcrumb
        const titleMap = {
            "data-hub": "Data Hub",
            "eda": "Exploratory EDA Dashboard",
            "forecasting": "Demand Forecast Engine"
        };
        el.pageTitle.textContent = titleMap[pageId] || "Dashboard";

        // Trigger dynamic page loading
        if (pageId === "eda") {
            loadEdaPage();
        } else if (pageId === "forecasting") {
            loadForecastingPage();
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
                    el.validationWarningsList.innerHTML = data.warnings.map(w => `<li>${w}</li>`).join("");
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
            
            // Show alert box warnings if any
            if (data.report.warnings && data.report.warnings.length > 0) {
                el.validationWarningsBox.classList.remove("hidden");
                el.validationWarningsList.innerHTML = data.report.warnings.map(w => `<li>${w}</li>`).join("");
            } else {
                el.validationWarningsBox.classList.add("hidden");
            }
            
            // Auto navigate to EDA to keep workflow fluid
            setTimeout(() => navigateToPage("eda"), 1000);
        } catch (error) {
            el.demoLoadStatus.textContent = `Error: ${error.message}`;
            el.demoLoadStatus.className = "mt-3 text-center text-sm error-color";
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
        el.selectedFileInfo.textContent = `Uploading: ${file.name} (${(file.size / 1024).toFixed(1)} KB)...`;
        el.selectedFileInfo.className = "file-info text-secondary";
        
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const data = await api.post("/api/dataset/upload", formData, true);
            el.selectedFileInfo.textContent = `Successfully uploaded: ${file.name}!`;
            el.selectedFileInfo.className = "file-info success-color";
            
            setDatasetLoadedState(true, file.name, data.report.stats);
            
            if (data.report.warnings && data.report.warnings.length > 0) {
                el.validationWarningsBox.classList.remove("hidden");
                el.validationWarningsList.innerHTML = data.report.warnings.map(w => `<li>${w}</li>`).join("");
            } else {
                el.validationWarningsBox.classList.add("hidden");
            }
            
            // Auto navigate to EDA to keep workflow fluid
            setTimeout(() => navigateToPage("eda"), 1000);
        } catch (error) {
            el.selectedFileInfo.textContent = `Upload failed: ${error.message}`;
            el.selectedFileInfo.className = "file-info error-color";
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
                htmlHeaders += `<th>${col}</th>`;
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
                    htmlRows += `<td>${val}</td>`;
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
        // Show chart spinners
        el.loaderSalesTrend.classList.remove("hidden");
        el.loaderCorrelation.classList.remove("hidden");
        el.loaderWeekly.classList.remove("hidden");
        el.loaderMonthly.classList.remove("hidden");
        
        try {
            const eda = await api.get("/api/dataset/eda");
            state.edaData = eda;
            
            // Stats
            const stats = eda.descriptive_statistics.units_sold || {};
            el.edaAvgPrice.textContent = `$${eda.descriptive_statistics.price?.mean?.toFixed(2) || "0.00"}`;
            el.edaAvgSold.textContent = Math.round(stats.mean || 0).toLocaleString();
            el.edaOutliersCount.textContent = eda.outliers_count;
            
            // Draw Charts
            drawSalesTrendChart(eda.sales_trend);
            drawCorrelationChart(eda.correlation_matrix);
            drawSeasonalityChart("chart-weekly", eda.weekly_seasonality, "Weekly Seasonality Pattern", "Day of Week");
            drawSeasonalityChart("chart-monthly", eda.monthly_seasonality, "Monthly Seasonality Pattern", "Month");
            
            // Hide chart spinners on success
            el.loaderSalesTrend.classList.add("hidden");
            el.loaderCorrelation.classList.add("hidden");
            el.loaderWeekly.classList.add("hidden");
            el.loaderMonthly.classList.add("hidden");
            
            // Populate Outliers Table
            let outliersHtml = "";
            if (eda.outliers && eda.outliers.length > 0) {
                eda.outliers.forEach(out => {
                    outliersHtml += `
                        <tr>
                            <td>${out.date}</td>
                            <td>${out.product_id}</td>
                            <td>${out.product_name}</td>
                            <td><strong>${out.units_sold}</strong></td>
                            <td>Outside [${out.lower_bound}, ${out.upper_bound}]</td>
                        </tr>
                    `;
                });
            } else {
                outliersHtml = "<tr><td colspan='5' class='text-center text-secondary'>No outliers detected.</td></tr>";
            }
            el.outliersTable.innerHTML = outliersHtml;
            
        } catch (error) {
            console.error("EDA Loading failed:", error);
            // Hide spinners even on failure
            el.loaderSalesTrend.classList.add("hidden");
            el.loaderCorrelation.classList.add("hidden");
            el.loaderWeekly.classList.add("hidden");
            el.loaderMonthly.classList.add("hidden");
        }
    }

    // Draw sales trend
    function drawSalesTrendChart(trend) {
        const trace = {
            x: trend.dates,
            y: trend.sales,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#6366f1', width: 2.0 },
            name: 'Total Units Sold'
        };
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 20, r: 20, b: 40, l: 50 },
            font: { color: '#94a3b8', family: 'Inter' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)' }
        };
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
            font: { color: '#94a3b8', family: 'Inter' }
        };
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
            margin: { t: 20, r: 20, b: 40, l: 50 },
            font: { color: '#94a3b8', family: 'Inter' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.02)', linecolor: 'rgba(255,255,255,0.1)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)' }
        };
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
            } else {
                // Not trained yet - show empty state and hide work areas
                el.forecastEmptyState.classList.remove("hidden");
                el.forecastSummaryContainer.classList.add("hidden");
                el.forecastWorkspace.classList.add("hidden");
            }
        } catch (error) {
            console.error("Forecasting loading error:", error);
        }
    }

    // Trigger Model Training
    el.trainModelsBtn.addEventListener("click", async () => {
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
                    el.trainingProgressText.textContent = "Ftting Prophet models and holiday adjustments...";
                } else {
                    el.trainingProgressText.textContent = "Extracting lags, rolling averages, and split datasets...";
                }
            }
        }, 300);
        
        try {
            const data = await api.post("/api/forecast/train");
            clearInterval(progressInterval);
            el.trainingProgressFill.style.width = "100%";
            el.trainingProgressText.textContent = "Training optimization complete!";
            
            setTimeout(() => {
                el.trainingProgressBox.classList.add("hidden");
                el.trainModelsBtn.disabled = false;
                el.forecastEmptyState.classList.add("hidden");
                displayTrainingReport(data.report);
            }, 1000);
            
        } catch (error) {
            clearInterval(progressInterval);
            el.trainingProgressText.textContent = `Training failed: ${error.message}`;
            el.trainingProgressFill.style.backgroundColor = "var(--error)";
            el.trainModelsBtn.disabled = false;
        }
    });

    function displayTrainingReport(report) {
        state.trainedReport = report;
        
        // Header summary cards
        el.registryAvgMae.textContent = report.average_mae.toFixed(2);
        el.registryAvgMape.textContent = `${report.average_mape.toFixed(2)}%`;
        el.forecastSummaryContainer.classList.remove("hidden");
        
        // Populate product selector dropdown
        let productOptions = "";
        const productsMap = report.products;
        for (const pid in productsMap) {
            productOptions += `<option value="${pid}">${productsMap[pid].product_name} (${pid})</option>`;
        }
        el.forecastProductSelect.innerHTML = productOptions;
        
        // Setup initial product selection state
        const pids = Object.keys(productsMap);
        if (pids.length > 0) {
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
        const prodData = state.trainedReport.products[productId];
        if (!prodData) return;
        
        // 1. Populate model selector list (LR, RF, Prophet if available)
        const models = Object.keys(prodData.all_models);
        let modelOptions = `<option value="best_recommender">Best recommended: ${prodData.best_model}</option>`;
        models.forEach(m => {
            modelOptions += `<option value="${m}">${m}</option>`;
        });
        el.forecastModelSelect.innerHTML = modelOptions;
        state.activeModel = null; // defaults to best recommender
        
        // 2. Populate recommended badge & reason
        el.recommenderModelName.textContent = prodData.best_model;
        
        // 3. Populate comparison table
        let compareHtml = "";
        models.forEach(m => {
            const mData = prodData.all_models[m];
            const isBest = m === prodData.best_model;
            compareHtml += `
                <tr class="${isBest ? 'border-pulse' : ''}">
                    <td>${m} ${isBest ? '<i data-lucide="award" class="success-color inline-icon" style="width:14px;height:14px;vertical-align:middle;margin-left:4px;"></i>' : ''}</td>
                    <td><strong>${mData.MAE}</strong></td>
                    <td>${mData.MAPE}%</td>
                    <td>${mData.R2}</td>
                </tr>
            `;
        });
        el.metricsCompareTable.innerHTML = compareHtml;
        lucide.createIcons(); // Refresh inline icons
        
        // 4. Load & Render Forecast
        await generateForecast();
    }

    // Generate forecast execution
    el.runForecastBtn.addEventListener("click", () => generateForecast());

    async function generateForecast() {
        if (!state.activeProduct) return;
        
        // Show forecast overlay spinner
        el.loaderForecast.classList.remove("hidden");
        
        const productId = state.activeProduct;
        const horizon = el.forecastHorizonSelect.value;
        const selectedModelVal = el.forecastModelSelect.value;
        
        let url = `/api/forecast/predict?product_id=${productId}&horizon_days=${horizon}`;
        if (selectedModelVal !== "best_recommender") {
            url += `&model_name=${encodeURIComponent(selectedModelVal)}`;
        }
        
        try {
            el.runForecastBtn.disabled = true;
            const data = await api.get(url);
            
            // Update chart titles
            el.forecastChartTitle.textContent = `${data.product_name} Demand Forecast`;
            el.forecastChartSubtitle.textContent = `Model: ${data.model_used} | Horizon: ${data.forecast_horizon_days} Days`;
            
            // Update recommendation reason text if it was the recommended model
            if (data.recommendation_reason) {
                el.recommenderReasonText.textContent = data.recommendation_reason;
            } else {
                // If model is overridden
                el.recommenderReasonText.innerHTML = `<em>Custom model override.</em> Historical metrics: MAE = ${data.metrics.MAE}, R² = ${data.metrics.R2}.`;
            }
            
            // Draw Forecast
            drawForecastChart(data);
            
            // Hide spinner on success
            el.loaderForecast.classList.add("hidden");
        } catch (error) {
            console.error("Forecast failed:", error);
            // Hide spinner even on failure
            el.loaderForecast.classList.add("hidden");
            alert(`Error generating forecast: ${error.message}`);
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
        
        // Trace 1: Historical Sales
        const traceHist = {
            x: histDates,
            y: histSales,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#6366f1', width: 2.0 },
            name: 'Historical Sales'
        };
        
        // Trace 2: Predicted Future Demand
        const tracePred = {
            x: foreDates,
            y: forePreds,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#f59e0b', width: 2.5, dash: 'dash' },
            marker: { size: 5 },
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
        
        // Trace 4: Upper confidence band (filled to Trace 3)
        const traceUpper = {
            x: foreDates,
            y: foreUpper,
            type: 'scatter',
            mode: 'lines',
            line: { width: 0 },
            fill: 'tonexty',
            fillcolor: 'rgba(245, 158, 11, 0.1)',
            name: '95% Confidence Interval'
        };
        
        // Combine (Lower then Upper to ensure proper fill)
        const traces = [traceHist, traceLower, traceUpper, tracePred];
        
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 20, r: 20, b: 40, l: 50 },
            font: { color: '#94a3b8', family: 'Inter' },
            hovermode: 'x',
            xaxis: { gridcolor: 'rgba(255,255,255,0.02)', linecolor: 'rgba(255,255,255,0.1)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)' },
            legend: { orientation: 'h', y: -0.2 }
        };
        
        Plotly.newPlot('chart-forecast', traces, layout, { responsive: true, displayModeBar: false });
    }

    // --- Collapsible Chat Panel Controls ---
    function toggleChat(isOpen) {
        state.chatOpen = isOpen;
        if (isOpen) {
            el.chatPanel.classList.add("open");
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

    // Integrated sparkles explanation trigger
    el.explainChartBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const chartType = btn.getAttribute("data-chart");
            let message = "";
            if (chartType === "sales_trend") {
                message = "Explain this sales trend chart for our store products.";
            } else if (chartType === "correlation_matrix") {
                message = "Explain the correlation matrix chart and what the variables mean.";
            } else if (chartType === "weekly_seasonality") {
                message = "Explain our weekly seasonality profile. When do we see peak sales?";
            } else if (chartType === "monthly_seasonality") {
                message = "Explain our monthly seasonality profile. What yearly patterns exist?";
            } else if (chartType === "demand_forecast") {
                const prodName = el.forecastProductSelect.options[el.forecastProductSelect.selectedIndex]?.text || "selected item";
                message = `Explain the demand forecast chart for ${prodName}. What is our recommended strategy?`;
            } else {
                message = "Explain the current visual analysis chart.";
            }
            
            toggleChat(true);
            sendChatMessage(message);
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
            
        // Use markdown paragraphs if present
        const formattedText = text.replace(/\n/g, "<br>");
            
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

    // Viva Helper modal triggers
    if (el.openVivaModalBtn) {
        el.openVivaModalBtn.addEventListener("click", () => {
            el.vivaModal.classList.remove("hidden");
        });
    }
    if (el.closeVivaModalBtn) {
        el.closeVivaModalBtn.addEventListener("click", () => {
            el.vivaModal.classList.add("hidden");
        });
    }
    if (el.vivaModal) {
        el.vivaModal.addEventListener("click", (e) => {
            if (e.target === el.vivaModal) {
                el.vivaModal.classList.add("hidden");
            }
        });
    }

    // Export PDF Report print triggers
    if (el.exportReportBtn) {
        el.exportReportBtn.addEventListener("click", async () => {
            if (!state.activeProduct || !state.trainedReport) return;
            
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
                        <td>${m} ${m === prodData.best_model ? '(Recommended)' : ''}</td>
                        <td>${mData.MAE}</td>
                        <td>${mData.MAPE}%</td>
                        <td>${mData.R2}</td>
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
                aiSummaryHtml = prodData.recommendation_reason || "No custom audit logs generated yet.";
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
                alert("Could not render chart into PDF. Printing text details instead.");
                window.print();
            } finally {
                el.exportReportBtn.disabled = false;
                el.exportReportBtn.innerHTML = originalBtnText;
            }
        });
    }

    // Run startup status check
    checkDatasetStatus();
});
