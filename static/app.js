// BioAI Insight Engine - Frontend App Controller
document.addEventListener("DOMContentLoaded", () => {
    // API base configuration (relative since we host it from FastAPI)
    const API_BASE = "";
    
    // Charts cache
    let modelChart = null;
    let globalShapChart = null;
    
    // Track current tab
    let activeTab = "patient-predict";
    
    // Initialize
    initNavigation();
    initPatientPredictor();
    initModelAnalytics();
    initRagSearch();
    initBatchInference();
    checkApiHealth();
    
    // Periodically check health (every 10 seconds)
    setInterval(checkApiHealth, 10000);

    // ==========================================
    // 1. Navigation Controller
    // ==========================================
    function initNavigation() {
        const navItems = document.querySelectorAll(".nav-item");
        const tabs = document.querySelectorAll(".tab-content");
        const headerTitle = document.getElementById("current-tab-title");
        const headerDesc = document.querySelector(".dashboard-header p");

        const tabMeta = {
            "patient-predict": {
                title: "Patient Diagnostics Dashboard",
                desc: "Clinical breast cancer risk classification with SHAP local explainability and feature attributions."
            },
            "model-metrics": {
                title: "Model Analytics & Engine Tuning",
                desc: "Production metrics comparison across Logistic Regression, SVM, and Random Forest classifiers."
            },
            "rag-search": {
                title: "Biomedical Literature Search (RAG)",
                desc: "Semantic retrieval across 500+ research abstracts and clinical guidelines using FAISS."
            },
            "batch-inference": {
                title: "Batch Diagnostics Pipeline",
                desc: "Upload bulk clinical data via CSV to run concurrent inference on multiple patient records."
            }
        };

        navItems.forEach(item => {
            item.addEventListener("click", (e) => {
                e.preventDefault();
                const targetTab = item.getAttribute("data-tab");
                if (targetTab === activeTab) return;
                
                activeTab = targetTab;
                
                // Toggle nav active state
                navItems.forEach(nav => nav.classList.remove("active"));
                item.classList.add("active");
                
                // Toggle tab content visibility
                tabs.forEach(tab => tab.classList.remove("active"));
                document.getElementById(`${targetTab}-tab`).classList.add("active");
                
                // Update header titles
                headerTitle.textContent = tabMeta[targetTab].title;
                headerDesc.textContent = tabMeta[targetTab].desc;
                
                // Trigger chart rendering or resource fetches if entering tabs
                if (targetTab === "model-metrics") {
                    loadAndRenderMetrics();
                }
            });
        });
    }

    // ==========================================
    // API Health & Latency Monitor
    // ==========================================
    async function checkApiHealth() {
        const startTime = Date.now();
        const statusDot = document.querySelector(".status-dot");
        const statusText = document.getElementById("api-status-text");
        const latencyBadge = document.getElementById("api-latency");
        
        try {
            const response = await fetch(`${API_BASE}/health`);
            if (response.ok) {
                const data = await response.json();
                const latency = Date.now() - startTime;
                
                statusDot.className = "status-dot online";
                statusText.textContent = "API Online";
                latencyBadge.textContent = `${latency} ms latency`;
                
                // If we are in model metrics, update with the stats
                if (activeTab === "model-metrics" && !modelChart) {
                    loadAndRenderMetrics();
                }
            } else {
                throw new Error("API not healthy");
            }
        } catch (error) {
            statusDot.className = "status-dot offline";
            statusText.textContent = "API Offline";
            latencyBadge.textContent = "-- ms latency";
        }
    }

    // ==========================================
    // 2. Patient Diagnostics (Tab 1)
    // ==========================================
    function initPatientPredictor() {
        const sliders = document.querySelectorAll(".range-slider");
        
        // Update slider value indicators in UI and trigger predict
        sliders.forEach(slider => {
            const id = slider.id;
            const displayVal = document.getElementById(`val-${id}`);
            
            slider.addEventListener("input", (e) => {
                displayVal.textContent = e.target.value;
                debouncePredict();
            });
        });

        // Run initial prediction on load
        runPatientPrediction();
    }

    let predictTimeout;
    function debouncePredict() {
        clearTimeout(predictTimeout);
        predictTimeout = setTimeout(runPatientPrediction, 150);
    }

    async function runPatientPrediction() {
        const form = document.getElementById("predict-form");
        const payload = {
            "Cl.thickness": parseFloat(form.cl_thickness.value),
            "Cell.size": parseFloat(form.cell_size.value),
            "Cell.shape": parseFloat(form.cell_shape.value),
            "Marg.adhesion": parseFloat(form.marg_adhesion.value),
            "Epith.c.size": parseFloat(form.epith_c_size.value),
            "Bare.nuclei": parseFloat(form.bare_nuclei.value),
            "Bl.cromatin": parseFloat(form.bl_cromatin.value),
            "Normal.nucleoli": parseFloat(form.normal_nucleoli.value),
            "Mitoses": parseFloat(form.mitoses.value)
        };

        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Prediction request failed");
            
            const data = await response.json();
            renderPredictionResults(data);
            
        } catch (error) {
            console.error("Prediction error:", error);
        }
    }

    function renderPredictionResults(data) {
        const riskBadge = document.getElementById("risk-badge");
        const probText = document.getElementById("result-probability");
        const riskDesc = document.getElementById("risk-explanation");
        const gaugeStroke = document.getElementById("gauge-stroke");
        
        const probability = data.confidence_score;
        const probPct = (probability * 100).toFixed(1);
        probText.textContent = `${probPct}%`;

        // SVG gauge circumference is 2 * PI * r = 2 * 3.1416 * 50 = 314.16
        const offset = 314.16 - (probability * 314.16);
        gaugeStroke.style.strokeDashoffset = offset;

        if (data.prediction === 1) {
            riskBadge.textContent = "High Risk";
            riskBadge.className = "badge high-risk";
            gaugeStroke.style.stroke = "var(--color-high-risk)";
            riskDesc.textContent = `High risk of breast cancer malignancy detected (${probPct}% confidence). Clinical verification recommended.`;
        } else {
            riskBadge.textContent = "Low Risk";
            riskBadge.className = "badge low-risk";
            gaugeStroke.style.stroke = "var(--color-low-risk)";
            riskDesc.textContent = `Healthy profile detected (${(100 - probPct).toFixed(1)}% benign confidence). Standard screening advised.`;
        }

        // Update individual model breakdowns
        const probs = data.model_probabilities;
        if (probs) {
            document.getElementById("model-rf-bar").style.width = `${probs.random_forest * 100}%`;
            document.getElementById("model-rf-val").textContent = `${(probs.random_forest * 100).toFixed(1)}%`;
            
            document.getElementById("model-lr-bar").style.width = `${probs.logistic_regression * 100}%`;
            document.getElementById("model-lr-val").textContent = `${(probs.logistic_regression * 100).toFixed(1)}%`;
            
            document.getElementById("model-svm-bar").style.width = `${probs.svm * 100}%`;
            document.getElementById("model-svm-val").textContent = `${(probs.svm * 100).toFixed(1)}%`;
        }

        // Render local SHAP attributions
        renderLocalShap(data.shap_attributions, data.shap_base_value);
    }

    function renderLocalShap(attributions, baseValue) {
        const container = document.getElementById("shap-cont");
        
        // Remove existing items (except axis line)
        const items = container.querySelectorAll(".shap-bar-item, .shap-base-info");
        items.forEach(el => el.remove());

        if (!attributions || attributions.length === 0) {
            const noExplain = document.createElement("div");
            noExplain.className = "shap-base-info";
            noExplain.textContent = "SHAP local explanation not available for this model configuration.";
            container.appendChild(noExplain);
            return;
        }

        // Find max absolute value to scale bars relative to 100% of container side
        const maxVal = Math.max(...attributions.map(a => Math.abs(a.shap_value)), 0.05);

        // Map feature codes to pretty names
        const prettyNames = {
            "Cl.thickness": "Clump Thickness",
            "Cell.size": "Cell Size Uniformity",
            "Cell.shape": "Cell Shape Uniformity",
            "Marg.adhesion": "Marginal Adhesion",
            "Epith.c.size": "Epithelial Cell Size",
            "Bare.nuclei": "Bare Nuclei",
            "Bl.cromatin": "Bland Chromatin",
            "Normal.nucleoli": "Normal Nucleoli",
            "Mitoses": "Mitoses"
        };

        attributions.forEach(attr => {
            const barItem = document.createElement("div");
            barItem.className = "shap-bar-item";

            const featLabel = document.createElement("div");
            featLabel.className = "shap-feat-name";
            featLabel.textContent = prettyNames[attr.feature] || attr.feature;
            barItem.appendChild(featLabel);

            const barWrapper = document.createElement("div");
            barWrapper.className = "shap-bar-wrapper";

            const bar = document.createElement("div");
            
            // Calculate percentage width (max absolute value represents 45% width to leave padding)
            const pctWidth = (Math.abs(attr.shap_value) / maxVal) * 45;
            bar.style.width = `${pctWidth}%`;

            if (attr.shap_value >= 0) {
                bar.className = "shap-bar positive";
                bar.style.left = "50%";
                
                const valLabel = document.createElement("div");
                valLabel.className = "shap-bar-val";
                valLabel.style.left = `calc(50% + ${pctWidth}% + 6px)`;
                valLabel.textContent = `+${attr.shap_value.toFixed(2)}`;
                barItem.appendChild(valLabel);
            } else {
                bar.className = "shap-bar negative";
                // for negative values, left should be 50% - width%
                bar.style.left = `${50 - pctWidth}%`;
                
                const valLabel = document.createElement("div");
                valLabel.className = "shap-bar-val";
                valLabel.style.right = `calc(50% + ${pctWidth}% + 6px)`;
                valLabel.textContent = `${attr.shap_value.toFixed(2)}`;
                barItem.appendChild(valLabel);
            }

            barWrapper.appendChild(bar);
            barItem.appendChild(barWrapper);
            container.appendChild(barItem);
        });

        // Add base value indicator footer
        const baseFooter = document.createElement("div");
        baseFooter.className = "shap-base-info";
        baseFooter.innerHTML = `<i class="fa-solid fa-calculator"></i> Base Value (Prior Probability): <strong>${(baseValue * 100).toFixed(1)}%</strong>`;
        container.appendChild(baseFooter);
    }

    // ==========================================
    // 3. Model Analytics (Tab 2)
    // ==========================================
    function initModelAnalytics() {
        const retrainBtn = document.getElementById("btn-retrain");
        
        retrainBtn.addEventListener("click", async () => {
            const seedVal = parseInt(document.getElementById("training-seed").value) || 48;
            const statusEl = document.getElementById("retrain-output");
            
            statusEl.style.display = "flex";
            statusEl.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin text-accent"></i> Retraining models on backend...`;
            retrainBtn.disabled = true;
            
            try {
                const response = await fetch(`${API_BASE}/train`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ seed: seedVal })
                });
                
                if (!response.ok) throw new Error("Training request failed");
                const data = await response.json();
                
                statusEl.innerHTML = `<span class="text-accent"><i class="fa-solid fa-circle-check text-low-risk"></i> Success! Seed ${data.seed_used} training completed.</span>`;
                
                // Re-render prediction and metrics dashboard
                runPatientPrediction();
                loadAndRenderMetrics(true);
                
            } catch (err) {
                statusEl.innerHTML = `<span class="text-high-risk"><i class="fa-solid fa-circle-xmark"></i> Failed: ${err.message}</span>`;
            } finally {
                retrainBtn.disabled = false;
                setTimeout(() => {
                    statusEl.style.display = "none";
                }, 8000);
            }
        });
    }

    async function loadAndRenderMetrics(forceReload = false) {
        try {
            const response = await fetch(`${API_BASE}/health`);
            if (!response.ok) throw new Error("Could not retrieve health logs");
            
            const data = await response.json();
            const perf = data.model_performance;
            
            if (!perf || Object.keys(perf).length === 0) return;
            
            renderModelPerformanceCards(perf);
            renderComparisonChart(perf);
            loadGlobalShapImportance();
            
        } catch (error) {
            console.error("Metrics render error:", error);
        }
    }

    function renderModelPerformanceCards(perf) {
        const container = document.getElementById("metrics-cards-container");
        container.innerHTML = "";
        
        const friendlyName = {
            "logistic_regression": "Logistic Regression",
            "svm": "SVM (RBF Kernel)",
            "random_forest": "Random Forest"
        };
        
        Object.entries(perf).forEach(([modelKey, metrics]) => {
            const card = document.createElement("div");
            card.className = "card";
            card.style.padding = "16px";
            card.innerHTML = `
                <h4 style="font-size: 0.95rem; margin-bottom: 12px; color: var(--text-primary); border-bottom: 1px solid var(--border-color); padding-bottom: 8px;">
                    ${friendlyName[modelKey] || modelKey}
                </h4>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                    <div style="background: rgba(255,255,255,0.02); padding: 8px; border-radius: 6px; text-align: center;">
                        <small style="font-size: 0.65rem; color: var(--text-secondary); display: block;">Accuracy</small>
                        <strong style="color: var(--color-accent); font-size: 1.05rem;">${(metrics.accuracy * 100).toFixed(1)}%</strong>
                    </div>
                    <div style="background: rgba(255,255,255,0.02); padding: 8px; border-radius: 6px; text-align: center;">
                        <small style="font-size: 0.65rem; color: var(--text-secondary); display: block;">ROC-AUC</small>
                        <strong style="color: var(--color-purple); font-size: 1.05rem;">${metrics.roc_auc.toFixed(3)}</strong>
                    </div>
                    <div style="background: rgba(255,255,255,0.02); padding: 8px; border-radius: 6px; text-align: center;">
                        <small style="font-size: 0.65rem; color: var(--text-secondary); display: block;">F1-Score</small>
                        <strong style="color: var(--color-low-risk); font-size: 1.05rem;">${metrics.f1_score.toFixed(3)}</strong>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });
    }

    function renderComparisonChart(perf) {
        const ctx = document.getElementById("model-comparison-chart").getContext("2d");
        
        const models = Object.keys(perf);
        const accuracies = models.map(m => perf[m].accuracy * 100);
        const aucs = models.map(m => perf[m].roc_auc * 100);
        
        const labels = models.map(m => {
            if (m === "logistic_regression") return "Logistic Regression";
            if (m === "svm") return "SVM (RBF)";
            if (m === "random_forest") return "Random Forest";
            return m;
        });

        if (modelChart) {
            modelChart.destroy();
        }

        modelChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Accuracy (%)',
                        data: accuracies,
                        backgroundColor: 'rgba(0, 242, 254, 0.65)',
                        borderColor: 'var(--color-accent)',
                        borderWidth: 1.5,
                        borderRadius: 4
                    },
                    {
                        label: 'ROC-AUC (%)',
                        data: aucs,
                        backgroundColor: 'rgba(168, 85, 247, 0.65)',
                        borderColor: 'var(--color-purple)',
                        borderWidth: 1.5,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 80,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'var(--text-secondary)' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: 'var(--text-secondary)' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: 'var(--text-primary)', font: { family: 'Outfit' } }
                    }
                }
            }
        });
    }

    async function loadGlobalShapImportance() {
        try {
            const response = await fetch(`${API_BASE}/explain/global?model_name=random_forest`);
            if (!response.ok) throw new Error("Could not retrieve global importance");
            const data = await response.json();
            
            renderGlobalShapChart(data);
        } catch (error) {
            console.error("Global SHAP load error:", error);
        }
    }

    function renderGlobalShapChart(importanceData) {
        const ctx = document.getElementById("global-importance-chart").getContext("2d");
        
        const prettyNames = {
            "Cl.thickness": "Clump Thickness",
            "Cell.size": "Cell Size Uniformity",
            "Cell.shape": "Cell Shape Uniformity",
            "Marg.adhesion": "Marginal Adhesion",
            "Epith.c.size": "Epithelial Cell Size",
            "Bare.nuclei": "Bare Nuclei",
            "Bl.cromatin": "Bland Chromatin",
            "Normal.nucleoli": "Normal Nucleoli",
            "Mitoses": "Mitoses"
        };
        
        const rawFeatures = Object.keys(importanceData);
        const values = Object.values(importanceData);
        const labels = rawFeatures.map(f => prettyNames[f] || f);

        if (globalShapChart) {
            globalShapChart.destroy();
        }

        globalShapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Mean Absolute SHAP Value (Impact)',
                    data: values,
                    backgroundColor: 'rgba(0, 242, 254, 0.4)',
                    borderColor: 'var(--color-accent)',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'var(--text-secondary)' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: 'var(--text-secondary)' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // ==========================================
    // 4. Biomedical Literature RAG Search (Tab 3)
    // ==========================================
    function initRagSearch() {
        const searchInput = document.getElementById("rag-query");
        const searchBtn = document.getElementById("btn-rag-search");
        const quickTags = document.querySelectorAll(".quick-tag");

        searchBtn.addEventListener("click", performRagSearch);
        searchInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") performRagSearch();
        });

        quickTags.forEach(tag => {
            tag.addEventListener("click", (e) => {
                e.preventDefault();
                searchInput.value = tag.textContent;
                performRagSearch();
            });
        });
    }

    async function performRagSearch() {
        const queryVal = document.getElementById("rag-query").value.trim();
        if (!queryVal) return;

        const container = document.getElementById("rag-results-container");
        const infoEl = document.getElementById("rag-results-info");
        
        container.innerHTML = `
            <div class="no-results">
                <i class="fa-solid fa-circle-notch fa-spin text-accent" style="opacity: 1; font-size: 32px;"></i>
                <h4>Searching Literature Database</h4>
                <p>Retrieving relevant research papers and sample reference files...</p>
            </div>
        `;
        infoEl.style.display = "none";

        try {
            const response = await fetch(`${API_BASE}/rag/retrieve`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: queryVal, k: 3 })
            });

            if (!response.ok) throw new Error("Search query failed");
            const data = await response.json();
            
            renderRagResults(data);
            
        } catch (error) {
            container.innerHTML = `
                <div class="no-results">
                    <i class="fa-solid fa-triangle-exclamation" style="color: var(--color-high-risk); opacity: 1;"></i>
                    <h4>Search Error</h4>
                    <p>${error.message}</p>
                </div>
            `;
        }
    }

    function renderRagResults(data) {
        const container = document.getElementById("rag-results-container");
        const infoEl = document.getElementById("rag-results-info");
        const countEl = document.getElementById("rag-results-count");
        const latencyEl = document.getElementById("rag-results-latency");

        container.innerHTML = "";
        
        if (!data.results || data.results.length === 0) {
            container.innerHTML = `
                <div class="no-results">
                    <i class="fa-solid fa-box-open"></i>
                    <h4>No Documents Found</h4>
                    <p>No literature matches the search query. Try broadening your query parameters.</p>
                </div>
            `;
            return;
        }

        // Show metadata info
        infoEl.style.display = "block";
        countEl.textContent = data.results.length;
        latencyEl.textContent = data.latency_ms.toFixed(1);

        // Highlight helper
        const highlightText = (text, query) => {
            if (!query) return text;
            const words = query.split(/\s+/).filter(w => w.length > 2);
            let highlighted = text;
            words.forEach(word => {
                const regex = new RegExp(`(${word})`, 'gi');
                highlighted = highlighted.replace(regex, "<mark style='background-color: rgba(0, 242, 254, 0.25); color: #fff; border-radius: 2px; padding: 0 2px;'>$1</mark>");
            });
            return highlighted;
        };

        data.results.forEach(res => {
            const card = document.createElement("div");
            card.className = "rag-result-card";
            
            // Highlight terms
            const highlightedAbstract = highlightText(res.abstract, data.query);
            
            card.innerHTML = `
                <div class="rag-card-header">
                    <h4 class="rag-card-title">${res.title}</h4>
                    <span class="rag-score-badge">Match: ${(res.similarity_score * 100).toFixed(1)}%</span>
                </div>
                <div class="rag-card-metadata">
                    <span><i class="fa-solid fa-user-group"></i> ${res.authors}</span>
                    <span><i class="fa-solid fa-book-journal-whills"></i> ${res.journal} (${res.year})</span>
                    <span><i class="fa-solid fa-server"></i> Source: ${res.source}</span>
                </div>
                <p class="rag-card-snippet">${highlightedAbstract}</p>
            `;
            container.appendChild(card);
        });
    }

    // ==========================================
    // 5. Batch Inference (Tab 4)
    // ==========================================
    let cachedBatchData = null; // store parsed CSV results for download

    function initBatchInference() {
        const dropzone = document.getElementById("csv-dropzone");
        const fileInput = document.getElementById("csv-file-input");
        const downloadTemplateBtn = document.getElementById("btn-download-template");
        const downloadResultsBtn = document.getElementById("btn-download-results");

        // Click actions
        dropzone.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", handleFileSelect);
        
        // Drag & Drop bindings
        ["dragenter", "dragover"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.style.borderColor = "var(--color-accent)";
                dropzone.style.backgroundColor = "rgba(0, 242, 254, 0.03)";
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.style.borderColor = "rgba(255, 255, 255, 0.12)";
                dropzone.style.backgroundColor = "rgba(255, 255, 255, 0.01)";
            }, false);
        });

        dropzone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect({ target: { files: files } });
            }
        });

        // Template builder
        downloadTemplateBtn.addEventListener("click", (e) => {
            e.preventDefault();
            const csvContent = "data:text/csv;charset=utf-8,Cl.thickness,Cell.size,Cell.shape,Marg.adhesion,Epith.c.size,Bare.nuclei,Bl.cromatin,Normal.nucleoli,Mitoses\n5,3,3,1,2,1,3,1,1\n8,8,8,5,7,10,9,7,1\n1,1,1,1,2,1,2,1,1\n";
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "patient_features_template.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });

        // Download processed results
        downloadResultsBtn.addEventListener("click", () => {
            if (!cachedBatchData) return;
            
            // Build CSV rows
            let csv = "Patient_Num,Cl.thickness,Cell.size,Cell.shape,Marg.adhesion,Epith.c.size,Bare.nuclei,Bl.cromatin,Normal.nucleoli,Mitoses,Predicted_Class,Label,Confidence,RF_Probability,LR_Probability,SVM_Probability\n";
            
            cachedBatchData.forEach((row) => {
                csv += `${row.num},${row.features.join(",")},${row.prediction},${row.label},${row.conf.toFixed(4)},${row.rf.toFixed(4)},${row.lr.toFixed(4)},${row.svm.toFixed(4)}\n`;
            });
            
            const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.setAttribute("href", url);
            link.setAttribute("download", "batch_cancer_risk_predictions.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (event) => {
            const text = event.target.result;
            processCsvText(text);
        };
        reader.readAsText(file);
    }

    async function processCsvText(csvText) {
        const lines = csvText.split(/\r?\n/).filter(line => line.trim() !== "");
        if (lines.length < 2) {
            alert("Empty or invalid CSV file.");
            return;
        }

        // Parse header and check features
        const headers = lines[0].toLowerCase().split(",").map(h => h.trim());
        
        // Map headers to indices
        const indices = {
            thickness: headers.findIndex(h => h.includes("thick")),
            size: headers.findIndex(h => h.includes("size")),
            shape: headers.findIndex(h => h.includes("shape")),
            adhesion: headers.findIndex(h => h.includes("adhes")),
            epith: headers.findIndex(h => h.includes("epith")),
            nuclei: headers.findIndex(h => h.includes("nuclei")),
            chromatin: headers.findIndex(h => h.includes("chrom") || h.includes("crom")),
            nucleoli: headers.findIndex(h => h.includes("nucleol")),
            mitoses: headers.findIndex(h => h.includes("mito"))
        };

        // Validate headers roughly
        if (Object.values(indices).includes(-1)) {
            alert("Error: CSV must contain columns: Cl.thickness, Cell.size, Cell.shape, Marg.adhesion, Epith.c.size, Bare.nuclei, Bl.cromatin, Normal.nucleoli, Mitoses");
            return;
        }

        const patientsList = [];
        const rawRowValues = [];

        // Parse lines
        for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(",").map(c => parseFloat(c.trim()));
            if (cols.length < headers.length) continue;

            const record = {
                "Cl.thickness": cols[indices.thickness],
                "Cell.size": cols[indices.size],
                "Cell.shape": cols[indices.shape],
                "Marg.adhesion": cols[indices.adhesion],
                "Epith.c.size": cols[indices.epith],
                "Bare.nuclei": cols[indices.nuclei],
                "Bl.cromatin": cols[indices.chromatin],
                "Normal.nucleoli": cols[indices.nucleoli],
                "Mitoses": cols[indices.mitoses]
            };

            // Double check values
            if (Object.values(record).some(val => isNaN(val))) continue;
            
            patientsList.push(record);
            rawRowValues.push([
                record["Cl.thickness"], record["Cell.size"], record["Cell.shape"],
                record["Marg.adhesion"], record["Epith.c.size"], record["Bare.nuclei"],
                record["Bl.cromatin"], record["Normal.nucleoli"], record["Mitoses"]
            ]);
        }

        if (patientsList.length === 0) {
            alert("No valid patient records found in CSV.");
            return;
        }

        // Trigger batch prediction on API
        try {
            const response = await fetch(`${API_BASE}/predict/batch`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ patients: patientsList })
            });

            if (!response.ok) throw new Error("Batch inference failed");
            const data = await response.json();
            
            // Build cache
            cachedBatchData = data.predictions.map((p, idx) => ({
                num: idx + 1,
                features: rawRowValues[idx],
                prediction: p.prediction,
                label: p.prediction_label,
                conf: p.confidence_score,
                rf: p.model_probabilities.random_forest,
                lr: p.model_probabilities.logistic_regression,
                svm: p.model_probabilities.svm
            }));

            renderBatchTable();
            
        } catch (error) {
            alert(`Batch Prediction Error: ${error.message}`);
        }
    }

    function renderBatchTable() {
        const container = document.getElementById("batch-results-area");
        const tbody = document.querySelector("#batch-table tbody");
        
        tbody.innerHTML = "";
        container.style.display = "block";
        
        cachedBatchData.forEach(row => {
            const tr = document.createElement("tr");
            const badgeClass = row.prediction === 1 ? "text-high-risk" : "text-low-risk";
            
            tr.innerHTML = `
                <td>Patient ${row.num}</td>
                <td class="${badgeClass}" style="font-weight: 700;">${row.label}</td>
                <td style="font-weight: 600;">${(row.conf * 100).toFixed(1)}%</td>
                <td>${(row.rf * 100).toFixed(1)}%</td>
                <td>${(row.lr * 100).toFixed(1)}%</td>
                <td>${(row.svm * 100).toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        });
        
        // Add color styles if needed directly
        const style = document.createElement('style');
        style.innerHTML = `
            .text-high-risk { color: var(--color-high-risk); }
            .text-low-risk { color: var(--color-low-risk); }
        `;
        document.head.appendChild(style);
    }
});
