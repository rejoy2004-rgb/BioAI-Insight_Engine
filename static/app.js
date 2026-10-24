// BioAI Insight Engine - frontend controller.
// All dynamic content is written with textContent / createElement, never as HTML strings.
"use strict";

const FEATURES = [
    { name: "Cl.thickness", id: "cl_thickness", label: "Clump thickness", def: 5, desc: "How densely cells are layered or clumped." },
    { name: "Cell.size", id: "cell_size", label: "Uniformity of cell size", def: 3, desc: "1 = uniform, 10 = highly varied." },
    { name: "Cell.shape", id: "cell_shape", label: "Uniformity of cell shape", def: 3, desc: "1 = uniform, 10 = highly varied." },
    { name: "Marg.adhesion", id: "marg_adhesion", label: "Marginal adhesion", def: 1, desc: "Tendency of cells to stick together at the margin." },
    { name: "Epith.c.size", id: "epith_c_size", label: "Single epithelial cell size", def: 2, desc: "1 = normal, 10 = markedly enlarged." },
    { name: "Bare.nuclei", id: "bare_nuclei", label: "Bare nuclei", def: 1, desc: "Nuclei not surrounded by cytoplasm." },
    { name: "Bl.cromatin", id: "bl_cromatin", label: "Bland chromatin", def: 3, desc: "1 = fine texture, 10 = coarse." },
    { name: "Normal.nucleoli", id: "normal_nucleoli", label: "Normal nucleoli", def: 1, desc: "Prominence of nucleoli." },
    { name: "Mitoses", id: "mitoses", label: "Mitoses", def: 1, desc: "Degree of mitotic activity." },
];
const MODEL_LABELS = { logistic_regression: "Logistic Regression", random_forest: "Random Forest", svm: "SVM (RBF)", baseline: "Majority baseline" };
const EXAMPLES = {
    benign: [2, 1, 1, 1, 2, 1, 2, 1, 1],
    malignant: [8, 8, 8, 5, 7, 10, 9, 7, 1],
};
const TAB_META = {
    "patient-predict": ["Biopsy Classifier with Explanations", "Classify a fine-needle-aspirate cytology profile and see which inputs drove the model's score."],
    "model-metrics": ["Model Evaluation", "Metrics computed by the training pipeline on a held-out test set; nothing on this page is hard-coded."],
    "rag-search": ["Scientific Evidence Search", "Cited, extractive answers from the documents indexed in this repository."],
    "batch-inference": ["Batch CSV Classification", "Run the same model on many rows. Inputs are validated before they are sent."],
};
const pct = (v, d = 1) => (v === null || v === undefined ? "n/a" : (v * 100).toFixed(d) + "%");
const num = (v, d = 3) => (v === null || v === undefined ? "n/a" : Number(v).toFixed(d));
const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k === "style") Object.assign(node.style, v);
        else node.setAttribute(k, v);
    }
    for (const c of children) if (c !== null && c !== undefined) node.append(c);
    return node;
}
const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

async function api(path, { method = "GET", body, timeoutMs = 15000, signal } = {}) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    if (signal) signal.addEventListener("abort", () => ctrl.abort());
    try {
        const res = await fetch(path, {
            method, signal: ctrl.signal,
            headers: body ? { "Content-Type": "application/json" } : undefined,
            body: body ? JSON.stringify(body) : undefined,
        });
        let data = null;
        try { data = await res.json(); } catch (_) { /* non-JSON error body */ }
        if (!res.ok) {
            let msg = (data && data.detail) || `Request failed (${res.status})`;
            if (data && Array.isArray(data.errors) && data.errors.length) {
                msg = data.errors.map((e) => `${e.field}: ${e.message}`).join("; ");
            }
            const err = new Error(typeof msg === "string" ? msg : "Request failed");
            err.status = res.status;
            throw err;
        }
        return data;
    } catch (e) {
        if (e.name === "AbortError") {
            const err = new Error(signal && signal.aborted ? "Request cancelled." : "The request timed out. Please try again.");
            err.aborted = true;
            throw err;
        }
        if (e instanceof TypeError) throw new Error("Cannot reach the API. Is the server running?");
        throw e;
    } finally {
        clearTimeout(timer);
    }
}

// ------------------------------------------------------------------ status
let apiState = { ready: false, trainingEnabled: false };
function showBanner(node, message) {
    node.textContent = message || "";
    node.hidden = !message;
}
async function checkReady() {
    const dot = $("status-dot"), text = $("api-status-text"), lat = $("api-latency"), banner = $("api-banner");
    const t0 = performance.now();
    try {
        const res = await fetch("/ready", { cache: "no-store" });
        const data = await res.json();
        lat.textContent = `${Math.round(performance.now() - t0)} ms`;
        apiState = { ready: res.ok, trainingEnabled: !!data.training_enabled };
        dot.className = "status-dot " + (res.ok ? "online" : "offline");
        text.textContent = res.ok ? "API ready" : "Models not loaded";
        showBanner(banner, res.ok ? "" : (data.detail || "The model artifacts are not loaded. Run: python -m src.ml_model"));
        $("retrain-card").hidden = !apiState.trainingEnabled;
    } catch (_) {
        apiState.ready = false;
        dot.className = "status-dot offline";
        text.textContent = "API offline";
        lat.textContent = "-- ms";
        showBanner(banner, "Cannot reach the API server.");
    }
}

// ------------------------------------------------------------------ tabs
function initTabs() {
    const tabs = [...document.querySelectorAll('[role="tab"]')];
    const activate = (tab, focus) => {
        const target = tab.dataset.tab;
        tabs.forEach((t) => {
            const on = t === tab;
            t.classList.toggle("active", on);
            t.setAttribute("aria-selected", String(on));
            t.tabIndex = on ? 0 : -1;
            const panel = $(`${t.dataset.tab}-tab`);
            panel.classList.toggle("active", on);
            panel.hidden = !on;
        });
        $("current-tab-title").textContent = TAB_META[target][0];
        $("current-tab-desc").textContent = TAB_META[target][1];
        if (focus) tab.focus();
        if (target === "model-metrics") loadMetrics();
        if (target === "rag-search") loadCorpusNote();
    };
    tabs.forEach((tab, i) => {
        tab.setAttribute("aria-label", tab.textContent.trim());
        tab.addEventListener("click", () => activate(tab, false));
        tab.addEventListener("keydown", (e) => {
            const keys = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 };
            if (e.key in keys) { e.preventDefault(); activate(tabs[(i + keys[e.key] + tabs.length) % tabs.length], true); }
            if (e.key === "Home") { e.preventDefault(); activate(tabs[0], true); }
            if (e.key === "End") { e.preventDefault(); activate(tabs[tabs.length - 1], true); }
        });
    });
}

// ------------------------------------------------------------------ classifier
function buildForm() {
    const form = $("predict-form");
    FEATURES.forEach((f) => {
        const val = el("span", { class: "slider-val", id: `val-${f.id}`, text: String(f.def) });
        const input = el("input", { type: "range", id: f.id, name: f.name, min: "1", max: "10", step: "1", value: String(f.def), class: "range-slider", "aria-describedby": `desc-${f.id}` });
        input.addEventListener("input", () => { val.textContent = input.value; schedulePredict(); });
        form.append(el("div", { class: "slider-group" },
            el("div", { class: "slider-header" }, el("label", { for: f.id, text: f.label }), val),
            input, el("p", { class: "slider-desc", id: `desc-${f.id}`, text: f.desc })));
    });
    const load = (arr) => FEATURES.forEach((f, i) => { $(f.id).value = arr[i]; $(`val-${f.id}`).textContent = arr[i]; });
    $("btn-example-benign").addEventListener("click", () => { load(EXAMPLES.benign); schedulePredict(); });
    $("btn-example-malignant").addEventListener("click", () => { load(EXAMPLES.malignant); schedulePredict(); });
    form.addEventListener("submit", (e) => e.preventDefault());
}
function currentPayload() {
    const out = {};
    FEATURES.forEach((f) => { out[f.name] = Number($(f.id).value); });
    return out;
}
let predictTimer = null, predictSeq = 0, predictCtrl = null;
function schedulePredict() { clearTimeout(predictTimer); predictTimer = setTimeout(runPrediction, 250); }

async function runPrediction() {
    const seq = ++predictSeq;
    if (predictCtrl) predictCtrl.abort();
    predictCtrl = new AbortController();
    const live = $("prediction-live");
    live.setAttribute("aria-busy", "true");
    try {
        const data = await api("/predict", { method: "POST", body: currentPayload(), signal: predictCtrl.signal });
        if (seq !== predictSeq) return;
        renderPrediction(data);
    } catch (e) {
        if (e.aborted && seq !== predictSeq) return;
        if (seq !== predictSeq) return;
        renderPredictionError(e.message);
    } finally {
        if (seq === predictSeq) live.setAttribute("aria-busy", "false");
    }
}
function renderPredictionError(message) {
    const badge = $("risk-badge");
    badge.textContent = "Unavailable";
    badge.className = "badge";
    $("risk-explanation").textContent = message;
    $("result-probability").textContent = "--";
    $("gauge-stroke").style.strokeDashoffset = "314.16";
    clear($("model-scores"));
    clear($("shap-cont"));
}
function renderPrediction(d) {
    const score = d.malignancy_score;
    $("result-probability").textContent = pct(score, 1);
    $("gauge-img").setAttribute("aria-label", `Model score ${pct(score, 1)}`);
    const stroke = $("gauge-stroke");
    stroke.style.strokeDashoffset = String(314.16 * (1 - score));
    const malignant = d.prediction === 1;
    stroke.style.stroke = malignant ? "var(--color-high-risk)" : "var(--color-low-risk)";
    const badge = $("risk-badge");
    badge.textContent = d.prediction_label;
    badge.className = "badge " + (malignant ? "high-risk" : "low-risk");
    $("risk-explanation").textContent =
        `Score ${pct(score, 1)} vs. decision threshold ${d.threshold.toFixed(2)} (${MODEL_LABELS[d.primary_model] || d.primary_model}). ` +
        (malignant ? "The model assigns this profile to the malignant class." : "The model assigns this profile to the benign class.") +
        " This is not a diagnosis.";

    const box = $("model-scores");
    clear(box);
    box.append(el("h4", { class: "mini-heading", text: "Score from each model" }));
    Object.entries(d.model_scores).forEach(([name, s]) => {
        const fill = el("div", { class: "prob-bar-fill", style: { width: `${(s * 100).toFixed(1)}%` } });
        box.append(el("div", { class: "model-prob-item" },
            el("span", { text: MODEL_LABELS[name] || name + (name === d.primary_model ? " (primary)" : "") }),
            el("div", { class: "prob-bar-container", role: "img", "aria-label": `${MODEL_LABELS[name] || name}: ${pct(s)}` }, fill),
            el("span", { class: "prob-num", text: pct(s) })));
    });
    renderShap(d.explanation);
}
function renderShap(ex) {
    const cont = $("shap-cont");
    clear(cont);
    if (!ex || ex.status !== "ok") {
        cont.append(el("div", { class: "shap-base-info", text: "Explanation unavailable" + (ex && ex.reason ? `: ${ex.reason}` : ".") }));
        return;
    }
    $("shap-units").textContent =
        `Contributions are in ${ex.units} (${ex.method}); positive values push toward the malignant class. ` +
        `SHAP explains the model's behaviour, not biological causes.`;
    const items = [...ex.attributions].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
    const maxVal = Math.max(...items.map((a) => Math.abs(a.shap_value)), 1e-6);
    items.forEach((a) => {
        const f = FEATURES.find((x) => x.name === a.feature);
        const w = (Math.abs(a.shap_value) / maxVal) * 36;
        const pos = a.shap_value >= 0;
        const bar = el("div", { class: "shap-bar " + (pos ? "positive" : "negative"), style: { width: `${w}%`, left: pos ? "50%" : `${50 - w}%` } });
        const label = el("div", { class: "shap-bar-val", text: (pos ? "+" : "") + a.shap_value.toFixed(2),
            style: pos ? { left: `calc(50% + ${w}% + 6px)` } : { right: `calc(50% + ${w}% + 6px)` } });
        cont.append(el("div", { class: "shap-bar-item", role: "img",
            "aria-label": `${f ? f.label : a.feature} = ${a.value}: contribution ${a.shap_value.toFixed(2)}, ${pos ? "toward malignant" : "toward benign"}` },
            el("div", { class: "shap-feat-name", text: `${f ? f.label : a.feature} = ${a.value}` }),
            el("div", { class: "shap-bar-wrapper" }, bar, label)));
    });
    cont.append(el("div", { class: "shap-base-info", text: `Baseline (average model output): ${ex.base_value.toFixed(2)} ${ex.units}` }));
}

// ------------------------------------------------------------------ evaluation tab
let metricsCache = null, charts = {};
const PALETTE = { logistic_regression: "#00b8d4", random_forest: "#a855f7", svm: "#f59e0b", baseline: "#64748b" };
async function loadMetrics(force = false) {
    const status = $("metrics-status");
    if (metricsCache && !force) return;
    status.hidden = false;
    status.textContent = "Loading evaluation results…";
    try {
        metricsCache = await api("/model/metrics");
        status.hidden = true;
        renderMetrics(metricsCache);
    } catch (e) {
        status.textContent = `Could not load evaluation results: ${e.message}`;
    }
}
function kv(label, value) { return el("div", { class: "kv" }, el("span", { text: label }), el("strong", { text: String(value) })); }
function renderMetrics(m) {
    const ds = m.dataset, sp = m.split;
    const sum = $("dataset-summary");
    clear(sum);
    sum.append(
        kv("Rows", ds.n_rows), kv("Benign / malignant", `${ds.class_counts.benign} / ${ds.class_counts.malignant}`),
        kv("Missing values", Object.entries(ds.missing_by_feature).map(([k, v]) => `${k}: ${v}`).join(", ") || "none"),
        kv("Duplicate feature rows", ds.duplicate_feature_rows), kv("Rows sharing a sample ID", ds.duplicate_ids),
        kv("Train / test rows", `${sp.n_train} / ${sp.n_test}`), kv("Seed", m.seed),
        el("p", { class: "fineprint", text: "Split is stratified and grouped by sample ID so related rows never appear on both sides. Imputation and scaling are fit on training folds only." }));

    $("eval-note").textContent = `Test set: ${sp.n_test} rows (${sp.test_malignant} malignant). Metrics use threshold 0.5; ` +
        `sensitivity = recall on the malignant class. Small test sets give wide uncertainty.`;
    const tb = $("metrics-table").querySelector("tbody");
    clear(tb);
    Object.entries(m.test_metrics).forEach(([name, t]) => {
        const cv = m.models[name].cv;
        const tr = el("tr");
        const nameCell = el("th", { scope: "row", text: (MODEL_LABELS[name] || name) + (name === m.primary_model ? " ★" : "") });
        tr.append(nameCell,
            el("td", { text: cv ? `${num(cv.roc_auc_mean)} ± ${num(cv.roc_auc_std)}` : "n/a" }),
            el("td", { text: num(t.roc_auc) }), el("td", { text: pct(t.recall_sensitivity) }), el("td", { text: pct(t.specificity) }),
            el("td", { text: pct(t.precision) }), el("td", { text: num(t.pr_auc) }));
        tb.append(tr);
    });
    $("selection-note").textContent = `★ primary model. ${m.primary_selection_reason} Selection used cross-validation on the training split only. ` +
        `CV figures come from the tuning folds and are slightly optimistic.`;

    const names = Object.keys(m.test_curves).filter((n) => m.test_curves[n] && n !== "baseline");
    const mk = (id, key, xk, yk, xl, yl, diag) => {
        if (charts[id]) charts[id].destroy();
        const datasets = names.map((n) => ({
            label: MODEL_LABELS[n] || n, borderColor: PALETTE[n] || "#fff", backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, showLine: true,
            data: m.test_curves[n][key][xk].map((x, i) => ({ x, y: m.test_curves[n][key][yk][i] })), stepped: key === "pr" ? "before" : false,
        }));
        if (diag) datasets.push({ label: "Chance", borderColor: "#64748b", borderDash: [5, 5], borderWidth: 1, pointRadius: 0, showLine: true, data: [{ x: 0, y: 0 }, { x: 1, y: 1 }] });
        charts[id] = new Chart($(id), { type: "scatter", data: { datasets }, options: chartOpts(xl, yl) });
    };
    if (window.Chart) {
        mk("roc-chart", "roc", "fpr", "tpr", "False positive rate", "True positive rate (sensitivity)", true);
        mk("pr-chart", "pr", "recall", "precision", "Recall (sensitivity)", "Precision", false);
    }

    const prim = m.primary_model;
    const cm = m.primary_test_metrics_at_operating_threshold;
    const c = cm.confusion_matrix;
    const conf = $("confusion");
    clear(conf);
    const cell = (t, v, cls) => el("div", { class: `cm-cell ${cls}` }, el("strong", { text: String(v) }), el("span", { text: t }));
    conf.append(el("div"), el("div", { class: "cm-head", text: "Predicted benign" }), el("div", { class: "cm-head", text: "Predicted malignant" }),
        el("div", { class: "cm-head", text: "Actually benign" }), cell("true negative", c.tn, "ok"), cell("false positive", c.fp, "warn"),
        el("div", { class: "cm-head", text: "Actually malignant" }), cell("false negative (missed)", c.fn, "bad"), cell("true positive", c.tp, "ok"));
    $("threshold-note").textContent =
        `${MODEL_LABELS[prim] || prim} at its operating threshold ${m.operating_threshold.toFixed(2)}: sensitivity ${pct(cm.recall_sensitivity)}, specificity ${pct(cm.specificity)}. ` +
        `The threshold was chosen from out-of-fold training predictions to target ≥${pct(m.target_recall, 0)} sensitivity; the test set was not used. ` +
        `Lowering the threshold catches more malignant cases at the cost of more false alarms.`;
    const th = $("threshold-table").querySelector("tbody");
    clear(th);
    m.threshold_tables[prim].forEach((r) => th.append(el("tr", {}, el("td", { text: r.threshold.toFixed(1) }), el("td", { text: pct(r.recall_sensitivity) }),
        el("td", { text: pct(r.specificity) }), el("td", { text: String(r.fn) }), el("td", { text: String(r.fp) }))));
    const rel = $("reliability-table").querySelector("tbody");
    clear(rel);
    m.reliability_primary.forEach((r) => rel.append(el("tr", {}, el("td", { text: r.bin }), el("td", { text: String(r.n) }), el("td", { text: pct(r.mean_score) }), el("td", { text: pct(r.observed_malignant_rate) }))));

    const gi = m.global_importance[prim];
    if (gi && window.Chart) {
        $("global-note").textContent = `Mean |SHAP| for ${MODEL_LABELS[prim]} on the training split, in ${gi.units}. Describes model behaviour, not causality.`;
        const labels = Object.keys(gi.importance), vals = Object.values(gi.importance);
        if (charts.g) charts.g.destroy();
        charts.g = new Chart($("global-importance-chart"), {
            type: "bar",
            data: { labels, datasets: [{ label: "Mean |SHAP|", data: vals, backgroundColor: "#00b8d4" }] },
            options: { ...chartOpts("Mean |SHAP|", ""), indexAxis: "y", plugins: { legend: { display: false } } },
        });
    } else {
        $("global-note").textContent = "Global SHAP importance is unavailable for the primary model.";
    }
}
function chartOpts(xl, yl) {
    const grid = { color: "rgba(255,255,255,0.08)" }, tick = { color: "#94a3b8" };
    return {
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: { x: { title: { display: !!xl, text: xl, color: "#94a3b8" }, grid, ticks: tick }, y: { title: { display: !!yl, text: yl, color: "#94a3b8" }, grid, ticks: tick } },
        plugins: { legend: { labels: { color: "#cbd5e1" } } },
    };
}
function initRetrain() {
    $("btn-retrain").addEventListener("click", async () => {
        const out = $("retrain-output"), btn = $("btn-retrain");
        out.hidden = false;
        out.textContent = "Retraining… this can take a minute.";
        btn.disabled = true;
        try {
            const d = await api("/train", { method: "POST", body: { seed: Number($("training-seed").value) || 48 }, timeoutMs: 300000 });
            out.textContent = `Done (seed ${d.seed}). Primary model: ${MODEL_LABELS[d.primary_model] || d.primary_model}.`;
            metricsCache = null;
            await loadMetrics(true);
            schedulePredict();
        } catch (e) { out.textContent = `Retraining failed: ${e.message}`; } finally { btn.disabled = false; }
    });
}

// ------------------------------------------------------------------ evidence search
let ragCtrl = null, corpusLoaded = false;
async function loadCorpusNote() {
    if (corpusLoaded) return;
    try {
        const s = await api("/rag/stats");
        corpusLoaded = true;
        $("rag-corpus-note").textContent = `Index: ${s.documents} document(s), ${s.chunks} passages (${s.retrieval_method}; ${s.answer_method}). ` +
            `Indexed: ${s.titles.join("; ") || "none"}. Add sources via papers/sources.json.`;
    } catch (_) { $("rag-corpus-note").textContent = "Corpus information unavailable."; }
}
function ragMessage(icon, title, text, isError) {
    const box = $("rag-results-container");
    clear(box);
    box.append(el("div", { class: "no-results" }, el("i", { class: `fa-solid ${icon}`, "aria-hidden": "true", style: isError ? { color: "var(--color-high-risk)" } : {} }),
        el("h4", { text: title }), el("p", { text })));
}
function safeUrl(u) { try { const x = new URL(u); return /^https?:$/.test(x.protocol) ? x.href : null; } catch (_) { return null; } }
async function askQuestion(q) {
    q = q.trim();
    if (!q) { ragMessage("fa-circle-info", "Enter a question", "Type a question about the indexed literature.", false); return; }
    if (ragCtrl) ragCtrl.abort();
    ragCtrl = new AbortController();
    const btn = $("btn-rag-search");
    btn.disabled = true;
    $("rag-results-info").hidden = true;
    ragMessage("fa-circle-notch fa-spin", "Searching…", "Retrieving passages from the indexed documents.", false);
    try {
        const d = await api("/rag/ask", { method: "POST", body: { query: q, k: 3 }, signal: ragCtrl.signal });
        renderAnswer(d);
    } catch (e) {
        if (!e.aborted || !ragCtrl.signal.aborted) ragMessage("fa-triangle-exclamation", "Search failed", e.message, true);
    } finally { btn.disabled = false; }
}
function renderAnswer(d) {
    const box = $("rag-results-container");
    clear(box);
    const info = $("rag-results-info");
    info.hidden = false;
    info.textContent = `${d.status === "ok" ? d.sources.length : 0} source passage(s) in ${d.latency_ms.toFixed(1)} ms`;
    if (d.status !== "ok") { ragMessage("fa-box-open", "No supporting evidence found", d.message, false); return; }
    const answer = el("div", { class: "rag-result-card answer-card" }, el("h4", { class: "rag-card-title", text: "Answer from the indexed documents" }));
    d.citations.forEach((c) => answer.append(el("blockquote", { class: "quote" }, `“${c.quote}” `, el("sup", { class: "cite", text: `[${c.n}]` }))));
    answer.append(el("p", { class: "fineprint", text: d.message }));
    box.append(answer);
    d.sources.forEach((s) => {
        const url = s.url ? safeUrl(s.url) : null;
        const meta = el("div", { class: "rag-card-metadata" });
        if (s.authors) meta.append(el("span", { text: s.authors }));
        meta.append(el("span", { text: [s.journal, s.year].filter(Boolean).join(" · ") }));
        if (s.identifier) meta.append(el("span", { text: s.identifier }));
        if (s.page) meta.append(el("span", { text: `page ${s.page}` }));
        const title = el("h4", { class: "rag-card-title" }, `[${s.n}] `, url ? el("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: s.title }) : s.title);
        const card = el("div", { class: "rag-result-card" },
            el("div", { class: "rag-card-header" }, title, el("span", { class: "rag-score-badge", title: "Cosine similarity of TF-IDF vectors (lexical match)", text: `Lexical match ${s.score.toFixed(2)}` })), meta);
        if (s.synthetic) card.append(el("p", { class: "fineprint", text: "Synthetic test document, not a real publication." }));
        box.append(card);
    });
}
function initRag() {
    $("rag-form").addEventListener("submit", (e) => { e.preventDefault(); askQuestion($("rag-query").value); });
    document.querySelectorAll(".quick-tag").forEach((b) => b.addEventListener("click", () => { $("rag-query").value = b.textContent; askQuestion(b.textContent); }));
}

// ------------------------------------------------------------------ batch
const MAX_BATCH_BYTES = 1024 * 1024, MAX_BATCH_ROWS = 1000;
let batchRows = null;
function parseCsv(text) {
    const rows = [];
    let row = [], field = "", quoted = false;
    text = text.replace(/^﻿/, "");
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (quoted) {
            if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (ch === '"') quoted = false; else field += ch;
        } else if (ch === '"') quoted = true;
        else if (ch === ",") { row.push(field); field = ""; } else if (ch === "\n" || ch === "\r") {
            if (ch === "\r" && text[i + 1] === "\n") i++;
            row.push(field); field = "";
            if (row.some((c) => c.trim() !== "")) rows.push(row);
            row = [];
        } else field += ch;
    }
    row.push(field);
    if (row.some((c) => c.trim() !== "")) rows.push(row);
    return rows;
}
function parseBatch(text) {
    const rows = parseCsv(text);
    if (rows.length < 2) throw new Error("The CSV needs a header row and at least one data row.");
    const header = rows[0].map((h) => h.trim());
    const want = FEATURES.map((f) => f.name);
    const missing = want.filter((w) => !header.includes(w)), extra = header.filter((h) => !want.includes(h));
    if (missing.length) throw new Error(`Missing column(s): ${missing.join(", ")}`);
    if (extra.length) throw new Error(`Unexpected column(s): ${extra.join(", ")}`);
    if (rows.length - 1 > MAX_BATCH_ROWS) throw new Error(`At most ${MAX_BATCH_ROWS} rows are allowed.`);
    return rows.slice(1).map((r, i) => {
        if (r.length !== header.length) throw new Error(`Row ${i + 1}: expected ${header.length} values, found ${r.length}.`);
        const o = {};
        header.forEach((h, j) => {
            const raw = r[j].trim(), v = Number(raw);
            if (raw === "" || !Number.isFinite(v) || !Number.isInteger(v) || v < 1 || v > 10) throw new Error(`Row ${i + 1}, ${h}: "${raw}" is not a whole number from 1 to 10.`);
            o[h] = v;
        });
        return o;
    });
}
async function handleFile(file) {
    const err = $("batch-error");
    showBanner(err, "");
    $("batch-results-area").hidden = true;
    if (!file) return;
    try {
        if (file.size > MAX_BATCH_BYTES) throw new Error("File is larger than 1 MB.");
        if (!/\.csv$/i.test(file.name) && file.type !== "text/csv") throw new Error("Please choose a .csv file.");
        const rows = parseBatch(await file.text());
        const data = await api("/predict/batch", { method: "POST", body: { patients: rows }, timeoutMs: 60000 });
        batchRows = data.predictions.map((p, i) => ({ row: i + 1, ...p }));
        renderBatch();
    } catch (e) { showBanner(err, e.message); }
}
function renderBatch() {
    const tb = $("batch-table").querySelector("tbody");
    clear(tb);
    batchRows.forEach((r) => tb.append(el("tr", {}, el("td", { text: String(r.row) }),
        el("td", { class: r.prediction ? "text-high-risk" : "text-low-risk", style: { fontWeight: "700" }, text: r.prediction ? "Malignant-class" : "Benign-class" }),
        el("td", { text: pct(r.malignancy_score) }), el("td", { text: pct(r.model_scores.logistic_regression) }),
        el("td", { text: pct(r.model_scores.random_forest) }), el("td", { text: pct(r.model_scores.svm) }))));
    $("batch-results-area").hidden = false;
}
function download(name, text) {
    const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
    const a = el("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function initBatch() {
    const zone = $("csv-dropzone"), input = $("csv-file-input");
    $("schema-list").append(...FEATURES.map((f) => el("div", { class: "schema-info-item" }, el("strong", { text: `${f.name}: ` }), f.label + ", whole number 1–10")));
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", () => { handleFile(input.files[0]); input.value = ""; });
    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("dragover"); }));
    zone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
    $("btn-download-template").addEventListener("click", () => download("biopsy_features_template.csv",
        FEATURES.map((f) => f.name).join(",") + "\n" + EXAMPLES.benign.join(",") + "\n" + EXAMPLES.malignant.join(",") + "\n"));
    $("btn-download-results").addEventListener("click", () => {
        if (!batchRows) return;
        const head = "row,prediction,malignancy_score,threshold,logistic_regression,random_forest,svm";
        const lines = batchRows.map((r) => [r.row, r.prediction ? "malignant-class" : "benign-class", r.malignancy_score.toFixed(4), r.threshold,
            r.model_scores.logistic_regression.toFixed(4), r.model_scores.random_forest.toFixed(4), r.model_scores.svm.toFixed(4)].join(","));
        download("batch_predictions.csv", [head, ...lines].join("\n") + "\n");
    });
}

// ------------------------------------------------------------------ boot
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    buildForm();
    initRag();
    initBatch();
    initRetrain();
    checkReady().then(runPrediction);
    setInterval(() => { if (!document.hidden) checkReady(); }, 15000);
});
