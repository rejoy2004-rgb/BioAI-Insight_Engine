# Changelog

## 2.0.0 - audit and hardening (branch `audit/healthcare-ml-hardening`)

Positioning: "Explainable Healthcare ML and Scientific Evidence Platform", an educational prototype. Details of the problems fixed: [docs/INITIAL_AUDIT.md](docs/INITIAL_AUDIT.md).

### Breaking changes
* **Removed the synthetic 510-paper corpus** (`data/papers_corpus.json`, `data/rag_documents.json`) and the FAISS cache. It contained fabricated papers under real journal names. The index now comes from `papers/sources.json` (one real open-access paper). `src/generate_papers.py` now only writes a clearly labelled synthetic *test* corpus that is not indexed.
* **API contract**: `/predict` now returns `malignancy_score` (was `confidence_score`), `prediction_label` ("Benign-class/Malignant-class prediction", was "Low/High Cancer Risk"), `threshold`, `primary_model`, `model_scores` (was `model_probabilities`) and a structured `explanation` object (was `shap_attributions`/`shap_base_value`). Batch responses changed the same way. `/health` is liveness only; metrics moved to `/model/metrics`; `/ready` added. `/rag/retrieve` returns `score` (was `similarity_score`) and per-chunk fields; `/rag/ask` added. Inputs must be whole numbers 1-10 and unknown fields are rejected.
* `/train` is disabled unless `BIOAI_ENABLE_TRAIN_ENDPOINT=1`.
* `data/scaler.pkl` removed (it was fit on all data); preprocessing now lives inside the model pipelines (`data/models/model_bundle.joblib`, git-ignored, built by `python -m src.ml_model`).
* Removed `app.py` (Streamlit duplicate UI built on the old API), `test.py` and `test_api.py` (not real tests; they overwrote artifacts), all recoverable from the baseline commit.
* `src/dl_model.py` -> `experiments/dl_model.py`, rewritten to be leak-free and evaluated; not used by the app.

### Added
* Data validation and quality report; grouped/stratified splitting; baseline + tuned models; CV; threshold analysis; calibration diagnostics; metric functions that return `None` when undefined.
* SHAP service built once at startup, shape-normalising, with explicit "unavailable".
* Extractive cited answers with abstention; optional dense retrieval; retrieval evaluation script.
* Request ids, body limits, security headers, strict CSP, structured error responses, readiness endpoint.
* Rewritten frontend controller (safe DOM, loading/error/empty states, tab semantics, keyboard support, responsive layout), model-evaluation dashboard, vendored Chart.js and Font Awesome.
* 134 pytest tests, headless-browser E2E script, ruff config, CI workflow, `.dockerignore`, `.env.example`, documentation set.

### Changed
* `requirements.txt` now lists what is actually imported (no torch/streamlit/faiss/sentence-transformers); dev/experiment extras split out.
* Dockerfile: no compiler toolchain, trains at build time, non-root user, healthcheck (not built here).
* FastAPI `lifespan` replaces `on_event`; Pydantic v2 config.

## 1.0.0 - original upload
Streamlit + FastAPI cancer-risk demo with FAISS/MiniLM retrieval over a synthetic corpus.
