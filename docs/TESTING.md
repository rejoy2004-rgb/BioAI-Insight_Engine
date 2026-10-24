# Testing

## How to run

```bash
pip install -r requirements-dev.txt
ruff check .                      # lint
python -m pytest                  # 134 tests, ~30 s, offline, no credentials
python scripts/e2e_smoke.py       # headless-Chromium run of the real UI (needs `playwright install chromium`)
python scripts/eval_rag.py [--dense]   # retrieval evaluation (see RAG_EVALUATION.md)
```

Unit/API tests train a *fast* variant (`quick=True`: smaller grids, 40 trees) on a deterministic **synthetic** dataset (240 rows, shared ids, missing values) - never on patient data - so the whole suite needs no downloads. One test additionally checks the real built artifacts if `data/models/` exists.

## Results actually obtained (2026-10-24, Windows 11, Python 3.11.9)

| Command | Outcome |
|---|---|
| `ruff check .` | All checks passed |
| `python -m pytest` | **134 passed**, 0 failed |
| `node --check static/app.js` | OK |
| `python scripts/e2e_smoke.py` | **E2E OK** - no console errors, page errors, failed requests or HTTP >= 400; checks below |
| Fresh venv + `pip install -r requirements-dev.txt` + `python -m src.ml_model` + `pytest` | ruff clean; same training metrics; **134 passed** with fastapi 0.162.0, pydantic 2.14.1, pandas 2.3.3, scikit-learn 1.5.2, shap 0.49.1, pymupdf 2.0 (the dev environment had fastapi 0.139, pandas 2.2.2, scikit-learn 1.5.0, shap 0.45.0) |
| Server start from a different working directory with missing models | server starts; `/health` 200, `/ready` 503 with instruction, `/predict` 422/503, `/` 200 (manual check) |

## What is covered

| Area | Tests |
|---|---|
| Data validation (`test_preprocessing.py`) | real dataset shape/missing/duplicates not silently altered; missing column; empty; out-of-range (0, 11, -3); infinity; non-numeric; unexpected target; constant feature; all-missing column; scaler fit only on training data |
| Metrics (`test_metrics.py`) | perfect classifier; one-class data -> `None` not 0; no predicted positives -> precision `None`; NaN/inf/out-of-range scores rejected; shape mismatch; empty; threshold trade-off; calibration error |
| Training & artifacts (`test_training.py`) | grouped + stratified split (no shared ids); reproducibility; baseline AUC = 0.5; scaler fitted on train rows only; single-class data rejected; invalid data rejected; missing artifacts reported not raised; tampered bundle rejected (hash); feature-list mismatch rejected; inference rejects missing/extra/NaN features; feature order independent of dict order; invalid model scores rejected |
| Explainability (`test_explainability.py`) | SHAP output-shape variants (list of classes, 3-D, 2-D, 1-D, single-element list, wrong shapes, NaN); base-value variants; additivity (RF in score space, LR in log-odds); direction of attributions; SVM/baseline reported unsupported; global importance computed, sorted, non-negative; missing background; graceful degradation when SHAP fails |
| RAG (`test_rag.py`) | chunking overlap/coverage; control characters; header/reference stripping; empty/missing/malformed manifest; missing metadata/files; documents with no text; duplicates; path traversal; inline text; shipped PDF loads; ranking; no-result queries; empty/whitespace/over-long queries; repeats; k clamp; empty corpus; verbatim-citation integrity; no-evidence; prompt-injection text; latency smoke (200 docs); dense fallback |
| API (`test_api.py`) | health/ready; 503 when models failed; response schema; whole-number floats; 9 invalid-field cases; malformed JSON (6 bodies); NaN/Infinity; wrong content type; 413; batch limits; metrics/explain endpoints; RAG endpoints and 8 invalid requests; safe 500 for internal retrieval failure (no leakage); `/train` absent by default; no CORS; static path traversal; real artifacts |
| Frontend static (`test_frontend.py`) | well-formed HTML, unique ids; every id referenced by JS exists; endpoints called by JS exist; response fields used by JS exist in schema; no `innerHTML`/`eval`/inline scripts; vendored assets present; tab/ARIA basics; JS syntax |
| Browser E2E (`scripts/e2e_smoke.py`) | page load + ready status; slider/example -> prediction + 9 SHAP bars; metrics tab (4 models, charts); evidence answer with quotes + citations; no-evidence state; XSS payload in query does not execute; malformed CSV -> readable error; valid CSV -> table; arrow-key tab navigation; 390 px viewport has no horizontal overflow |

## Regression tests for bugs found along the way

* ROC/PR charts rendered empty (Chart.js scatter needs `showLine`) - found by looking at the E2E screenshot; covered now by the E2E run only (no pixel assertion).
* Header/footer heuristic removed repeating body text - covered by `test_strip_boilerplate_removes_headers_and_references`.
* Torch DLL load-order failure on Windows - handled by import order in `experiments/dl_model.py`, `scripts/eval_rag.py` and `src/__init__.py` (not unit-tested).

## Not tested / not verified

* Docker build and run; GitHub Actions workflow; Linux/macOS; Python versions other than 3.11.
* Behaviour under load or concurrent users; slow-network behaviour in the browser.
* Screen-reader testing (only ARIA structure and keyboard navigation were checked).
* Cross-browser (only Chromium).
* The dense retriever is covered by the evaluation script and a fallback test, not by an automated unit test of embeddings (it needs a model download).
