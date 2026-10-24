# Final audit

Date: 2026-10-24. Environment: Windows 11, Python 3.11.9, scikit-learn 1.5.0, shap 0.45.0, FastAPI 0.139, Node 24, Chromium 151 (Playwright). Compared against [INITIAL_AUDIT.md](INITIAL_AUDIT.md).

## Status of each baseline finding

| # | Finding | Status | Evidence |
|---|---|---|---|
| 1 | Synthetic corpus presented as real literature | **Fixed** - removed; index = papers listed in `papers/sources.json` (one real CC BY paper); generator relabelled and not indexed | `tests/test_rag.py::test_shipped_corpus_loads_real_pdf`; UI/README state the corpus size |
| 2 | Scaler leakage | **Fixed** - imputer+scaler inside Pipelines, fit on training folds; `scaler.pkl` deleted | `test_preprocessor_fit_on_train_only`, `test_scaler_inside_pipeline_fitted_on_training_partition_only` |
| 3 | Misleading clinical framing | **Fixed** in UI/API/docs ("model score (uncalibrated)", "Malignant-class prediction", persistent disclaimer). Residual: the benign/malignant encoding (0/1) is inferred, not documented by the data source | UI screenshots; DATA_DICTIONARY |
| 4 | Splits ignore structure / rows dropped | **Fixed** - grouped+stratified, nothing dropped, duplicate-sensitivity reported | `test_split_is_grouped_and_stratified`; MODEL_CARD |
| 5 | Weak evaluation | **Fixed** - baseline, CV, sensitivity/specificity/PR-AUC/Brier/ECE, threshold table, confusion matrix, curves. **Residual:** no bootstrap confidence intervals | MODEL_CARD |
| 6 | Readiness | **Fixed** - `/ready` 503 until artifacts load and verify | `test_ready_is_503_when_models_failed_to_load` + manual run |
| 7 | Silent failures | **Fixed** - invalid scores raise; SHAP failure is reported as `unavailable` while the prediction still returns | `test_invalid_model_scores_are_rejected`, `test_service_explain_degrades_gracefully` |
| 8 | Open `/train` | **Mitigated** - off by default, single-flight; still unauthenticated when enabled | `test_train_endpoint_absent_by_default` |
| 9 | Hard-coded paths | **Fixed** - `src/config.py`; server verified from another working directory | manual run |
| 10 | SHAP inefficiency / wrong units | **Fixed** - explainers built once; global importance precomputed; units shown | additivity tests |
| 11 | RAG similarity/threshold/chunking | **Fixed** - minimum score, abstention, header/reference stripping, manifest metadata. Dense retrieval kept as an option because it measured better | RAG_EVALUATION |
| 12 | Dependencies / Docker | **Partly fixed** - requirements complete and minimal; verified by clean-venv install (see below). **Dockerfile not built (no Docker here)** | - |
| 13 | XSS / CDN / headers | **Fixed** - safe DOM, vendored libs, CSP, headers, body cap | tests + browser XSS check |
| 14 | Frontend UX/accessibility | **Mostly fixed** - tab semantics, keyboard operation, loading/error/empty states, mobile layout. **Not tested** with a screen reader | E2E |
| 15 | Encodings/deprecations | **Fixed** | pytest runs without those warnings being relevant |
| 16 | Duplicate Streamlit UI | **Removed** (history keeps it) | CHANGELOG |
| 17 | Provenance | **Partly** - documented as unknown/inferred; owner must add the source and licence | DATA_DICTIONARY |
| 18 | Secrets | none found; upstream history not inspected | SECURITY |

## Defects found during the work (and fixed)

* Empty ROC/PR charts (Chart.js scatter datasets lacked `showLine`) - caught only by viewing the screenshot.
* SHAP bars/labels misaligned and value labels overlapping bars; metrics tables clipped; squashed gauge - fixed in CSS.
* Reference-list and page-header text was being returned as "evidence" - added boilerplate stripping.
* My first retrieval design (TF-IDF only) was **worse** than the original dense approach on the evaluation set; a dense option was added instead of dropping it.
* A dataset-agnostic test package name (`tests`) collided with an unrelated installed package; helpers renamed.

## Review of the final diff (self-review)

* No accidental deletions: every removed file is listed in CHANGELOG and exists in the baseline commit.
* No dead modules: `experiments/dl_model.py` and `src/generate_papers.py` are intentionally outside the runtime path and documented as such.
* No fabricated numbers: every figure in README/MODEL_CARD/RAG_EVALUATION was copied from `data/models/metadata.json`, `experiments/dl_results.json` or `docs/rag_eval_output.json`.
* Claims *not* made: no clinical validation, no calibrated probability, no LLM, no semantic search by default, no Docker verification, no CI run.

## Verification summary

| Check | Result |
|---|---|
| `ruff check .` | pass |
| `python -m pytest` | 134 passed |
| `node --check static/app.js` | pass |
| `python scripts/e2e_smoke.py` | pass (no console errors; screenshots regenerated) |
| Fresh venv (`python -m venv`), `pip install -r requirements-dev.txt`, then `ruff`, `python -m src.ml_model`, `pytest` | **pass**: ruff clean; training reproduced the same metrics; 134 passed. Resolved versions: fastapi 0.162.0, pydantic 2.14.1, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.5.2, shap 0.49.1, pymupdf 2.0, uvicorn 0.54.0 (newer than the dev environment, so the version ranges are exercised at both ends) |
| Docker build/run | **NOT VERIFIED** (Docker not installed) |
| GitHub Actions run | **NOT VERIFIED** (not executed) |
| Linux/macOS, other Python versions | **NOT VERIFIED** |
| `pip-audit` / dependency CVE scan | **NOT RUN** |

## Remaining issues / blockers for anything beyond a local demo

1. Docker image and CI have never been executed.
2. Dataset provenance and licence are unknown; the class encoding is inferred.
3. Test set is small (141 rows); no confidence intervals; results are near ceiling and should not be over-interpreted.
4. Evidence base is one paper; retrieval default is lexical (dense is better on the small eval but optional); answers are extractive only.
5. No authentication / rate limiting / HTTPS if exposed publicly; `/train`, if enabled, is unauthenticated.
6. No `LICENSE` file in the repository.
7. Work is staged on branch `audit/healthcare-ml-hardening` in a local git repo created for this audit; it has **not** been committed or pushed, and the baseline commit was created locally from the received snapshot (it is not the GitHub history).
