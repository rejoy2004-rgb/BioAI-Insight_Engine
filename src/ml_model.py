"""Reproducible training + evaluation.

    python -m src.ml_model [--seed 48]

Protocol (see docs/MODEL_CARD.md):
  1. Validate the raw data; nothing is dropped.
  2. Hold out ~20% as a final test set with a stratified, *grouped* split
     (rows sharing a sample-code `Id` never straddle train/test).
  3. Tune each model with grouped, stratified CV on the training part only;
     imputation/scaling are inside the Pipeline so they are re-fit per fold.
  4. Pick the primary model from CV scores only (one-standard-error rule,
     preferring the more interpretable model). The test set is not used to select.
  5. Report held-out test metrics once, for every candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src import config
from src.metrics import compute_metrics, curves, threshold_table
from src.preprocessing import FEATURES, ID_COLUMN, TARGET, build_preprocessor, data_quality_report, load_data

log = logging.getLogger("bioai.train")

# Order = interpretability preference used by the one-standard-error rule.
INTERPRETABILITY_ORDER = ["logistic_regression", "random_forest", "svm"]
TARGET_RECALL = 0.95  # sensitivity goal for the operating threshold (chosen on OOF training data)
FRIENDLY = {"baseline": "Majority-class baseline", "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest", "svm": "SVM (RBF kernel)"}


def candidate_models(seed: int, quick: bool = False) -> dict[str, tuple[object, dict]]:
    """quick=True shrinks grids/trees; used only by the test-suite."""
    n_trees = 40 if quick else 300
    models = {
        "baseline": (DummyClassifier(strategy="prior"), {}),
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}),
        "svm": (
            SVC(kernel="rbf", probability=True, random_state=seed),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1], "clf__class_weight": [None, "balanced"]}),
        "random_forest": (
            RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=1),
            {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 3], "clf__class_weight": [None, "balanced"]}),
    }
    if quick:
        models = {k: (c, {kk: vv[:2] for kk, vv in g.items()}) for k, (c, g) in models.items()}
    return models


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def holdout_split(df: pd.DataFrame, seed: int, n_folds: int = 5):
    """Stratified, grouped hold-out (first of k folds -> ~1/k test)."""
    groups = df[ID_COLUMN] if ID_COLUMN in df else pd.Series(np.arange(len(df)))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df[FEATURES], df[TARGET], groups))
    if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
        raise RuntimeError("group leakage between train and test")
    return train_idx, test_idx


def _one_se_choice(cv: dict[str, dict]) -> tuple[str, str]:
    cands = {k: v for k, v in cv.items() if k in INTERPRETABILITY_ORDER}
    best = max(cands, key=lambda k: cands[k]["roc_auc_mean"])
    se = cands[best]["roc_auc_std"] / np.sqrt(cands[best]["n_folds"])
    cutoff = cands[best]["roc_auc_mean"] - se
    for name in INTERPRETABILITY_ORDER:
        if name in cands and cands[name]["roc_auc_mean"] >= cutoff:
            return name, (f"Highest CV ROC-AUC is {best} ({cands[best]['roc_auc_mean']:.4f}); applying the "
                          f"one-standard-error rule (cutoff {cutoff:.4f}) the most interpretable model within "
                          f"it is {name}.")
    return best, "Highest mean CV ROC-AUC."


def train_pipeline(seed: int = config.SEED, data_path=config.DATASET_PATH,
                   models_dir=config.MODELS_DIR, n_splits: int = 5, quick: bool = False) -> dict:
    from src.explainability import compute_global_importance, supports_shap

    df = load_data(data_path).reset_index(drop=True)
    quality = data_quality_report(df)
    train_idx, test_idx = holdout_split(df, seed)
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    X_tr, y_tr = train[FEATURES], train[TARGET].to_numpy()
    g_tr = train[ID_COLUMN].to_numpy() if ID_COLUMN in train else np.arange(len(train))
    X_te, y_te = test[FEATURES], test[TARGET].to_numpy()
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        raise ValueError("Train or test split contains a single class; cannot train/evaluate.")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fitted, cv_summary, best_params = {}, {}, {}
    for name, (clf, grid) in candidate_models(seed, quick).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv, refit="roc_auc", n_jobs=1,
                              scoring={"roc_auc": "roc_auc", "recall": "recall",
                                       "average_precision": "average_precision"})
            gs.fit(X_tr, y_tr, groups=g_tr)
            i, r = gs.best_index_, gs.cv_results_
            cv_summary[name] = {
                "roc_auc_mean": float(r["mean_test_roc_auc"][i]), "roc_auc_std": float(r["std_test_roc_auc"][i]),
                "recall_mean": float(r["mean_test_recall"][i]), "recall_std": float(r["std_test_recall"][i]),
                "pr_auc_mean": float(r["mean_test_average_precision"][i]), "n_folds": n_splits}
            best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
            fitted[name] = gs.best_estimator_
        else:
            pipe.fit(X_tr, y_tr)
            fitted[name] = pipe
            best_params[name] = {}
        log.info("trained %s", name)

    primary, reason = _one_se_choice(cv_summary)

    # Sensitivity analysis: test rows whose exact feature vector also occurs in training.
    tr_arr, te_arr = X_tr.to_numpy(dtype=float), X_te.to_numpy(dtype=float)
    seen = {tuple(r) for r in tr_arr[~np.isnan(tr_arr).any(axis=1)]}
    novel = np.array([tuple(r) not in seen for r in te_arr])

    test_metrics, test_curves, thresholds = {}, {}, {}
    for name, pipe in fitted.items():
        score = pipe.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, score)
        if novel.sum() and len(np.unique(y_te[novel])) == 2:
            m["roc_auc_excluding_train_duplicates"] = compute_metrics(y_te[novel], score[novel])["roc_auc"]
            m["n_test_without_train_duplicate"] = int(novel.sum())
        test_metrics[name] = m
        test_curves[name] = curves(y_te, score)
        thresholds[name] = threshold_table(y_te, score)

    # Operating threshold: highest threshold whose out-of-fold *training* recall >= target.
    oof = cross_val_predict(fitted[primary], X_tr, y_tr, cv=cv, groups=g_tr, method="predict_proba")[:, 1]
    operating_threshold = 0.5
    for t in np.arange(0.5, 0.0, -0.01):
        if compute_metrics(y_tr, oof, round(float(t), 2))["recall_sensitivity"] >= TARGET_RECALL:
            operating_threshold = round(float(t), 2)
            break
    else:
        operating_threshold = 0.01
    p = fitted[primary].predict_proba(X_te)[:, 1]
    primary_at_operating = compute_metrics(y_te, p, operating_threshold)
    bins = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, 4)
    reliability = [{"bin": f"{bins[b]:.1f}-{bins[b + 1]:.1f}", "n": int((idx == b).sum()),
                    "mean_score": float(p[idx == b].mean()),
                    "observed_malignant_rate": float(y_te[idx == b].mean())}
                   for b in range(5) if (idx == b).any()]

    background = X_tr.sample(n=min(100, len(X_tr)), random_state=seed)
    global_importance = {n: compute_global_importance(m, X_tr, background)
                         for n, m in fitted.items() if supports_shap(m)}

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = models_dir / "model_bundle.joblib"
    with tempfile.NamedTemporaryFile(dir=models_dir, delete=False, suffix=".tmp") as tmp:
        tmp_name = tmp.name
    joblib.dump({"pipelines": fitted, "primary": primary, "background": background,
                 "features": FEATURES}, tmp_name)
    os.replace(tmp_name, bundle_path)

    metadata = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "dataset": {"path": "data/cancer.csv", "sha256": sha256_file(Path(data_path)), **quality},
        "split": {"method": "StratifiedGroupKFold(5), first fold held out, grouped by Id",
                  "n_train": int(len(train)), "n_test": int(len(test)),
                  "train_malignant": int(y_tr.sum()), "test_malignant": int(y_te.sum())},
        "features": FEATURES, "positive_class": "malignant (1)",
        "models": {n: {"label": FRIENDLY[n], "best_params": best_params[n], "cv": cv_summary.get(n)} for n in fitted},
        "primary_model": primary, "primary_selection_reason": reason,
        "test_metrics": test_metrics, "test_curves": test_curves, "threshold_tables": thresholds,
        "operating_threshold": operating_threshold, "target_recall": TARGET_RECALL,
        "operating_threshold_note": "Chosen from out-of-fold training predictions only; test set not used.",
        "primary_test_metrics_at_operating_threshold": primary_at_operating,
        "reliability_primary": reliability, "global_importance": global_importance,
        "score_semantics": "Uncalibrated model score in [0,1]; not a validated probability of malignancy.",
        "bundle_sha256": sha256_file(bundle_path),
    }
    meta_tmp = models_dir / "metadata.json.tmp"
    meta_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(meta_tmp, models_dir / "metadata.json")
    return metadata


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    md = train_pipeline(seed=args.seed)
    print(f"Primary model: {md['primary_model']}\n  {md['primary_selection_reason']}")
    print(f"{'model':22s} {'cvAUC':>7s} {'testAUC':>8s} {'recall':>7s} {'spec':>6s} {'prec':>6s} {'acc':>6s}")

    def f(v):
        return "   n/a" if v is None else f"{v:6.3f}"
    for n, m in md["test_metrics"].items():
        cv = md["models"][n]["cv"]
        print(f"{n:22s} {f(cv['roc_auc_mean'] if cv else None)} {f(m['roc_auc'])}  "
              f"{f(m['recall_sensitivity'])} {f(m['specificity'])} {f(m['precision'])} {f(m['accuracy'])}")


if __name__ == "__main__":
    main()
