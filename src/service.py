"""Inference service: loads the trusted, hash-verified model bundle once and
serves predictions/explanations. Pickled artifacts are build outputs of
`python -m src.ml_model`; never load user-supplied pickles."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.explainability import Explainer, ExplanationUnavailable, supports_shap
from src.ml_model import sha256_file
from src.preprocessing import CLASS_LABELS, FEATURES, DataValidationError, validate_dataframe

log = logging.getLogger("bioai.service")


class ModelUnavailable(RuntimeError):
    pass


class ModelService:
    def __init__(self, models_dir: Path | str = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.pipelines: dict = {}
        self.primary: str | None = None
        self.metadata: dict = {}
        self.explainers: dict[str, Explainer] = {}
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return bool(self.pipelines) and self.primary in self.pipelines

    def load(self) -> bool:
        """Load and verify artifacts. Never raises; sets load_error on failure."""
        self.pipelines, self.primary, self.metadata, self.explainers = {}, None, {}, {}
        meta_path, bundle_path = self.models_dir / "metadata.json", self.models_dir / "model_bundle.joblib"
        try:
            if not meta_path.exists() or not bundle_path.exists():
                raise ModelUnavailable("Model artifacts not found. Run `python -m src.ml_model`.")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if sha256_file(bundle_path) != metadata.get("bundle_sha256"):
                raise ModelUnavailable("Model bundle failed integrity check (hash mismatch).")
            bundle = joblib.load(bundle_path)
            if list(bundle["features"]) != FEATURES or list(metadata["features"]) != FEATURES:
                raise ModelUnavailable("Model feature list does not match the application schema.")
            self.pipelines, self.primary, self.metadata = bundle["pipelines"], bundle["primary"], metadata
            for name, pipe in self.pipelines.items():
                if supports_shap(pipe):
                    try:
                        self.explainers[name] = Explainer(pipe, bundle.get("background"))
                    except ExplanationUnavailable as exc:
                        log.warning("explainer unavailable for %s: %s", name, exc)
            self.load_error = None
            log.info("model bundle loaded; primary=%s", self.primary)
            return True
        except Exception as exc:
            self.pipelines, self.primary = {}, None
            self.load_error = str(exc) if isinstance(exc, ModelUnavailable) else f"Failed to load models ({type(exc).__name__})."
            log.error("model load failed: %s", self.load_error)
            return False

    # ---- inference -------------------------------------------------------
    @staticmethod
    def to_frame(rows: list[dict]) -> pd.DataFrame:
        """Build a frame in canonical feature order; reject missing/extra/NaN."""
        if not rows:
            raise DataValidationError("No rows supplied.")
        df = pd.DataFrame(rows)
        extra = set(df.columns) - set(FEATURES)
        if extra:
            raise DataValidationError(f"Unexpected columns: {sorted(extra)}")
        df = df.reindex(columns=FEATURES)
        if df.isna().any().any():
            raise DataValidationError("All nine features are required for prediction; missing values are not imputed.")
        validate_dataframe(df.astype(float), require_target=False)
        return df.astype(float)

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("operating_threshold", 0.5))

    def predict(self, rows: list[dict], explain: bool = False) -> list[dict]:
        if not self.ready:
            raise ModelUnavailable(self.load_error or "Models not loaded.")
        X = self.to_frame(rows)
        scores = {}
        for name, pipe in self.pipelines.items():
            s = pipe.predict_proba(X)[:, 1]
            if not np.isfinite(s).all() or (s < 0).any() or (s > 1).any():
                raise RuntimeError(f"Model '{name}' produced invalid scores.")
            scores[name] = s
        out = []
        for i in range(len(X)):
            primary_score = float(scores[self.primary][i])
            pred = int(primary_score >= self.threshold)
            item = {
                "prediction": pred,
                "prediction_label": f"{CLASS_LABELS[pred].capitalize()}-class prediction",
                "malignancy_score": primary_score,
                "threshold": self.threshold,
                "primary_model": self.primary,
                "model_scores": {n: float(s[i]) for n, s in scores.items() if n != "baseline"},
            }
            if explain:
                item["explanation"] = self.explain(X.iloc[[i]])
            out.append(item)
        return out

    def explain(self, X: pd.DataFrame) -> dict:
        ex = self.explainers.get(self.primary)
        if ex is None:
            return {"status": "unavailable", "reason": f"No SHAP explainer for model '{self.primary}'."}
        try:
            return {"status": "ok", **ex.explain_one(X)}
        except ExplanationUnavailable as exc:
            log.warning("explanation failed: %s", exc)
            return {"status": "unavailable", "reason": str(exc)}

    def public_metrics(self) -> dict:
        if not self.metadata:
            raise ModelUnavailable(self.load_error or "Models not loaded.")
        keys = ["created_utc", "seed", "dataset", "split", "features", "positive_class", "models",
                "primary_model", "primary_selection_reason", "test_metrics", "test_curves",
                "threshold_tables", "operating_threshold", "target_recall", "operating_threshold_note",
                "primary_test_metrics_at_operating_threshold", "reliability_primary",
                "global_importance", "score_semantics"]
        out = {k: self.metadata[k] for k in keys if k in self.metadata}
        out["dataset"] = {k: v for k, v in out["dataset"].items() if k != "path"}
        return out
