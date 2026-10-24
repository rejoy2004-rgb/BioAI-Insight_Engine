"""FastAPI application: ML inference + explainability + scientific retrieval.

Run:  uvicorn src.main:app --host 127.0.0.1 --port 8000
Educational/research prototype - not a validated clinical diagnostic tool.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import config
from src.preprocessing import DataValidationError
from src.rag_engine import QueryError, RagEngine
from src.schemas import BatchPredictionRequest, BatchPredictionResponse, PatientFeatures, Prediction, QueryRequest, TrainRequest
from src.service import ModelService, ModelUnavailable

log = logging.getLogger("bioai.api")
VERSION = "2.0.0"
_train_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=os.getenv("BIOAI_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app.state.models = ModelService()
    app.state.rag = RagEngine()
    app.state.models.load()
    app.state.rag.build()
    yield


app = FastAPI(
    title="BioAI Insight Engine API", version=VERSION, lifespan=lifespan,
    description="Explainable ML on the Wisconsin breast-biopsy dataset and cited retrieval over a small "
                "scientific corpus. Educational prototype - not a clinical diagnostic tool.",
)

CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
# Swagger UI (/docs) loads assets from a CDN, so it gets a relaxed policy.
DOCS_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    request.state.request_id = rid
    start = time.perf_counter()
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > config.MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request body too large.", "request_id": rid}, status_code=413)
    if request.method in {"POST", "PUT", "PATCH"} and not declared:
        body = await request.body()
        if len(body) > config.MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request body too large.", "request_id": rid}, status_code=413)
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path not in DOCS_PATHS:
        response.headers["Content-Security-Policy"] = CSP
    if request.url.path.startswith("/static/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache"
    log.info("req=%s %s %s -> %s %.0fms", rid, request.method, request.url.path, response.status_code, ms)
    return response


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse({"detail": "Invalid request.", "errors": errors, "request_id": _rid(request)}, status_code=422)


@app.exception_handler(DataValidationError)
async def data_validation_handler(request: Request, exc: DataValidationError):
    return JSONResponse({"detail": str(exc), "request_id": _rid(request)}, status_code=422)


@app.exception_handler(ModelUnavailable)
async def model_unavailable_handler(request: Request, exc: ModelUnavailable):
    return JSONResponse({"detail": str(exc), "request_id": _rid(request)}, status_code=503)


@app.exception_handler(QueryError)
async def query_error_handler(request: Request, exc: QueryError):
    return JSONResponse({"detail": str(exc), "request_id": _rid(request)}, status_code=422)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled error req=%s", _rid(request))   # traceback goes to logs only
    return JSONResponse({"detail": "Internal server error.", "request_id": _rid(request)}, status_code=500)


def _models(request: Request) -> ModelService:
    svc: ModelService = request.app.state.models
    if not svc.ready:
        raise ModelUnavailable(svc.load_error or "Models not loaded.")
    return svc


# ---------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


# ---------------------------------------------------------------- health
@app.get("/health")
def health():
    """Liveness only: the process is up."""
    return {"status": "ok", "version": VERSION}


@app.get("/ready")
def ready(request: Request):
    """Readiness: required model artifacts loaded. 503 otherwise."""
    models: ModelService = request.app.state.models
    rag: RagEngine = request.app.state.rag
    body = {
        "status": "ready" if models.ready else "not_ready",
        "version": VERSION,
        "models_loaded": sorted(models.pipelines),
        "primary_model": models.primary,
        "rag_ready": rag.ready,
        "rag_chunks": len(rag.retriever),
        "training_enabled": config.ENABLE_TRAIN_ENDPOINT,
        "detail": None if models.ready else models.load_error,
    }
    return JSONResponse(body, status_code=200 if models.ready else 503)


# ---------------------------------------------------------------- ML
@app.post("/predict", response_model=Prediction)
def predict(features: PatientFeatures, request: Request):
    svc = _models(request)
    t0 = time.perf_counter()
    result = svc.predict([features.as_row()], explain=True)[0]
    log.info("predict req=%s model=%s %.0fms", _rid(request), svc.primary, (time.perf_counter() - t0) * 1000)
    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(body: BatchPredictionRequest, request: Request):
    svc = _models(request)
    preds = svc.predict([p.as_row() for p in body.patients], explain=False)
    return {"total_records": len(preds), "predictions": preds}


@app.get("/model/metrics")
def model_metrics(request: Request):
    return _models(request).public_metrics()


@app.get("/explain/global")
def explain_global(request: Request, model_name: str | None = None):
    svc = _models(request)
    name = model_name or svc.primary
    imp = svc.metadata.get("global_importance", {}).get(name)
    if imp is None:
        raise HTTPException(status_code=404, detail=f"Global explanations are not available for model '{name}'.")
    return {"model": name, **imp,
            "note": "Mean |SHAP| describes model behaviour on the training data, not biological causality."}


# ---------------------------------------------------------------- RAG
@app.post("/rag/retrieve")
def rag_retrieve(body: QueryRequest, request: Request):
    rag: RagEngine = request.app.state.rag
    t0 = time.perf_counter()
    results = rag.retrieve(body.query, body.k)
    ms = (time.perf_counter() - t0) * 1000
    log.info("rag.retrieve req=%s n=%d %.0fms", _rid(request), len(results), ms)
    return {"latency_ms": round(ms, 2), "count": len(results), "results": results}


@app.post("/rag/ask")
def rag_ask(body: QueryRequest, request: Request):
    rag: RagEngine = request.app.state.rag
    t0 = time.perf_counter()
    out = rag.ask(body.query, body.k)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    log.info("rag.ask req=%s status=%s %.0fms", _rid(request), out["status"], out["latency_ms"])
    return out


@app.get("/rag/stats")
def rag_stats(request: Request):
    return request.app.state.rag.stats()


# ---------------------------------------------------------------- training (opt-in)
if config.ENABLE_TRAIN_ENDPOINT:
    @app.post("/train")
    def train(body: TrainRequest, request: Request):
        """Disabled unless BIOAI_ENABLE_TRAIN_ENDPOINT=1. Unauthenticated: local demos only."""
        from src.ml_model import train_pipeline
        if not _train_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Training already in progress.")
        try:
            md = train_pipeline(seed=body.seed)
            request.app.state.models.load()
        finally:
            _train_lock.release()
        return {"message": "Training completed.", "seed": body.seed, "primary_model": md["primary_model"]}
