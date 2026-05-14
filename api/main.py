"""
api/main.py
===========
FastAPI backend exposing the BioAI prediction model as a REST API.

WHY THIS FILE EXISTS:
  A Streamlit app is great for demos, but production systems need a language-
  agnostic API that any frontend (React, mobile app, EHR system) can call.
  FastAPI is the modern Python API framework: async, fast, and auto-documented.

INTERVIEW VALUE:
  Building an API around an ML model (model serving) is a key ML Engineering
  skill. This demonstrates: Pydantic validation, dependency injection, 
  proper HTTP status codes, and OpenAPI documentation.

USAGE:
  uvicorn api.main:app --reload --port 8000
  
  Then visit http://localhost:8000/docs for the auto-generated Swagger UI.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path when running via uvicorn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.predict import predict, get_model_info
from src.preprocessing import get_feature_names

# ─── App Initialisation ──────────────────────────────────────────────────────

app = FastAPI(
    title="BioAI Insight Engine API",
    description=(
        "REST API for cancer risk prediction using ML models trained on the "
        "Wisconsin Breast Cancer dataset. "
        "Run 'python src/train.py' before starting the server."
    ),
    version="1.0.0",
    docs_url="/docs",          # Swagger UI
    redoc_url="/redoc",        # ReDoc UI
)

# Allow CORS so the Streamlit frontend (different port) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track server start time for uptime reporting in /health
_START_TIME = time.time()


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Schema for incoming prediction requests.

    Each field maps to one of the 9 biological features in the dataset.
    Values should be on the original 1–10 scale (scaler handles normalisation).
    """
    Cl_thickness:   float = Field(..., ge=1, le=10, description="Clump Thickness (1-10)")
    Cell_size:      float = Field(..., ge=1, le=10, description="Uniformity of Cell Size (1-10)")
    Cell_shape:     float = Field(..., ge=1, le=10, description="Uniformity of Cell Shape (1-10)")
    Marg_adhesion:  float = Field(..., ge=1, le=10, description="Marginal Adhesion (1-10)")
    Epith_c_size:   float = Field(..., ge=1, le=10, description="Single Epithelial Cell Size (1-10)")
    Bare_nuclei:    float = Field(..., ge=1, le=10, description="Bare Nuclei (1-10)")
    Bl_cromatin:    float = Field(..., ge=1, le=10, description="Bland Chromatin (1-10)")
    Normal_nucleoli:float = Field(..., ge=1, le=10, description="Normal Nucleoli (1-10)")
    Mitoses:        float = Field(..., ge=1, le=10, description="Mitoses (1-10)")

    class Config:
        json_schema_extra = {
            "example": {
                "Cl_thickness":    5,
                "Cell_size":       1,
                "Cell_shape":      1,
                "Marg_adhesion":   1,
                "Epith_c_size":    2,
                "Bare_nuclei":     1,
                "Bl_cromatin":     3,
                "Normal_nucleoli": 1,
                "Mitoses":         1,
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction responses returned to the client."""
    prediction:    int   = Field(..., description="0 = Benign, 1 = Malignant")
    label:         str   = Field(..., description="'Benign' or 'Malignant'")
    confidence:    float = Field(..., description="Model confidence (0–1)")
    model_used:    str   = Field(..., description="Name of the model that made the prediction")
    request_id:    str   = Field(..., description="Unique identifier for this request")


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    uptime_s:     float
    model_info:   Optional[Dict[str, Any]]


class BatchPredictionRequest(BaseModel):
    samples: List[PredictionRequest] = Field(..., min_items=1, max_items=100)


# ─── Helper ──────────────────────────────────────────────────────────────────

def _request_to_features(req: PredictionRequest) -> List[float]:
    """Converts a PredictionRequest to the ordered feature list expected by predict()."""
    return [
        req.Cl_thickness,
        req.Cell_size,
        req.Cell_shape,
        req.Marg_adhesion,
        req.Epith_c_size,
        req.Bare_nuclei,
        req.Bl_cromatin,
        req.Normal_nucleoli,
        req.Mitoses,
    ]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Welcome message and link to docs."""
    return {
        "message": "BioAI Insight Engine API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """
    Health check endpoint.

    Tells you whether the API is running, whether the model is loaded,
    and how long the server has been up.

    Use this in Docker healthchecks and CI/CD pipelines to verify readiness.
    """
    model_loaded = True
    model_info = None

    try:
        model_info = get_model_info()
    except FileNotFoundError:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        uptime_s=round(time.time() - _START_TIME, 2),
        model_info=model_info,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Predict cancer risk for a single patient",
)
async def predict_endpoint(request: PredictionRequest):
    """
    Predicts whether a tissue sample is **Benign (0)** or **Malignant (1)**.

    All feature values should be on the **original 1–10 scale** from the
    Wisconsin Breast Cancer dataset. The API handles scaling internally.

    Returns the prediction label, confidence score, and the model used.
    """
    import uuid

    try:
        features = _request_to_features(request)
        result = predict(features)
        return PredictionResponse(
            **result,
            request_id=str(uuid.uuid4())[:8],
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@app.post(
    "/predict/batch",
    tags=["Prediction"],
    summary="Predict cancer risk for multiple patients at once",
)
async def batch_predict_endpoint(request: BatchPredictionRequest):
    """
    Batch prediction for up to 100 patients in a single API call.

    Returns a list of predictions in the same order as the input samples.
    """
    import uuid

    try:
        results = []
        for sample in request.samples:
            features = _request_to_features(sample)
            result = predict(features)
            result["request_id"] = str(uuid.uuid4())[:8]
            results.append(result)
        return {"predictions": results, "count": len(results)}

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", tags=["Monitoring"])
async def model_info_endpoint():
    """Returns metadata about the currently loaded model."""
    try:
        return get_model_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/features", tags=["Metadata"])
async def feature_names_endpoint():
    """Returns the list of feature names expected by the /predict endpoint."""
    try:
        names = get_feature_names()
        return {"features": names, "count": len(names)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
