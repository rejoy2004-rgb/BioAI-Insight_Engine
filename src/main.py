import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.preprocessing import get_scaler, load_data
from src.explainability import explain_instance, get_global_shap_importance
from src.rag_engine import retrieve, load_or_build_index
from src.ml_model import train_pipeline, MODELS_DIR

app = FastAPI(
    title="BioAI Insight Engine API", 
    description="Cancer Risk Prediction (Logistic Regression, SVM, Random Forest) with SHAP Explainability & RAG Search Engine",
    version="1.0.0"
)

# Ensure static files directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

models = {}
scaler = None

def load_resources():
    global scaler
    print("Loading model resources...")
    scaler = get_scaler()
    
    for name in ["logistic_regression", "svm", "random_forest"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
                print(f"Loaded model: {name}")
        else:
            print(f"Model file not found: {path}")
            
    # Load RAG index
    try:
        load_or_build_index()
    except Exception as e:
        print(f"Error loading RAG index: {e}")

@app.on_event("startup")
def startup_event():
    load_resources()

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Schemas
class PatientFeatures(BaseModel):
    Cl_thickness: float = Field(..., alias="Cl.thickness", ge=1, le=10)
    Cell_size: float = Field(..., alias="Cell.size", ge=1, le=10)
    Cell_shape: float = Field(..., alias="Cell.shape", ge=1, le=10)
    Marg_adhesion: float = Field(..., alias="Marg.adhesion", ge=1, le=10)
    Epith_c_size: float = Field(..., alias="Epith.c.size", ge=1, le=10)
    Bare_nuclei: float = Field(..., alias="Bare.nuclei", ge=1, le=10)
    Bl_cromatin: float = Field(..., alias="Bl.cromatin", ge=1, le=10)
    Normal_nucleoli: float = Field(..., alias="Normal.nucleoli", ge=1, le=10)
    Mitoses: float = Field(..., alias="Mitoses", ge=1, le=10)

    model_config = {
        "populate_by_name": True,
        "allow_population_by_field_name": True
    }

class BatchPredictionRequest(BaseModel):
    patients: List[PatientFeatures]

class QueryRequest(BaseModel):
    query: str
    k: int = 3

class TrainRequest(BaseModel):
    seed: int = 48

# Endpoints
@app.get("/health")
def health_check():
    global scaler
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    loaded_metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                loaded_metrics = json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            
    from src.rag_engine import DOCUMENTS_CACHE_PATH
    rag_doc_count = 0
    if os.path.exists(DOCUMENTS_CACHE_PATH):
        try:
            with open(DOCUMENTS_CACHE_PATH, "r") as f:
                rag_doc_count = len(json.load(f))
        except Exception as e:
            print(f"Error reading documents cache: {e}")

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": list(models.keys()),
        "scaler_loaded": scaler is not None,
        "rag_document_count": rag_doc_count,
        "model_performance": loaded_metrics
    }

@app.post("/predict")
def predict_risk(features: PatientFeatures):
    if not models or scaler is None:
        load_resources()
        if not models or scaler is None:
            raise HTTPException(
                status_code=503, 
                detail="Models or scaler not loaded. Please train the models first."
            )
            
    # Extract feature values in exact sequence of train columns
    raw_values = [
        features.Cl_thickness,
        features.Cell_size,
        features.Cell_shape,
        features.Marg_adhesion,
        features.Epith_c_size,
        features.Bare_nuclei,
        features.Bl_cromatin,
        features.Normal_nucleoli,
        features.Mitoses
    ]
    
    feature_names = [
        "Cl.thickness", "Cell.size", "Cell.shape", "Marg.adhesion", 
        "Epith.c.size", "Bare.nuclei", "Bl.cromatin", "Normal.nucleoli", "Mitoses"
    ]
    sample_df = pd.DataFrame([raw_values], columns=feature_names)
    sample_scaled = scaler.transform(sample_df)
    
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        try:
            pred = int(model.predict(sample_scaled)[0])
            prob = float(model.predict_proba(sample_scaled)[0][1])
            predictions[name] = pred
            probabilities[name] = prob
        except Exception as e:
            print(f"Error predicting with model {name}: {e}")
            
    primary_model = "random_forest"
    if primary_model not in models:
        primary_model = list(models.keys())[0] if models else None
        
    if primary_model is None:
        raise HTTPException(status_code=500, detail="No models loaded for prediction.")
        
    final_pred = predictions.get(primary_model, 0)
    final_conf = probabilities.get(primary_model, 0.0)
    
    # Local SHAP Explanation (Random Forest is default, falls back to Logistic Regression)
    shap_explanation = None
    try:
        shap_explanation = explain_instance(sample_scaled[0], model_name=primary_model)
        # Fallback to logistic regression explanation if RF fails
        if shap_explanation is None and "logistic_regression" in models:
            shap_explanation = explain_instance(sample_scaled[0], model_name="logistic_regression")
    except Exception as e:
        print(f"Error running local SHAP explanation: {e}")
        
    feature_names = [
        "Cl.thickness", "Cell.size", "Cell.shape", "Marg.adhesion", 
        "Epith.c.size", "Bare.nuclei", "Bl.cromatin", "Normal.nucleoli", "Mitoses"
    ]
    
    shap_attributions = []
    if shap_explanation and "shap_values" in shap_explanation:
        for idx, name in enumerate(feature_names):
            shap_attributions.append({
                "feature": name,
                "value": float(raw_values[idx]),
                "shap_value": float(shap_explanation["shap_values"][idx])
            })
            
    return {
        "prediction": final_pred,
        "prediction_label": "High Cancer Risk" if final_pred == 1 else "Low Cancer Risk",
        "confidence_score": final_conf,
        "model_probabilities": probabilities,
        "shap_base_value": shap_explanation.get("base_value") if shap_explanation else None,
        "shap_attributions": shap_attributions
    }

@app.post("/predict/batch")
def predict_risk_batch(request: BatchPredictionRequest):
    if not models or scaler is None:
        load_resources()
        if not models or scaler is None:
            raise HTTPException(status_code=503, detail="Models or scaler not available.")
            
    results = []
    
    for features in request.patients:
        raw_values = [
            features.Cl_thickness,
            features.Cell_size,
            features.Cell_shape,
            features.Marg_adhesion,
            features.Epith_c_size,
            features.Bare_nuclei,
            features.Bl_cromatin,
            features.Normal_nucleoli,
            features.Mitoses
        ]
        
        feature_names = [
            "Cl.thickness", "Cell.size", "Cell.shape", "Marg.adhesion", 
            "Epith.c.size", "Bare.nuclei", "Bl.cromatin", "Normal.nucleoli", "Mitoses"
        ]
        sample_df = pd.DataFrame([raw_values], columns=feature_names)
        sample_scaled = scaler.transform(sample_df)
        
        probabilities = {}
        for name, model in models.items():
            try:
                probabilities[name] = float(model.predict_proba(sample_scaled)[0][1])
            except Exception as e:
                probabilities[name] = 0.0
                
        rf_prob = probabilities.get("random_forest", 0.0)
        pred = 1 if rf_prob >= 0.5 else 0
        
        results.append({
            "prediction": pred,
            "prediction_label": "High Cancer Risk" if pred == 1 else "Low Cancer Risk",
            "confidence_score": rf_prob,
            "model_probabilities": probabilities
        })
        
    return {
        "total_records": len(results),
        "predictions": results
    }

@app.get("/explain/global")
def get_global_explanations(model_name: str = "random_forest"):
    try:
        importances = get_global_shap_importance(model_name)
        if importances is None:
            raise HTTPException(status_code=404, detail=f"Global explanations not available for model '{model_name}'.")
        return importances
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/retrieve")
def retrieve_papers(request: QueryRequest):
    start_time = time.time()
    try:
        results = retrieve(request.query, request.k)
        latency_ms = (time.time() - start_time) * 1000
        return {
            "query": request.query,
            "latency_ms": latency_ms,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_model_endpoint(request: TrainRequest):
    try:
        metrics = train_pipeline(seed=request.seed)
        load_resources()
        return {
            "message": "Model training pipeline completed successfully.",
            "seed_used": request.seed,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
