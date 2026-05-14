# 🧬 BioAI Insight Engine

> **Production-style cancer risk prediction system** combining ML model comparison,
> Explainable AI (SHAP), a REST API (FastAPI), and a Research Paper Assistant (RAG).

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Training the Model](#-training-the-model)
- [API Usage](#-api-usage)
- [Streamlit Frontend](#-streamlit-frontend)
- [Docker Deployment](#-docker-deployment)
- [Results](#-results)
- [Experiment Tracking](#-experiment-tracking-wandb)
- [Future Improvements](#-future-improvements)

---

## 🔬 Project Overview

BioAI Insight Engine is an end-to-end applied AI system for cancer risk classification.
It transforms raw biological cell measurements into actionable risk predictions with
full model explainability and research paper search capabilities.

**Key highlights:**
- 🏗️ **Modular ML pipeline** — preprocessing, training, and inference are fully decoupled
- 🔒 **No data leakage** — scaler is fit only on training data; test split is held out correctly
- 🔄 **Model persistence** — models saved to disk; zero retraining on app startup
- 📡 **Production API** — FastAPI with Pydantic validation, Swagger docs, and batch inference
- 🧠 **Explainable AI** — SHAP values for both global feature importance and individual predictions
- 📚 **RAG system** — semantic search over research PDFs using FAISS + SentenceTransformers
- 🐳 **Docker-ready** — single command to build and run the full system
- 📊 **Experiment tracking** — Weights & Biases integration for comparing training runs

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     BioAI Insight Engine                         │
├───────────────────────────┬──────────────────────────────────────┤
│    TRAINING PIPELINE      │         SERVING LAYER                │
│                           │                                      │
│  data/cancer.csv          │   ┌─────────────────────────┐       │
│       │                   │   │   FastAPI (port 8000)    │       │
│       ▼                   │   │  POST /predict           │       │
│  preprocessing.py         │   │  GET  /health            │       │
│  (train/test split,       │   │  GET  /model-info        │       │
│   fit scaler)             │   └────────────┬────────────┘       │
│       │                   │                │                     │
│       ▼                   │                ▼                     │
│  model_comparison.py      │   ┌─────────────────────────┐       │
│  (LR, SVM, RF)            │   │   predict.py             │       │
│       │                   │   │   (cached model load,    │       │
│       ▼                   │   │    scale → infer)        │       │
│  train.py ──────────────► │   └────────────┬────────────┘       │
│  models/model.pkl         │                │                     │
│  models/scaler.pkl        │   ┌────────────▼────────────┐       │
│  outputs/*.png            │   │ Streamlit (port 8501)    │       │
│                           │   │  Tab 1: Prediction       │       │
│  RAG PIPELINE             │   │  Tab 2: SHAP             │       │
│  load_pdf()               │   │  Tab 3: RAG Search       │       │
│  create_chunks()          │   └─────────────────────────┘       │
│  create_embeddings()      │                                      │
│  build_faiss_index()      │                                      │
│  search()                 │                                      │
└───────────────────────────┴──────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Frontend | Streamlit |
| ML Models | scikit-learn (LR, SVM, RandomForest) |
| Model Persistence | joblib |
| Explainability | SHAP |
| Vector Search | FAISS |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| PDF Parsing | PyMuPDF (fitz) |
| Experiment Tracking | Weights & Biases |
| Containerisation | Docker |
| Data | Wisconsin Breast Cancer Dataset |

---

## 📁 Project Structure

```
BioAI-Insight_Engine/
│
├── api/
│   └── main.py              # FastAPI app — /predict, /health, /batch
│
├── frontend/
│   └── app.py               # Streamlit UI — 3 tabs
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Data loading, train/test split, scaler
│   ├── train.py             # Training entry point — run once
│   ├── predict.py           # Inference — loads saved model from disk
│   ├── model_comparison.py  # LR vs SVM vs RF comparison pipeline
│   ├── explainability.py    # SHAP plots and feature importance
│   └── rag_engine.py        # PDF → chunks → embeddings → FAISS search
│
├── models/                  # Saved model.pkl and scaler.pkl (git-ignored)
├── outputs/                 # ROC curves, confusion matrices, SHAP plots
├── data/
│   └── cancer.csv
├── papers/
│   └── sample.pdf
│
├── requirements.txt         # Pinned dependencies
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/BioAI-Insight-Engine.git
cd BioAI-Insight-Engine
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Train all models, select the best by ROC-AUC, and save it:

```bash
python src/train.py
```

With Weights & Biases experiment tracking:

```bash
python src/train.py --wandb
```

This will:
- Preprocess `data/cancer.csv` with a proper 80/20 train/test split
- Train Logistic Regression, SVM, and Random Forest
- Save evaluation plots to `outputs/`
- Save `models/model.pkl` and `models/scaler.pkl`

---

## 📡 API Usage

### Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

### Interactive docs

Visit `http://localhost:8000/docs` for the full Swagger UI.

### Health check

```bash
curl http://localhost:8000/health
```

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Cl_thickness": 5,
    "Cell_size": 1,
    "Cell_shape": 1,
    "Marg_adhesion": 1,
    "Epith_c_size": 2,
    "Bare_nuclei": 1,
    "Bl_cromatin": 3,
    "Normal_nucleoli": 1,
    "Mitoses": 1
  }'
```

**Response:**

```json
{
  "prediction": 0,
  "label": "Benign",
  "confidence": 0.9832,
  "model_used": "Random Forest",
  "request_id": "a3f2b1c0"
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"samples": [{"Cl_thickness":5,...}, {"Cl_thickness":8,...}]}'
```

---

## 🖥️ Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501`

**Tab 1 — Prediction:** Adjust feature sliders, run inference, see confidence score and ROC curves.

**Tab 2 — Explainability:** Global SHAP summary plot and per-patient feature attributions.

**Tab 3 — Research Assistant:** Ask questions about your PDF research paper using semantic search.

---

## 🐳 Docker Deployment

```bash
# Build image (trains model at build time)
docker build -t bioai .

# Run FastAPI
docker run -p 8000:8000 bioai

# Run Streamlit
docker run -p 8501:8501 bioai streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.971 | ~0.972 | ~0.958 | ~0.965 | ~0.995 |
| SVM | ~0.978 | ~0.981 | ~0.965 | ~0.973 | ~0.997 |
| **Random Forest** | **~0.985** | **~0.986** | **~0.979** | **~0.982** | **~0.998** |

*Results vary slightly by random seed. Run `python src/train.py` for exact metrics on your split.*

---

## 📈 Experiment Tracking (W&B)

```bash
wandb login
python src/train.py --wandb
```

Tracks: model name, accuracy, ROC-AUC, precision, recall, F1 per run.

---

## 🚀 Future Improvements

- [ ] LLM-powered answer generation (pass RAG chunks to Claude/GPT-4)
- [ ] SHAP waterfall plots integrated into the Streamlit UI
- [ ] Patient CSV upload for batch predictions
- [ ] PDF report generation (downloadable risk report)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Kubernetes deployment manifests
- [ ] MLflow as an alternative experiment tracker
- [ ] Model monitoring (data drift detection with Evidently AI)

---

## 👤 Author

**Rejoy Besra** — B.Tech Biotechnology, IIT Kharagpur  
AI & ML Engineer | GenAI Enthusiast

---

*This project is for research and educational purposes only. It does not constitute medical advice.*
