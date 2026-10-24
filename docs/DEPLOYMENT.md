# Deployment

## Local (verified on Windows 11 / Python 3.11.9)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.ml_model                                   # trains + evaluates, writes data/models/ (~1 min)
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

The server can be started from any directory (paths resolve from the package location). If `data/models/` is missing the server still starts, `/health` returns 200, `/ready` returns **503** with the instruction to train, prediction routes return 503, and the UI shows a banner.

## Configuration

See `.env.example`. Variables: `BIOAI_LOG_LEVEL`, `BIOAI_RETRIEVER` (`tfidf`|`dense`), `BIOAI_ENABLE_TRAIN_ENDPOINT`, `BIOAI_MAX_BODY_BYTES`, `BIOAI_MODELS_DIR`, `BIOAI_DATA_DIR`. No secrets are needed. (The app does not auto-load `.env`; export the variables or use your process manager.)

## Docker (NOT VERIFIED - Docker is not installed on the machine used for this work)

```bash
docker build -t bioai-insight .
docker run --rm -p 8000:8000 bioai-insight
```

Design: slim Python 3.11 image, dependencies from `requirements.txt` only (no torch/streamlit/faiss), the model is **trained during the build** (`RUN python -m src.ml_model`) so the image contains hash-verified artifacts, runs as a non-root user, `HEALTHCHECK` polls `/ready`. A single-stage build is used because there is no compiler toolchain to discard (all dependencies ship wheels). Expect to debug first-build issues; the `Dockerfile` is syntactically straightforward but has not been executed.

For the dense retriever in Docker add `sentence-transformers` to the image (large) and set `BIOAI_RETRIEVER=dense`; the model downloads at first start unless baked in.

## Health endpoints

| Route | Meaning |
|---|---|
| `GET /health` | liveness: process is up |
| `GET /ready` | 200 only if the model bundle loaded and passed its integrity check; body lists loaded models and whether the RAG index has passages |

## CI

`.github/workflows/ci.yml` runs on push/PR: install dev requirements, `ruff check`, train, `pytest`, `node --check`. **Not executed yet** (no GitHub run was triggered from this machine).

## Not provided

No hosted demo, TLS, auth, rate limiting, autoscaling or monitoring stack; none were requested and none are claimed.
