# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — BioAI Insight Engine
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY THIS FILE EXISTS:
#   Docker ensures the app runs identically on any machine — your laptop, a
#   cloud VM, or a Kubernetes cluster. This eliminates "works on my machine"
#   problems, which is a key concern in ML deployment.
#
# BUILD & RUN:
#   docker build -t bioai .
#   docker run -p 8000:8000 bioai            # starts FastAPI
#   docker run -p 8501:8501 bioai streamlit  # starts Streamlit
#
# USAGE:
#   curl http://localhost:8000/health
# ─────────────────────────────────────────────────────────────────────────────

# Use a slim Python 3.10 base image
FROM python:3.10-slim

# Metadata labels (best practice for production images)
LABEL maintainer="BioAI Insight Engine"
LABEL description="Cancer risk prediction API + Streamlit frontend"
LABEL version="1.0.0"

# Set working directory inside the container
WORKDIR /app

# System dependencies required by PyMuPDF, FAISS, and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer cache: only re-install if requirements change)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project (respects .dockerignore)
COPY . .

# Create directories expected by the app
RUN mkdir -p models outputs data papers

# Train the model at build time so the container starts with a pre-trained model
# Comment this out if you want to volume-mount pre-trained models instead
RUN python src/train.py || echo "Training failed — start container and run manually"

# Expose ports
EXPOSE 8000   
# FastAPI (uvicorn)
EXPOSE 8501   
# Streamlit

# Default: start FastAPI
# Override with: docker run bioai streamlit run frontend/app.py --server.port 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─── Health Check ─────────────────────────────────────────────────────────────
# Docker will mark the container unhealthy if /health returns non-200
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
