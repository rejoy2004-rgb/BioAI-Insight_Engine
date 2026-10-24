# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Wheels exist for every dependency, so no compiler toolchain is needed.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/
COPY data/cancer.csv ./data/cancer.csv
COPY papers/ ./papers/

# Train at build time: artifacts are reproducible from the dataset and carry a SHA-256 integrity hash.
RUN python -m src.ml_model

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
