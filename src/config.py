"""Central configuration. All paths resolve relative to the repository root,
so the app works from any working directory."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("BIOAI_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("BIOAI_MODELS_DIR", DATA_DIR / "models"))
PAPERS_DIR = Path(os.getenv("BIOAI_PAPERS_DIR", PROJECT_ROOT / "papers"))
STATIC_DIR = PROJECT_ROOT / "static"
DATASET_PATH = DATA_DIR / "cancer.csv"
SOURCES_PATH = PAPERS_DIR / "sources.json"

BUNDLE_PATH = MODELS_DIR / "model_bundle.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

SEED = 48
MAX_QUERY_CHARS = 500
MAX_BATCH_ROWS = 1000
MAX_BODY_BYTES = int(os.getenv("BIOAI_MAX_BODY_BYTES", 512 * 1024))
ENABLE_TRAIN_ENDPOINT = os.getenv("BIOAI_ENABLE_TRAIN_ENDPOINT", "0") == "1"
