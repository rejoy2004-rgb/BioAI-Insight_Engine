import os

if os.getenv("BIOAI_RETRIEVER", "").lower() == "dense":
    # On some Windows installs torch fails to load its DLLs if scikit-learn/numpy are imported first.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
