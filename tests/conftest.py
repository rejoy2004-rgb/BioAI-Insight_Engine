"""Shared fixtures. Everything here is deterministic, offline and small."""
import json

import pytest

from bioai_testutil import INJECTION, make_dataset


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    make_dataset().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained(synthetic_csv, tmp_path_factory):
    from src.ml_model import train_pipeline
    out = tmp_path_factory.mktemp("models")
    meta = train_pipeline(seed=7, data_path=synthetic_csv, models_dir=out, quick=True)
    return out, meta


@pytest.fixture(scope="session")
def service(trained):
    from src.service import ModelService
    svc = ModelService(trained[0])
    assert svc.load(), svc.load_error
    return svc


@pytest.fixture()
def client(service, tmp_path):
    """TestClient with the fast synthetic-trained service and a tiny corpus injected after startup."""
    from fastapi.testclient import TestClient

    from src.main import app
    from src.rag_engine import RagEngine
    (tmp_path / "doc.txt").write_text(
        "Mammography screening reduces mortality from breast cancer in women over fifty years old. "
        + INJECTION + " BRCA1 and BRCA2 gene mutations are associated with hereditary breast cancer risk in families.",
        encoding="utf-8")
    (tmp_path / "sources.json").write_text(json.dumps([{"id": "t1", "file": "doc.txt", "title": "Test Doc", "authors": "A. Author",
                                                         "journal": "Test J", "year": 2020}]), encoding="utf-8")
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.models = service
        rag = RagEngine(tmp_path / "sources.json")
        rag.build()
        app.state.rag = rag
        yield c
