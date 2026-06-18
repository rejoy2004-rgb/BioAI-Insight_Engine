import time
from fastapi.testclient import TestClient
from src.main import app

def test_health(client):
    print("Testing GET /health...")
    start = time.time()
    response = client.get("/health")
    latency = (time.time() - start) * 1000
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "logistic_regression" in data["models_loaded"]
    assert "svm" in data["models_loaded"]
    assert "random_forest" in data["models_loaded"]
    print(f"[SUCCESS] /health passed. Latency: {latency:.2f}ms")

def test_predict(client):
    print("Testing POST /predict...")
    payload = {
        "Cl.thickness": 5.0,
        "Cell.size": 3.0,
        "Cell.shape": 3.0,
        "Marg.adhesion": 1.0,
        "Epith.c.size": 2.0,
        "Bare.nuclei": 1.0,
        "Bl.cromatin": 3.0,
        "Normal.nucleoli": 1.0,
        "Mitoses": 1.0
    }
    
    start = time.time()
    response = client.post("/predict", json=payload)
    latency = (time.time() - start) * 1000
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence_score" in data
    assert "shap_attributions" in data
    assert len(data["shap_attributions"]) == 9
    print(f"[SUCCESS] /predict passed. Latency: {latency:.2f}ms. Label: {data['prediction_label']}, Conf: {data['confidence_score']:.4f}")

def test_predict_batch(client):
    print("Testing POST /predict/batch...")
    payload = {
        "patients": [
            {
                "Cl.thickness": 5.0, "Cell.size": 3.0, "Cell.shape": 3.0, "Marg.adhesion": 1.0,
                "Epith.c.size": 2.0, "Bare.nuclei": 1.0, "Bl.cromatin": 3.0, "Normal.nucleoli": 1.0, "Mitoses": 1.0
            },
            {
                "Cl.thickness": 8.0, "Cell.size": 8.0, "Cell.shape": 8.0, "Marg.adhesion": 5.0,
                "Epith.c.size": 7.0, "Bare.nuclei": 10.0, "Bl.cromatin": 9.0, "Normal.nucleoli": 7.0, "Mitoses": 1.0
            }
        ]
    }
    
    start = time.time()
    response = client.post("/predict/batch", json=payload)
    latency = (time.time() - start) * 1000
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    print(f"[SUCCESS] /predict/batch passed. Latency: {latency:.2f}ms")

def test_explain_global(client):
    print("Testing GET /explain/global...")
    start = time.time()
    response = client.get("/explain/global?model_name=random_forest")
    latency = (time.time() - start) * 1000
    
    assert response.status_code == 200
    data = response.json()
    assert "Bare.nuclei" in data or "Cell.size" in data
    print(f"[SUCCESS] /explain/global passed. Latency: {latency:.2f}ms")

def test_rag_retrieve(client):
    print("Testing POST /rag/retrieve...")
    payload = {
        "query": "biomedical neural networks in cancer screening",
        "k": 3
    }
    
    start = time.time()
    response = client.post("/rag/retrieve", json=payload)
    latency = (time.time() - start) * 1000
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3
    print(f"[SUCCESS] /rag/retrieve passed. Latency: {latency:.2f}ms")

if __name__ == "__main__":
    print("=== STARTING API VERIFICATION TESTS ===")
    try:
        # Use TestClient context manager to trigger FastAPI startup event
        with TestClient(app) as client:
            test_health(client)
            test_predict(client)
            test_predict_batch(client)
            test_explain_global(client)
            test_rag_retrieve(client)
        print("ALL TESTS PASSED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"TEST VERIFICATION FAILED: {e}")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
