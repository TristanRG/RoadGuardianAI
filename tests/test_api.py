import os
import json
import pytest
from fastapi.testclient import TestClient
from RoadGuardianAI.api.app import app
from RoadGuardianAI.batch.runner import BatchPredictor
import pandas as pd

client = TestClient(app)

API_KEY = os.getenv("API_KEY", "")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "ok"


def test_predict_missing_body():
    r = client.post("/predict", headers={"X-Api-Key": API_KEY})
    assert r.status_code in (400, 422)


def test_predict_minimal():
    payload = {"items": [{"segment_id": "40.5611_-74.1698"}]}
    r = client.post(
        "/predict",
        headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    assert r.status_code == 200
    j = r.json()
    assert "predictions" in j
    assert isinstance(j["predictions"], list)


def test_predict_unauthorized():
    payload = {"items": [{"segment_id": "40.5611_-74.1698"}]}
    r = client.post("/predict", json=payload)  
    assert r.status_code in (401, 403)


def test_predict_multiple_items():
    payload = {"items": [
        {"segment_id": "40.5611_-74.1698"},
        {"segment_id": "40.7128_-74.0060"},
    ]}
    r = client.post("/predict", headers={"X-Api-Key": API_KEY}, json=payload)
    assert r.status_code == 200
    j = r.json()
    assert len(j["predictions"]) == 2


def test_risk_get():
    r = client.get("/risk", headers={"X-Api-Key": API_KEY})
    assert r.status_code == 200
    j = r.json()
    assert "results" in j


def test_risk_with_params():
    r = client.get("/risk?top_k=5", headers={"X-Api-Key": API_KEY})
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j.get("results"), list)
    assert len(j["results"]) <= 5


def test_risk_pretty_output():
    r = client.get("/risk?pretty=true", headers={"X-Api-Key": API_KEY})
    assert r.status_code == 200
    j = r.json()
    assert "results" in j


def test_batch_predictor_runs(tmp_path):
    out_file = tmp_path / "test_batch.parquet"
    bp = BatchPredictor()
    out_path = bp.run_once(out_path=str(out_file)) 
    assert out_file.exists()
    df = pd.read_parquet(out_path)
    assert not df.empty