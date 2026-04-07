import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

import src.api as api_module
from src.api import app


VALID_PATIENT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 1, "ca": 0, "thal": 6,
}


@pytest.fixture(scope="module")
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.full((100, 1), 0.72)

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 13))

    with patch.object(api_module, "model", mock_model), \
         patch.object(api_module, "scaler", mock_scaler):
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        res = client.post("/predict", json=VALID_PATIENT)
        assert res.status_code == 200

    def test_predict_response_structure(self, client):
        data = client.post("/predict", json=VALID_PATIENT).json()
        assert "risk_probability" in data
        assert "risk_percentage" in data
        assert "uncertainty_std" in data
        assert "requires_review" in data
        assert "prediction" in data
        assert "mc_samples" in data

    def test_risk_probability_in_range(self, client):
        data = client.post("/predict", json=VALID_PATIENT).json()
        assert 0.0 <= data["risk_probability"] <= 1.0

    def test_uncertainty_is_non_negative(self, client):
        data = client.post("/predict", json=VALID_PATIENT).json()
        assert data["uncertainty_std"] >= 0

    def test_prediction_label_is_valid(self, client):
        data = client.post("/predict", json=VALID_PATIENT).json()
        assert data["prediction"] in ["Healthy", "Heart Disease Risk"]


class TestInputValidation:
    def test_missing_field_returns_422(self, client):
        incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
        res = client.post("/predict", json=incomplete)
        assert res.status_code == 422

    def test_age_too_high_returns_422(self, client):
        bad = {**VALID_PATIENT, "age": 999}
        res = client.post("/predict", json=bad)
        assert res.status_code == 422

    def test_negative_cholesterol_returns_422(self, client):
        bad = {**VALID_PATIENT, "chol": -10}
        res = client.post("/predict", json=bad)
        assert res.status_code == 422

    def test_empty_body_returns_422(self, client):
        res = client.post("/predict", json={})
        assert res.status_code == 422
