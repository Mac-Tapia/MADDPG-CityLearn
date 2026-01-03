"""
Tests para la API de MADDPG CityLearn
"""
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint():
    """Test health check endpoint"""
    from maddpg_tesis.api.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "maddpg-citylearn"}


def test_metrics_endpoint():
    """Test metrics endpoint"""
    from maddpg_tesis.api.main import app

    client = TestClient(app)
    response = client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "uptime_seconds" in data


def test_predict_endpoint_validation():
    """Test predict endpoint with invalid input"""
    from maddpg_tesis.api.main import app

    client = TestClient(app)

    # Test con payload vacío
    response = client.post("/predict", json={"observations": []})
    assert response.status_code == 400
    assert "No se recibieron observaciones" in response.json()["detail"]


# Test de integración (requiere modelo cargado)
@pytest.mark.skip(reason="Requiere modelo entrenado")
def test_predict_endpoint_with_valid_data():
    """Test predict endpoint with valid observations"""
    from maddpg_tesis.api.main import app

    client = TestClient(app)

    # Crear observaciones de prueba (ajustar dimensiones según tu modelo)
    test_observations = {
        "observations": [
            {"obs": [0.1, 0.2, 0.3, 0.4]},
            {"obs": [0.5, 0.6, 0.7, 0.8]},
        ]
    }

    response = client.post("/predict", json=test_observations)
    assert response.status_code == 200
    data = response.json()
    assert "actions" in data
    assert len(data["actions"]) == 2
