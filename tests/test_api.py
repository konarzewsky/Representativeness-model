from fastapi.testclient import TestClient

from api.env import API_AUTH_TOKEN
from api.main import app

client = TestClient(app)

headers = {
    "Content-Type": "application/json;charset=UTF-8",
    "Auth-Token": API_AUTH_TOKEN,
}


def test_root():
    response = client.get("/", headers=headers)
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to 'Representativeness model' service."
    }


def test_unauthorized_request():
    response = client.get("/")
    assert response.status_code == 400
    assert response.json()["detail"] == "Auth-Token header not provided"


def test_invalid_auth_token():
    response = client.get("/", headers={"Auth-Token": "Invalid-token"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid Auth-Token"


def test_initial_status():
    response = client.get("/status", headers=headers)
    assert response.status_code == 200
    assert response.json().get("details") == "No training recorded so far"


def test_pred_no_model():
    response = client.post("/predict", json={"data": []}, headers=headers)
    assert response.status_code == 200
    assert response.json().get("details") == "No models trained yet"
