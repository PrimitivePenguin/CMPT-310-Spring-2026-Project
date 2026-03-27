import io
import pytest

from src.app.backend.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_returns_400_when_image_missing(client):
    response = client.post("/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json() == {"error": "No image file provided"}


def test_predict_returns_400_when_filename_empty(client):
    data = {
        "image": (io.BytesIO(b"fake image data"), "")
    }

    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json() == {"error": "No selected file"}


def test_predict_returns_mock_prediction_with_valid_image(client):
    data = {
        "image": (io.BytesIO(b"fake image data"), "test.jpg")
    }

    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200

    json_data = response.get_json()

    assert "emotion" in json_data
    assert "confidence" in json_data
    assert "source" in json_data

    assert json_data["source"] == "mock"