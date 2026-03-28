import os
import io
import pytest
from unittest.mock import patch

from src.app.backend.app import create_app
from src.config import LABELS, CNN_MODEL_PATH

CNN_MODEL = CNN_MODEL_PATH if os.path.exists(CNN_MODEL_PATH) else os.path.join("tests", "assets", "model.pt")
INVALID_MODEL = os.path.join("tests", "assets", "bad-model.pt")
SAMPLE_IMAGE = os.path.join("tests", "assets", "sample.jpg")

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


@patch('src.app.backend.routes.predict.CNN_MODEL_PATH', "")
def test_load_cnn_returns_400_when_model_missing(client):
    data = {
        "image": (io.BytesIO(b"fake image data"), "test.jpg")
    }

    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json() == {"error": "Unable to load model"}


@patch('src.app.backend.routes.predict.CNN_MODEL_PATH', INVALID_MODEL)
def test_load_cnn_returns_400_when_load_invalid_model(client):
    # Check invalid model exists
    assert os.path.exists(INVALID_MODEL)

    data = {
        "image": (io.BytesIO(b"fake image data"), "test.jpg")
    }

    response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json() == {"error": "Unable to load model"}


@patch('src.app.backend.routes.predict.CNN_MODEL_PATH', CNN_MODEL)
def test_predict_returns_cnn_prediction_with_valid_image(client):
    with open(SAMPLE_IMAGE, 'rb') as img:
        data = {
            "image": (img, "test.jpg")
        }
        response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200

    json_data = response.get_json()

    assert "emotion" in json_data
    assert "confidence" in json_data
    assert "source" in json_data

    assert json_data["emotion"] in LABELS
    assert 0 <= json_data["confidence"] <= 1
    assert json_data["source"] == "CNN"