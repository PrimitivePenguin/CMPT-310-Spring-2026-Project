from flask import Blueprint, jsonify, request

from src.app.backend.services.mock_predict import mock_predict
from src.app.backend.services.cnn_predict import cnn_predict
from src.app.backend.services.load_cnn import load_cnn

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@predict_bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    model = load_cnn()
    if model == None:
        return jsonify({"error": "Unable to load model"}), 400

    result = cnn_predict(model, image_file)
    return jsonify(result), 200
