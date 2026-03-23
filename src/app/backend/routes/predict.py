from flask import Blueprint, jsonify

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200