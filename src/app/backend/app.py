from flask import Flask
from flask_cors import CORS

from routes.predict import predict_bp


def create_app() -> Flask:
    app = Flask(__name__)

    # allow frontend to call backend for local dev
    CORS(app)

    # Register route blueprints
    app.register_blueprint(predict_bp)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)