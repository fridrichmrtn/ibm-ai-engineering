"""Flask server for PDF RAG QA.

Endpoints:
- GET  /                 -> serve index.html
- POST /process-document -> upload a PDF and build the retrieval index
- POST /process-message  -> ask questions against the built index
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import worker_openai

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

# Keep upload dir local to the app root
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Single RAG instance (stateful across requests for simplicity)
rag = worker_openai.RagQA()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index() -> Any:
    """Serve the landing page."""
    return render_template("index.html")


@app.route("/process-message", methods=["POST"])
def process_message_route() -> Any:
    """Run a user question through the RAG chain."""
    payload: Dict[str, Any] | None = request.get_json(silent=True)
    if not payload or "userMessage" not in payload:
        return jsonify({"botResponse": "Missing 'userMessage'."}), 400

    user_message = str(payload["userMessage"])
    app.logger.debug("user_message=%s", user_message)

    try:
        bot_response = rag.ask(user_message)
    except RuntimeError as exc:
        # Likely index not built yet
        return jsonify({"botResponse": f"Error: {exc}"}), 400
    except Exception as exc:  # pylint: disable=broad-except
        app.logger.exception("Unexpected error in /process-message: %s", exc)
        return jsonify({"botResponse": "Unexpected server error."}), 500

    return jsonify({"botResponse": bot_response}), 200


@app.route("/process-document", methods=["POST"])
def process_document_route() -> Any:
    """Upload a PDF and build the retrieval index."""
    if "file" not in request.files:
        msg = (
            "File upload missing. Please attach a PDF and try again. "
            "If it persists, try a different file."
        )
        return jsonify({"botResponse": msg}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename or "document.pdf")
    if not filename.lower().endswith(".pdf"):
        return jsonify({"botResponse": "Only PDF files are supported."}), 400

    file_path = UPLOAD_DIR / filename
    try:
        file.save(file_path)
        rag.build_index_from_pdf(str(file_path))
    except Exception as exc:  # pylint: disable=broad-except
        app.logger.exception("Failed to process document: %s", exc)
        return jsonify({"botResponse": "Failed to process the PDF."}), 500

    msg = (
        "Your PDF has been indexed. You can now ask questions about its contents."
    )
    return jsonify({"botResponse": msg}), 200


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Use production WSGI server in real deployments (e.g., gunicorn/uvicorn).
    app.run(debug=True, port=8000, host="0.0.0.0")
