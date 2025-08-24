"""Flask server providing ASR, TTS, and OpenAI chat endpoints."""

import base64
import json
import os

from flask import Flask, render_template, request
from flask_cors import CORS

from worker import speech_to_text, text_to_speech, openai_process_message

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def index():
    """Serve the index page."""
    return render_template("index.html")


@app.route("/speech-to-text", methods=["POST"])
def speech_to_text_route():
    """Convert posted audio to text using ASR pipeline."""
    audio_binary = request.data
    text = speech_to_text(audio_binary)

    response = app.response_class(
        response=json.dumps({"text": text}),
        status=200,
        mimetype="application/json",
    )
    return response


@app.route("/process-message", methods=["POST"])
def process_prompt_route():
    """Process user message via OpenAI and return text + speech (base64 WAV)."""
    user_message = request.json["userMessage"]
    print(f"User message: {user_message}")

    openai_response_text = openai_process_message(user_message)
    # Collapse multiple blank lines
    openai_response_text = os.linesep.join(
        [s for s in openai_response_text.splitlines() if s.strip()]
    )

    # Generate speech as WAV (bytes), then base64 encode
    openai_response_speech = text_to_speech(openai_response_text, as_wav=True)
    openai_response_speech = base64.b64encode(openai_response_speech).decode("utf-8")

    response = app.response_class(
        response=json.dumps(
            {
                "openaiResponseText": openai_response_text,
                "openaiResponseSpeech": openai_response_speech,
            }
        ),
        status=200,
        mimetype="application/json",
    )
    print(response)
    return response


if __name__ == "__main__":
    app.run(port=8000, host="0.0.0.0")
