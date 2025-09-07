import base64
import json
from flask import Flask, render_template, request
from flask_cors import CORS
import os
from worker import speech_to_text, text_to_speech, process_message

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print("processing Speech-to-Text")
    audio_binary = request.data
    text = speech_to_text(audio_binary)

    response = app.response_class(
        response=json.dumps({"text": text}),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json["userMessage"]
    response_text = process_message(user_message)
    
    voice = request.json["voice"]
    audio_content = text_to_speech(response_text, voice)

    if audio_content is not None:
        response_speech = base64.b64encode(audio_content).decode('utf-8')
    else:
        response_speech = None

    response = app.response_class(
        response=json.dumps({"watsonxResponseText": response_text, "watsonxResponseSpeech": response_speech}),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')
