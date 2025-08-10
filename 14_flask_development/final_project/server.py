'''
The module runs flask instance and contains routing funcs.
'''

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")

@app.route("/")
def get_home():
    '''
    Routing for the index site.
    '''
    return render_template("index.html")

@app.route("/emotionDetector")
def get_emotions():
    '''
    Analyze text acquired from GET method & the webform.
    '''
    text_to_analyze = request.args.get("textToAnalyze", type=str)
    emotions = emotion_detector(text_to_analyze)
    if emotions["dominant_emotion"] is not None:
        return f"""For the given statement, the system response is 'anger': {emotions["anger"]},
        'disgust': {emotions["disgust"]}, 'fear': {emotions["fear"]},
        'joy': {emotions["joy"]}, 'sadness': {emotions["sadness"]}.
        The dominant emotion is <b>{emotions["dominant_emotion"]}</b>."""
    return "<b>Invalid text! Please try again!</b>"

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
