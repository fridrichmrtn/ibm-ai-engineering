''' Executing this function initiates the application of sentiment
    analysis to be executed over the Flask channel and deployed on
    localhost:5000.
'''

from flask import Flask, render_template, request, escape
from SentimentAnalysis.sentiment_analysis import sentiment_analyzer

app = Flask("Sentiment Analyzer Web App")

@app.route("/sentimentAnalyzer")
def sent_analyzer():
    ''' This code receives the text from the HTML interface and 
        runs sentiment analysis over it using sentiment_analysis()
        function. The output returned shows the label and its confidence 
        score for the provided text.
    '''
    text_to_analyze = escape(request.args.get("textToAnalyze", type=str))
    sentiment_results = sentiment_analyzer(text_to_analyze)
    label = sentiment_results["label"].split("_")[1]
    score = sentiment_results["score"]
    return f"The given text has been identified as {label} with a score of {score:3f}."

@app.route("/")
def render_index_page():
    ''' This func renders the home from index template.
    '''
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
