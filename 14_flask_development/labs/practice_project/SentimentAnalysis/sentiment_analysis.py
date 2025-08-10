import requests
import json

def sentiment_analyzer(text_to_analyse):
    url = "https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict"
    headers = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}
    input = { "raw_document": { "text": text_to_analyse } }    

    response = requests.post(url, json=input, headers=headers)
    
    if response.status_code == 200:
        out = json.loads(response.text)
        return {"label":out["documentSentiment"]["label"],
                "score":out["documentSentiment"]["score"]}

    elif response.status_code == 500:
        return {"label":None,
                "score":None}

#if __name__=="__main__":
#    print(sentiment_analyzer("I love this new technology"))