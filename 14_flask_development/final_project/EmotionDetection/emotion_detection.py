'''
Purpose of the module is to house an emotion detection func.
'''
import json
import requests

def emotion_detector(text_to_analyze):
    '''
    Func returns emotions detected in the input string through Watson AI API.
    '''
    response = requests.post(
        "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
        json={"raw_document":{"text":text_to_analyze}},
        headers={"grpc-metadata-mm-model-id":"emotion_aggregated-workflow_lang_en_stock"},
        timeout=2.5)
    if response.status_code == 200:
        parsed_response = json.loads(response.text)
        response = parsed_response["emotionPredictions"][0]["emotion"]
        max_score = 0
        dominant_emotion = None
        for emotion,score in response.items():
            if score > max_score:
                max_score = score
                dominant_emotion = emotion
        response["dominant_emotion"] = dominant_emotion
        return response
    if response.status_code == 400:
        return {'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None}
    return {'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None}                

if __name__ == "__main__":
    print(emotion_detector("I love this new tech"))
