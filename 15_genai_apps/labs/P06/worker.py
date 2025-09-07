import io
import os
from openai import OpenAI

#load api token
api_token = os.getenv("OPENAI_API_KEY")
if api_token is None:
    with open("openai_token","r") as file:
        api_token = file.read().strip()

client = OpenAI(api_key=api_token)


def speech_to_text(audio_binary):

    buffer = io.BytesIO(audio_binary)
    buffer.name = "audio.mp3"
    buffer.seek(0)

    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=buffer,
    response_format="text",
    language="en"
    )
    print(transcription)
    return transcription

def text_to_speech(text, voice=""):
    if voice == "default" or voice == "":
        voice = "alloy"

    synthesis = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )

    return synthesis.content

def process_message(user_message):

    prompt = f"""
        Translate the following English sentence into Spanish. 
        Reply ONLY with the translation, no explanations, no formatting, no extra text.
        English:
        {user_message}

        Spanish:
        """
 
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content
