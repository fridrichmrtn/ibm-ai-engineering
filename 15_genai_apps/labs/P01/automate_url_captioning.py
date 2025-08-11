"""
Captioning images from a webpage
"""

from io import BytesIO
import requests
from bs4 import BeautifulSoup
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# preload model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b"
)

# URL of the page to scrape
URL = "https://en.wikipedia.org/wiki/IBM"

# Download the page
response = requests.get(URL, timeout=100)

# Parse the page with BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Iterate over each img elements
with open("captions.txt", "w", encoding="utf-8") as caption_file:

    for img_element in soup.find_all("img"):
        URL = img_element.get("src")
        if URL.startswith("//"):
            URL = "https:" + URL
        if not URL.startswith("https://") and not URL.startswith("http://"):
            continue
        if "svg" in URL or "1x1" in URL:
            continue

        try:
            response = requests.get(URL, timeout=100)
            raw_image = Image.open(BytesIO(response.content))
            if (
                raw_image.size[0] * raw_image.size[1] > 400
            ):  # Filter out very small images
                continue

            raw_image = raw_image.convert("RGB")

            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=100)
            caption = processor.decode(out[0], skip_special_tokens=True)
            #print(f"Caption for {URL}: {caption}")
            caption_file.write(f"Caption for {URL}: {caption}\n")
        except Exception as e:
            print(f"Error processing {URL}: {e}")
            continue
