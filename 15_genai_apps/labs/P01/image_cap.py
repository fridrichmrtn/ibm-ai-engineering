import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_path = "https://static1.moviewebimages.com/wordpress/wp-content/uploads/article/cG8MiS5yM2gxa1r7a0RYRXTTonMhlV.jpg"

img = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
text = "an image of"

inputs = processor(images=img, text=text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)

caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)
