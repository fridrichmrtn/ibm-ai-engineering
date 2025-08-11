"""
Image Captioning App
"""

import gradio as gr
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

# preload model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


def caption_image(image: np.ndarray):
    """
    Generate a caption for the given image.
    """
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Textbox(),
    title="Image Captioning App",
    description="Upload an image to generate a caption using BLIP model.",
)

iface.launch()
