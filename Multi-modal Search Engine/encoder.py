import torch
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def encoding_image(file_path):
    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
    image_features = None
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.numpy()
def encoding_text(text: str):
    text = clip.tokenize(text).to(device)
    text_features = None
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features.numpy()