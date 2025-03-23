import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ====== CONFIG ======
IMAGE_DIR = 'LAF_items'
EMBEDDING_FILE = "image_embeddings.npy"
PATHS_FILE = "image_paths.json"

# ====== SETUP MODEL ======
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

class ImageSearch:
    def __init__(self):
        if not os.path.exists(EMBEDDING_FILE) or not os.path.exists(PATHS_FILE):
            print("Embeddings not found, generating...")
            self.image_paths, self.image_embeddings = self._generate_and_save_embeddings()
        else:
            print("Loading precomputed embeddings...")
            self.image_embeddings = np.load(EMBEDDING_FILE)
            with open(PATHS_FILE, "r") as f:
                self.image_paths = json.load(f)

    def _generate_and_save_embeddings(self):
        if not os.path.exists(IMAGE_DIR):
            raise FileNotFoundError(f"The folder '{IMAGE_DIR}' does not exist.")

        image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_paths:
            raise ValueError("No image files found in the folder.")

        images, valid_paths = [], []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping image {path}: {e}")

        inputs = clip_processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            img_features = clip_model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        np.save(EMBEDDING_FILE, img_features.cpu().numpy())
        with open(PATHS_FILE, "w") as f:
            json.dump(valid_paths, f)

        return valid_paths, img_features.cpu().numpy()

    def search(self, description, top_k=5, show=False):
        inputs = clip_processor(text=[description], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = cosine_similarity(text_features.cpu().numpy(), self.image_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            match = {
                "path": self.image_paths[idx],
                "score": float(similarities[idx])
            }
            results.append(match)

            if show:
                img = Image.open(self.image_paths[idx])
                plt.imshow(img)
                plt.title(f"{os.path.basename(self.image_paths[idx])} (Score: {similarities[idx]:.4f})")
                plt.axis('off')
                plt.show()

        return results
