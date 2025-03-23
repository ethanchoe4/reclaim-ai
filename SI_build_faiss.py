
# watch_similarity_dino_standalone.py

import os
import torch
import timm
import faiss
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
from transformers import CLIPProcessor, CLIPModel

IMAGE_DIR = 'LAF_items'
INDEX_FILE = "items.faiss"
PATH_FILE = "image_paths.pkl"
SEARCH_DIR = 'search_items'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Load CLIP ===
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# # === Load DINOv2 ===
# model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
# model.eval()

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

def get_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]  # Shape: (512,)


# def get_dino_embedding(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         embedding = model(image_tensor)
#         normed = embedding / embedding.norm(p=2, dim=1, keepdim=True)
#     return normed.squeeze(0).numpy()





# # === Build or Load FAISS Index ===
# def build_faiss_index():
#     print("Building CLIP FAISS index...")
#     image_paths = []
#     embeddings = []

#     for fname in tqdm(os.listdir(IMAGE_DIR)):
#         if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
#             path = os.path.join(IMAGE_DIR, fname)
#             try:
#                 emb = get_clip_embedding(path)
#                 embeddings.append(emb)
#                 image_paths.append(path)
#             except Exception as e:
#                 print(f"⚠️ Skipping {fname}: {e}")

#     embeddings = np.stack(embeddings).astype("float32")
#     embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Redundant but safe

#     index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine
#     index.add(embeddings)

#     faiss.write_index(index, INDEX_FILE)
#     with open(PATH_FILE, "wb") as f:
#         pickle.dump(image_paths, f)
#     print("✅ Index built and saved.")





# === Build FAISS Index for Images ===
# def build_faiss_index(image_folder):
    # embeddings = []
    # paths = []
    # for filename in tqdm(os.listdir(image_folder)):
    #     path = os.path.join(image_folder, filename)
    #     if path.lower().endswith(('.jpg', '.jpeg', '.png')):
    #         emb = get_dino_embedding(path)
    #         embeddings.append(emb)
    #         paths.append(path)

    # vectors = np.vstack(embeddings).astype("float32")
    # index = faiss.IndexFlatIP(vectors.shape[1])
    # index.add(vectors)

    # faiss.write_index(index, INDEX_FILE)
    # with open(PATH_FILE, "wb") as f:
    #     pickle.dump(paths, f)

    # print("✅ FAISS index built and saved.")
def build_faiss_index():
    print("🔧 Building CLIP FAISS index...")
    image_paths = []
    embeddings = []

    for fname in tqdm(os.listdir(IMAGE_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(IMAGE_DIR, fname)
            try:
                emb = get_clip_embedding(path)
                if emb is not None and isinstance(emb, np.ndarray):
                    if emb.ndim == 1:
                        embeddings.append(emb)
                        image_paths.append(path)
                    else:
                        print(f"⚠️ Skipped {fname}: embedding shape is {emb.shape}, expected 1D vector")
                else:
                    print(f"⚠️ Skipped {fname}: invalid or None embedding")
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")

    if not embeddings:
        raise ValueError("❌ No valid embeddings were extracted. Check get_clip_embedding and your image files.")

    embeddings = np.stack(embeddings).astype("float32")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine sim with normalized vectors
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(PATH_FILE, "wb") as f:
        pickle.dump(image_paths, f)

    print(f"✅ FAISS index built and saved. Total indexed: {len(image_paths)} images.")


if __name__ == "__main__":
    # Step 1: Index all images in the folder
    build_faiss_index()

