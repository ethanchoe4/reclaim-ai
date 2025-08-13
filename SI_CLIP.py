import os
import torch
import faiss
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

from SI_build_faiss import build_faiss_index, get_clip_embedding

class ImageSimilaritySearch:
    def __init__(self, image_dir='LAF_items', index_file='items.faiss', path_file='image_paths.pkl', top_k=5):
        self.image_dir = image_dir
        self.index_file = index_file
        self.path_file = path_file
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

        # Build index if not found
        if not os.path.exists(self.index_file) or not os.path.exists(self.path_file):
            print("Building FAISS index...")
            build_faiss_index(self.image_dir)

        # Load index and image paths
        self.index = faiss.read_index(self.index_file)
        with open(self.path_file, "rb") as f:
            self.image_paths = pickle.load(f)

    def search(self, query_img_path):
        # Compute embedding for query image
        query_vec = get_clip_embedding(query_img_path)
        query_vec = query_vec / np.linalg.norm(query_vec)  # normalize
        query_vec = query_vec.astype("float32").reshape(1, -1)

        assert query_vec.shape[1] == self.index.d, f"Query dim {query_vec.shape[1]} != index dim {self.index.d}"

        scores, indices = self.index.search(query_vec, self.top_k)
        # top_matches = [self.image_paths[i] for i in indices[0]]

        # return top_matches
        top_matches = []
        for idx in indices[0]:
            top_matches.append({
                "path": self.image_paths[idx],
                "score": float(scores[0][list(indices[0]).index(idx)])
            })
        return top_matches

