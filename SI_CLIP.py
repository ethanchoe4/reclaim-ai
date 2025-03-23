# import os
# import torch
# import faiss
# import pickle
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from torchvision import transforms
# import matplotlib.pyplot as plt
# from transformers import CLIPProcessor, CLIPModel

# from SI_build_faiss import build_faiss_index

# # === CONFIG ===
# IMAGE_DIR = 'LAF_items'
# SEARCH_DIR = 'search_items'
# INDEX_FILE = "items.faiss"
# PATH_FILE = "image_paths.pkl"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TOP_K = 5

# from SI_build_faiss import get_clip_embedding

# # === Load CLIP ===
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# # === Get CLIP Embedding ===


# # === Search Function ===
# def search_similar_images(query_img_path, top_k=TOP_K):
#     if not os.path.exists(INDEX_FILE) or not os.path.exists(PATH_FILE):
#         build_faiss_index(IMAGE_DIR)

#     index = faiss.read_index(INDEX_FILE)
#     with open(PATH_FILE, "rb") as f:
#         image_paths = pickle.load(f)

#     # query_vec = get_clip_embedding(query_img_path).reshape(1, -1).astype("float32")
#     # query_vec = query_vec / np.linalg.norm(query_vec)  # Ensure normalized
#     # GOOD ORDERING: normalize BEFORE reshaping and after .numpy()
#     query_vec = get_clip_embedding(query_img_path)       # shape: (512,)
#     query_vec = query_vec / np.linalg.norm(query_vec)    # normalize 1D vector
#     query_vec = query_vec.astype("float32").reshape(1, -1)  # shape: (1, 512)

#     assert query_vec.shape[1] == index.d, f"Query dim {query_vec.shape[1]} != index dim {index.d}"

#     scores, indices = index.search(query_vec, top_k)

#     top_matches = [image_paths[i] for i in indices[0]]

#     # === Plot Results ===
#     plt.figure(figsize=(15, 3))
#     plt.subplot(1, top_k + 1, 1)
#     plt.imshow(Image.open(query_img_path))
#     plt.title("Query")
#     plt.axis("off")

#     for i, path in enumerate(top_matches):
#         img = Image.open(path)
#         plt.subplot(1, top_k + 1, i + 2)
#         plt.imshow(img)
#         plt.title(f"Match {i+1}")
#         plt.axis("off")

#     plt.tight_layout()
#     plt.show()

# # === MAIN ===
# if __name__ == "__main__":
#     search_similar_images(f'{SEARCH_DIR}/IMG_8101.png')

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

    # def _display_results(self, query_img_path, top_matches):
    #     plt.figure(figsize=(15, 3))
    #     plt.subplot(1, self.top_k + 1, 1)
    #     plt.imshow(Image.open(query_img_path))
    #     plt.title("Query")
    #     plt.axis("off")

    #     for i, path in enumerate(top_matches):
    #         img = Image.open(path)
    #         plt.subplot(1, self.top_k + 1, i + 2)
    #         plt.imshow(img)
    #         plt.title(f"Match {i+1}")
    #         plt.axis("off")

    #     plt.tight_layout()
    #     plt.show()

