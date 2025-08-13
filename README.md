# Reclaim AI — Lost & Found Visual Search

Reclaim AI is a small Flask web app that helps users locate lost items by searching a gallery of known items using either:
- Text-to-image search (describe your item in text), or
- Image-to-image search (upload a photo of your item)

Under the hood it uses OpenAI CLIP (ViT-B/32) to embed images and text, cosine similarity for text→image search, and a FAISS index for fast image→image search.

## Features
- Web UI to submit a short description or upload an image
- Top-5 visual matches with thumbnail previews
- Two search modes:
  - Text → Image: CLIP text embedding vs. precomputed image embeddings
  - Image → Image: CLIP image embedding + FAISS nearest neighbors
- Assets and results hosted under `static/`

## Repository layout (key files)
- `app.py` — Flask server and routes; wires the two search engines to the UI.
- `templates/index.html` — UI template with upload and description form, results grid, and campus info.
- `static/` — Styles, images, and runtime result thumbnails (`static/results/`).
- `LAF_items/` — Source image catalog for known/lost-and-found items (JPEG/PNG).
- `txt_to_image_results.py` — Embeds all images with CLIP and saves: `image_embeddings.npy`, `image_paths.json` for text→image search.
- `SI_build_faiss.py` — Builds a FAISS index (`items.faiss`) and `image_paths.pkl` for image→image search.
- `SI_CLIP.py` — ImageSimilaritySearch wrapper that loads CLIP, the FAISS index, and returns top matches.
- `img_to_img.py` — Optional/experimental S3-based template matching example (not required for the app).

Precomputed artifacts (optional, can be regenerated):
- `image_embeddings.npy`, `image_paths.json` — for text→image search
- `items.faiss`, `image_paths.pkl` — for image→image search

## Requirements
- Python 3.10+ (3.11 recommended)
- Internet access on first run (to download CLIP weights from Hugging Face)

Install Python packages:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Prepare the image catalog
Place your known items in `LAF_items/` as `.jpg`, `.jpeg`, or `.png`. Filenames are used only for display; paths are persisted when embeddings/index are created.

## Build embeddings and the FAISS index
You can run the app directly; missing artifacts will be generated lazily by the respective components. To precompute explicitly:

- Text→Image (embeddings file + paths):
```bash
python -c "from txt_to_image_results import ImageSearch; ImageSearch()"
```
- Image→Image (FAISS index + paths):
```bash
python SI_build_faiss.py
```

This produces the four artifacts listed above alongside the scripts.

## Run the app
```bash
python app.py
```
Then open http://localhost:5001 in a browser.

Usage:
- Enter a brief description (e.g., "black umbrella with wooden handle") to run text→image search, or
- Upload a photo of your item to run image→image search.
The top 5 matches appear as thumbnails; copies are written to `static/results/` for display.

## Optional: S3 template matching
`img_to_img.py` contains an S3-based example using OpenCV template matching. It is not used by the Flask app. To experiment, update `BUCKET_NAME` and provide a prefix of images in your bucket.

## Notes & troubleshooting
- First run will download CLIP model weights; allow a moment for initialization.
- If FAISS or embedding files are missing, the app will create them (or you can precompute as shown above).
- Ensure `LAF_items/` contains images before building embeddings or running searches; empty folders will raise errors.

## Attribution
- CLIP model and processor: OpenAI CLIP via Hugging Face Transformers.
- Nearest neighbor search: FAISS.
