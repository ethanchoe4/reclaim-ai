from flask import Flask, render_template, request, url_for
from SI_CLIP import ImageSimilaritySearch
from txt_to_image_results import ImageSearch
import os
from shutil import copyfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
txt_search_engine = ImageSearch()
image_search_engine = ImageSimilaritySearch()

UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    result_images = []

    if request.method == 'POST':
        query = request.form.get('description', '').strip()
        uploaded_file = request.files.get('image')

        matches = []

        # === Option 1: Use text description
        if query:
            matches = txt_search_engine.search(query)

        # === Option 2: Use uploaded image
        elif uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename)
            saved_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(saved_path)

            matches = image_search_engine.search(saved_path)

        # Save results to static/results/ folder
        result_images = []
        for i, match in enumerate(matches):
            dst = os.path.join(RESULTS_FOLDER, f"match_{i}.jpg")
            copyfile(match["path"], dst)
            result_images.append({
                "filename": f"results/match_{i}.jpg",
                "location": match.get("location", "MSC Suite 2240")  # Default fallback
            })

    return render_template('index.html', title="Reclaim AI", result_images=result_images)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)