import cv2
import numpy as np
from PIL import Image
import boto3
from io import BytesIO
import matplotlib.pyplot as plt

IMAGE_DIR = 'LAF_items'

# AWS S3 Setup
s3 = boto3.client('s3')
BUCKET_NAME = 'your-s3-bucket-name'  # change to your bucket

# Convert PIL Image or upload input into grayscale OpenCV image
def preprocess_image(img_input):
    if isinstance(img_input, Image.Image):
        img = np.array(img_input.convert('RGB'))
    elif isinstance(img_input, str):  # file path
        img = cv2.imread(img_input)
    else:
        raise ValueError("Unsupported image input format")
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Fetch and preprocess all stored images from S3
def get_s3_images_as_grayscale(prefix='items/'):
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    image_data = []

    for obj in response.get('Contents', []):
        key = obj['Key']
        if not key.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        s3_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        image = Image.open(BytesIO(s3_obj['Body'].read()))
        gray = preprocess_image(image)
        image_data.append({'key': key, 'gray': gray, 'original': image})
    return image_data

# Main Function: Match Uploaded Image to Best S3 Image
def match_image_to_s3(input_image):
    input_gray = preprocess_image(input_image)
    s3_images = get_s3_images_as_grayscale()

    best_score = -np.inf
    best_match = None

    for img_data in s3_images:
        stored_gray = img_data['gray']
        # Resize if necessary to match dimensions
        if stored_gray.shape[0] < input_gray.shape[0] or stored_gray.shape[1] < input_gray.shape[1]:
            continue
        result = cv2.matchTemplate(stored_gray, input_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_match = img_data

    # Display match
    if best_match:
        plt.imshow(best_match['original'])
        plt.title(f"Best Match from S3: {best_match['key']} (Score: {best_score:.4f})")
        plt.axis('off')
        plt.show()

        return {
            "matched_s3_key": best_match['key'],
            "score": float(best_score),
            "url": f"https://{BUCKET_NAME}.s3.amazonaws.com/{best_match['key']}"
        }
    else:
        return {"message": "No match found"}