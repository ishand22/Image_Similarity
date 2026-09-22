import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from model import L2NormalizationLayer


# ============================================================
# CHANGE THESE TWO IMAGE PATHS
# ============================================================

IMAGE_PATH_1 = r"C:\Users\ishan\OneDrive\Pictures\Screenshots\Screenshot 2026-09-22 214406.png"
IMAGE_PATH_2 = r"C:\Users\ishan\OneDrive\Pictures\Screenshots\Screenshot 2026-09-22 213653.png"

MODEL_PATH = r"visual_similarity.h5"
THRESHOLD_PATH = r"data/val_threshold.json"


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(img_path, target_size=(224, 224)):

    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Image path not found: {img_path}"
        )

    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)

    img_arr = np.array(
        img,
        dtype=np.float32
    ) / 255.0

    return np.expand_dims(img_arr, axis=0)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    # --------------------------------------------------------
    # Load trained encoder
    # --------------------------------------------------------

    print("Loading model...")

    encoder = load_model(
        MODEL_PATH,
        custom_objects={
            "L2NormalizationLayer": L2NormalizationLayer
        }
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Load learned threshold
    # --------------------------------------------------------

    if not os.path.exists(THRESHOLD_PATH):
        print(f"ERROR: Threshold file not found: {THRESHOLD_PATH}")
        return

    with open(THRESHOLD_PATH, "r") as f:
        data = json.load(f)

    threshold = data["threshold"]

    print(f"Similarity threshold: {threshold:.4f}")

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    print(f"\nImage 1: {IMAGE_PATH_1}")
    print(f"Image 2: {IMAGE_PATH_2}")

    img1 = preprocess_image(IMAGE_PATH_1)
    img2 = preprocess_image(IMAGE_PATH_2)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    emb1 = encoder.predict(
        img1,
        verbose=0
    )

    emb2 = encoder.predict(
        img2,
        verbose=0
    )

    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    # Since embeddings are L2-normalized,
    # dot product = cosine similarity.

    cos_sim = float(
        np.sum(
            emb1 * emb2,
            axis=-1
        )[0]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = (
        "SIMILAR"
        if cos_sim >= threshold
        else "DIFFERENT"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print(f"Cosine Similarity : {cos_sim:.4f}")
    print(f"Threshold         : {threshold:.4f}")
    print(f"Prediction        : {prediction}")
    print("=" * 50)


if __name__ == "__main__":
    main()