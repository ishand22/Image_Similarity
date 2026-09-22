"""
Movie Poster Visual Similarity - Standalone Single File

Usage from Command Line:
    python movie_poster_similarity.py poster1.jpg poster2.jpg

Usage from Python:
    from movie_poster_similarity import compare_posters, get_image_embedding

    sim, prediction = compare_posters("poster1.jpg", "poster2.jpg")
    print(f"Similarity: {sim:.3f} Prediction: {prediction}")
"""

import sys
import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Custom Architecture & Layers
# -----------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="Custom")
class L2NormalizationLayer(tf.keras.layers.Layer):
    """Custom Keras layer performing L2 normalization on embedding vectors."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


def build_encoder(input_shape=(224, 224, 3), embedding_dim=256):
    """
    Builds the shared CNN encoder:
    Input: 224 x 224 x 3
    Conv2D(32, 3x3) ReLU -> Conv2D(32, 3x3) ReLU -> MaxPooling2D
    Conv2D(64, 3x3) ReLU -> Conv2D(64, 3x3) ReLU -> MaxPooling2D
    Conv2D(128, 3x3) ReLU -> Conv2D(128, 3x3) ReLU -> MaxPooling2D
    Conv2D(256, 3x3) ReLU -> GlobalAveragePooling2D -> Dense(256) -> L2 Normalize
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="image_input")

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense Embedding + L2 Normalization
    x = tf.keras.layers.Dense(embedding_dim, name="dense_embedding")(x)
    outputs = L2NormalizationLayer(name="l2_norm_output")(x)

    encoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="shared_cnn_encoder")
    return encoder

# -----------------------------------------------------------------------------
# 2. Image Preprocessing & Embedding
# -----------------------------------------------------------------------------

def preprocess_image(img_path, target_size=(224, 224)):
    """Loads an image, converts to RGB, resizes to (224, 224), and scales pixels to [0, 1]."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path not found: {img_path}")
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize(target_size, Image.BILINEAR)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            return np.expand_dims(img_arr, axis=0)  # Shape: (1, 224, 224, 3)
    except Exception as e:
        raise ValueError(f"Error loading image {img_path}: {e}")


def load_similarity_model(model_path="visual_similarity.h5"):
    """Loads the trained CNN encoder model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file '{model_path}' not found! Please train or place visual_similarity.h5 in the working directory.")
    
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"L2NormalizationLayer": L2NormalizationLayer}
    )


def get_image_embedding(img_path, model=None, model_path="visual_similarity.h5"):
    """Computes a 256-dimensional L2-normalized embedding for an image."""
    if model is None:
        model = load_similarity_model(model_path)
    
    img_tensor = preprocess_image(img_path)
    embedding = model.predict(img_tensor, verbose=0)
    return embedding[0]  # 256-d vector


def compare_posters(img_path1, img_path2, model=None, model_path="visual_similarity.h5", threshold=0.945):
    """
    Compares two movie poster images for visual appearance similarity.
    Returns:
        similarity (float): Cosine similarity score [-1.0, 1.0]
        prediction (str): "SIMILAR" or "DIFFERENT"
    """
    if model is None:
        model = load_similarity_model(model_path)
    
    emb1 = get_image_embedding(img_path1, model=model)
    emb2 = get_image_embedding(img_path2, model=model)

    # Cosine similarity (dot product of L2-normalized vectors)
    cos_sim = float(np.sum(emb1 * emb2))
    prediction = "SIMILAR" if cos_sim >= threshold else "DIFFERENT"
    
    return cos_sim, prediction

# -----------------------------------------------------------------------------
# 3. CLI Execution Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Movie Poster Visual Similarity Checker")
    parser.add_argument("poster1", type=str, help="Path to first poster image")
    parser.add_argument("poster2", type=str, help="Path to second poster image")
    parser.add_argument("--model", type=str, default="visual_similarity.h5", help="Path to visual_similarity.h5 model file")
    parser.add_argument("--threshold", type=float, default=0.945, help="Similarity threshold (default: 0.945)")

    args = parser.parse_args()

    try:
        cos_sim, prediction = compare_posters(
            args.poster1,
            args.poster2,
            model_path=args.model,
            threshold=args.threshold
        )
        print(f"Similarity: {cos_sim:.3f} Prediction: {prediction}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
