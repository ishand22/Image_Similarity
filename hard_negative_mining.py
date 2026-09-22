import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from generate_triplets import FastImageCache
from model import L2NormalizationLayer

def mine_hard_negatives(
    encoder_path="visual_similarity.h5",
    data_dir="data",
    top_k=5,
    max_samples=2000,
    output_json="data/hard_negatives.json"
):
    print("--- Step: Hard-Negative Mining ---", flush=True)

    print(f"Loading encoder model from {encoder_path}...", flush=True)
    encoder = load_model(encoder_path, custom_objects={"L2NormalizationLayer": L2NormalizationLayer})

    train_split_path = os.path.join(data_dir, "train_split.json")
    with open(train_split_path, "r") as f:
        train_paths = json.load(f)

    # Use up to max_samples for fast, high-quality hard negative mining
    sample_paths = train_paths[:max_samples]
    print(f"Extracting embeddings for {len(sample_paths)} training posters...", flush=True)
    cache = FastImageCache(sample_paths, base_dir=".")

    imgs_arr = np.array([cache.get_image(p).astype(np.float32) / 255.0 for p in sample_paths], dtype=np.float32)
    embeddings = encoder.predict(imgs_arr, batch_size=128, verbose=1)
    print(f"Computed embeddings shape: {embeddings.shape}", flush=True)

    print("Computing pairwise distance matrix...", flush=True)
    dot_products = np.dot(embeddings, embeddings.T)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    distances = np.sqrt(np.maximum(2.0 - 2.0 * dot_products, 0.0))

    np.fill_diagonal(distances, np.inf)

    print(f"Selecting top-{top_k} hard negatives for each anchor...", flush=True)
    hard_negatives_dict = {}
    for idx, rel_path in enumerate(sample_paths):
        hard_indices = np.argsort(distances[idx])[:top_k]
        hard_neg_paths = [sample_paths[h_idx] for h_idx in hard_indices]
        hard_negatives_dict[rel_path] = hard_neg_paths

    with open(output_json, "w") as f:
        json.dump(hard_negatives_dict, f, indent=2)

    print(f"Saved {len(hard_negatives_dict)} hard negative mappings to {output_json} successfully!", flush=True)
    return hard_negatives_dict

if __name__ == "__main__":
    mine_hard_negatives()
