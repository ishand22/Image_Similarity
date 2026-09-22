import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from tensorflow.keras.models import load_model

from generate_triplets import FastImageCache, augment_image
from model import L2NormalizationLayer

def generate_eval_pairs(split_paths, base_dir=".", num_pairs_per_type=200):
    """
    Constructs comprehensive positive and negative pairs for validation or test sets:
    Positive Pairs:
      - Original vs Zoomed
      - Original vs Cropped
      - Original vs Translated
      - Original vs Resized
      - Original vs Mild Brightness/Contrast variation
    Negative Pairs:
      - Random Different Posters
      - Different Posters with similar composition/colors
    """
    cache = FastImageCache(split_paths, base_dir=base_dir)
    
    pos_pairs = []
    neg_pairs = []

    print(f"Generating evaluation test set pairs from {len(split_paths)} images...", flush=True)
    
    # 1. Generate Positive Pairs (Original vs Augmented Transformation)
    for p_rel in split_paths:
        img_orig = cache.get_image(p_rel)
        img_aug = augment_image(img_orig)
        pos_pairs.append((img_orig, img_aug))

    # 2. Generate Negative Pairs (Different Posters)
    n_images = len(split_paths)
    for idx in range(n_images):
        img_a = cache.get_image(split_paths[idx])
        # Pick different poster
        rand_idx = (idx + np.random.randint(1, n_images)) % n_images
        img_b = cache.get_image(split_paths[rand_idx])
        neg_pairs.append((img_a, img_b))

    return pos_pairs, neg_pairs

def compute_cosine_similarity(emb1, emb2):
    """Computes cosine similarity between two sets of L2-normalized embeddings."""
    return np.sum(emb1 * emb2, axis=-1)

def compute_euclidean_distance(emb1, emb2):
    """Computes Euclidean distance between two sets of L2-normalized embeddings."""
    return np.sqrt(np.maximum(np.sum(np.square(emb1 - emb2), axis=-1), 0.0))

def find_optimal_threshold(val_sims, val_labels):
    """
    Sweeps candidate cosine similarity thresholds on validation set
    to find threshold that maximizes F1 score.
    """
    best_thresh = 0.5
    best_f1 = -1.0
    best_metrics = {}

    thresholds = np.linspace(-1.0, 1.0, 401)
    for thresh in thresholds:
        preds = (val_sims >= thresh).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            prec = precision_score(val_labels, preds, zero_division=0)
            rec = recall_score(val_labels, preds, zero_division=0)
            best_metrics = {
                "threshold": float(best_thresh),
                "validation_f1": float(best_f1),
                "validation_precision": float(prec),
                "validation_recall": float(rec)
            }

    print(f"\n--- Validation Threshold Search Result ---", flush=True)
    print(f"  Optimal Cosine Similarity Threshold: {best_thresh:.4f}", flush=True)
    print(f"  Validation F1-Score:                {best_f1:.4f}", flush=True)
    return best_thresh, best_metrics

def evaluate_model(
    encoder_path="visual_similarity.h5",
    data_dir="data",
    save_thresh_path="data/val_threshold.json"
):
    print("--- Step: Evaluating Siamese Triplet Encoder ---", flush=True)

    # 1. Load encoder model
    encoder = load_model(encoder_path, custom_objects={"L2NormalizationLayer": L2NormalizationLayer})

    # 2. Load Validation and Test Splits
    with open(os.path.join(data_dir, "val_split.json"), "r") as f:
        val_paths = json.load(f)
    with open(os.path.join(data_dir, "test_split.json"), "r") as f:
        test_paths = json.load(f)

    # -------------------------------------------------------------
    # A. Validation Evaluation & Threshold Determination
    # -------------------------------------------------------------
    print("\n--- Phase 1: Validation Set Threshold Selection ---", flush=True)
    val_pos, val_neg = generate_eval_pairs(val_paths)

    def extract_pair_embeddings(pairs):
        imgs1 = np.array([p[0].astype(np.float32) / 255.0 for p in pairs], dtype=np.float32)
        imgs2 = np.array([p[1].astype(np.float32) / 255.0 for p in pairs], dtype=np.float32)

        emb1 = encoder.predict(imgs1, batch_size=64, verbose=0)
        emb2 = encoder.predict(imgs2, batch_size=64, verbose=0)
        return emb1, emb2

    val_pos_e1, val_pos_e2 = extract_pair_embeddings(val_pos)
    val_neg_e1, val_neg_e2 = extract_pair_embeddings(val_neg)

    val_pos_sims = compute_cosine_similarity(val_pos_e1, val_pos_e2)
    val_neg_sims = compute_cosine_similarity(val_neg_e1, val_neg_e2)

    val_sims = np.concatenate([val_pos_sims, val_neg_sims])
    val_labels = np.concatenate([np.ones(len(val_pos_sims)), np.zeros(len(val_neg_sims))])

    optimal_thresh, val_metrics_dict = find_optimal_threshold(val_sims, val_labels)

    # Save threshold
    with open(save_thresh_path, "w") as f:
        json.dump(val_metrics_dict, f, indent=2)
    print(f"Saved optimal threshold metadata to {save_thresh_path}", flush=True)

    # -------------------------------------------------------------
    # B. Test Set Evaluation using Validation Threshold
    # -------------------------------------------------------------
    print("\n--- Phase 2: Held-Out Test Set Evaluation ---", flush=True)
    test_pos, test_neg = generate_eval_pairs(test_paths)

    test_pos_e1, test_pos_e2 = extract_pair_embeddings(test_pos)
    test_neg_e1, test_neg_e2 = extract_pair_embeddings(test_neg)

    test_pos_sims = compute_cosine_similarity(test_pos_e1, test_pos_e2)
    test_neg_sims = compute_cosine_similarity(test_neg_e1, test_neg_e2)

    test_pos_dists = compute_euclidean_distance(test_pos_e1, test_pos_e2)
    test_neg_dists = compute_euclidean_distance(test_neg_e1, test_neg_e2)

    y_true = np.concatenate([np.ones(len(test_pos_sims)), np.zeros(len(test_neg_sims))])
    y_sims = np.concatenate([test_pos_sims, test_neg_sims])
    y_dists = np.concatenate([test_pos_dists, test_neg_dists])

    # Classify based on optimal validation threshold
    y_pred = (y_sims >= optimal_thresh).astype(int)

    # Compute Metrics
    auc = roc_auc_score(y_true, y_sims)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n==================================================")
    print("           HELD-OUT TEST SET METRICS              ")
    print("==================================================")
    print(f"  Similarity Threshold (Learned): {optimal_thresh:.4f}")
    print(f"  ROC-AUC Score:                 {auc:.4f}")
    print(f"  Precision:                     {prec:.4f}")
    print(f"  Recall:                        {rec:.4f}")
    print(f"  F1-Score:                      {f1:.4f}")
    print("\n  Confusion Matrix:")
    print("                 Predicted Negative  Predicted Positive")
    print(f"  Actual Negative        {cm[0, 0]:<17} {cm[0, 1]}")
    print(f"  Actual Positive        {cm[1, 0]:<17} {cm[1, 1]}")
    print("\n  Similarity Distributions:")
    print(f"    Positive Pairs -> Mean Cosine Sim: {np.mean(test_pos_sims):.4f} (Std: {np.std(test_pos_sims):.4f}) | Mean Eucl Dist: {np.mean(test_pos_dists):.4f}")
    print(f"    Negative Pairs -> Mean Cosine Sim: {np.mean(test_neg_sims):.4f} (Std: {np.std(test_neg_sims):.4f}) | Mean Eucl Dist: {np.mean(test_neg_dists):.4f}")
    print("==================================================\n")

    results = {
        "threshold": float(optimal_thresh),
        "roc_auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "positive_pairs_mean_cosine_sim": float(np.mean(test_pos_sims)),
        "negative_pairs_mean_cosine_sim": float(np.mean(test_neg_sims)),
        "positive_pairs_mean_euclidean_dist": float(np.mean(test_pos_dists)),
        "negative_pairs_mean_euclidean_dist": float(np.mean(test_neg_dists))
    }
    
    with open("data/test_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    evaluate_model()
