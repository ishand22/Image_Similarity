import os
import glob
import json
import random
import shutil
import numpy as np
from PIL import Image
import kagglehub

def prepare_dataset(base_dir="."):
    data_dir = os.path.join(base_dir, "data")
    posters_dir = os.path.join(data_dir, "posters")
    os.makedirs(posters_dir, exist_ok=True)

    print("--- Step 1: Downloading & Preparing Movie Poster Dataset ---", flush=True)
    
    image_paths = []
    try:
        path_rc = kagglehub.dataset_download("raman77768/movie-classifier")
        images_found = glob.glob(os.path.join(path_rc, "**", "*.jpg"), recursive=True) + \
                       glob.glob(os.path.join(path_rc, "**", "*.png"), recursive=True)
        print(f"Found {len(images_found)} images in raman77768/movie-classifier", flush=True)
        image_paths.extend(images_found)
    except Exception as e:
        print(f"Could not download raman77768/movie-classifier: {e}", flush=True)

    try:
        path_nz = kagglehub.dataset_download("nazimamzz/imdb-dataset-of-5000-movie-posters")
        nz_images = glob.glob(os.path.join(path_nz, "**", "*.jpg"), recursive=True) + \
                    glob.glob(os.path.join(path_nz, "**", "*.png"), recursive=True)
        if nz_images:
            print(f"Found {len(nz_images)} images in nazimamzz/imdb-dataset-of-5000-movie-posters", flush=True)
            image_paths.extend(nz_images)
    except Exception as e:
        print(f"Check nazimamzz dataset: {e}", flush=True)

    if not image_paths:
        raise RuntimeError("No poster images found! Please check dataset downloads.")

    print(f"Total raw images collected: {len(image_paths)}", flush=True)

    valid_images = []
    print("Validating image integrity and copying...", flush=True)
    for idx, img_path in enumerate(image_paths):
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            continue
        target_name = f"poster_{idx:05d}{ext}"
        target_path = os.path.join(posters_dir, target_name)
        if not os.path.exists(target_path):
            try:
                shutil.copyfile(img_path, target_path)
            except Exception:
                continue
        valid_images.append(os.path.relpath(target_path, base_dir))
        if (idx + 1) % 2000 == 0:
            print(f"Processed {idx + 1} / {len(image_paths)} images...", flush=True)

    print(f"Successfully processed {len(valid_images)} poster images.", flush=True)

    valid_images = sorted(valid_images)
    random.seed(42)
    random.shuffle(valid_images)

    n_total = len(valid_images)
    n_train = int(0.80 * n_total)
    n_val = int(0.10 * n_total)

    train_split = valid_images[:n_train]
    val_split = valid_images[n_train:n_train + n_val]
    test_split = valid_images[n_train + n_val:]

    print(f"\nDataset Split (Original Images):", flush=True)
    print(f"  Train:      {len(train_split)} ({len(train_split)/n_total*100:.1f}%)", flush=True)
    print(f"  Validation: {len(val_split)} ({len(val_split)/n_total*100:.1f}%)", flush=True)
    print(f"  Test:       {len(test_split)} ({len(test_split)/n_total*100:.1f}%)", flush=True)

    with open(os.path.join(data_dir, "train_split.json"), "w") as f:
        json.dump(train_split, f, indent=2)
    with open(os.path.join(data_dir, "val_split.json"), "w") as f:
        json.dump(val_split, f, indent=2)
    with open(os.path.join(data_dir, "test_split.json"), "w") as f:
        json.dump(test_split, f, indent=2)

    print("Saved dataset split metadata successfully!", flush=True)
    return train_split, val_split, test_split

if __name__ == "__main__":
    prepare_dataset()
