import os
import json
import random
import numpy as np
from PIL import Image, ImageEnhance
from concurrent.futures import ThreadPoolExecutor

def augment_image(img_np):
    """
    Applies realistic visual transformations preserving appearance identity:
    - Moderate zoom / crop
    - Small translation
    - Mild brightness & contrast adjustments
    - Very mild rotation
    - Resize back to (224, 224, 3)
    """
    img = Image.fromarray(img_np)
    w, h = img.size

    # 1. Very mild rotation (-5 to 5 degrees)
    if random.random() < 0.5:
        angle = random.uniform(-5.0, 5.0)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # 2. Moderate zoom & crop (crop between 85% and 98% of original dimension)
    crop_factor = random.uniform(0.85, 0.98)
    crop_w = int(w * crop_factor)
    crop_h = int(h * crop_factor)

    # 3. Small translation
    max_dx = max(0, w - crop_w)
    max_dy = max(0, h - crop_h)
    dx = random.randint(0, max_dx) if max_dx > 0 else 0
    dy = random.randint(0, max_dy) if max_dy > 0 else 0

    img = img.crop((dx, dy, dx + crop_w, dy + crop_h))

    # 4. Mild brightness adjustment (0.85 to 1.15)
    if random.random() < 0.6:
        brightness_factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)

    # 5. Mild contrast adjustment (0.85 to 1.15)
    if random.random() < 0.6:
        contrast_factor = random.uniform(0.85, 1.15)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

    # 6. Resize back to (224, 224)
    img = img.resize((224, 224), Image.BILINEAR)
    return np.array(img)

def load_single_image(img_path, target_size=(224, 224)):
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size, Image.BILINEAR)
            return np.array(img, dtype=np.uint8)
    except Exception:
        return np.zeros((*target_size, 3), dtype=np.uint8)

class FastImageCache:
    """Pre-loads and caches 224x224 uint8 images in memory for ultra-fast training."""
    def __init__(self, relative_paths, base_dir=".", max_workers=16):
        self.base_dir = base_dir
        self.rel_paths = list(relative_paths)
        self.cache = {}

        print(f"Pre-loading {len(self.rel_paths)} images into memory cache...", flush=True)
        abs_paths = [os.path.join(base_dir, p) for p in self.rel_paths]

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(load_single_image, abs_paths))

        for rel_p, img_arr in zip(self.rel_paths, results):
            self.cache[rel_p] = img_arr

        print("Finished caching images in RAM!", flush=True)

    def get_image(self, rel_path):
        if rel_path in self.cache:
            return self.cache[rel_path]
        return load_single_image(os.path.join(self.base_dir, rel_path))

class TripletDataGenerator:
    """Fast Triplet Generator using RAM image cache."""
    def __init__(self, relative_paths, base_dir=".", batch_size=32, hard_negatives_dict=None, image_cache=None):
        self.rel_paths = list(relative_paths)
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.hard_negatives_dict = hard_negatives_dict
        
        if image_cache is None:
            self.cache = FastImageCache(self.rel_paths, base_dir=base_dir)
        else:
            self.cache = image_cache

    def generate_batch(self):
        anchors, positives, negatives = [], [], []

        for _ in range(self.batch_size):
            # Select random anchor
            anchor_rel = random.choice(self.rel_paths)
            anchor_uint8 = self.cache.get_image(anchor_rel)

            # Positive: augmented version
            pos_uint8 = augment_image(anchor_uint8)

            # Negative: choose hard negative if present, else random different poster
            neg_rel = None
            if self.hard_negatives_dict and anchor_rel in self.hard_negatives_dict:
                candidates = self.hard_negatives_dict[anchor_rel]
                if candidates:
                    neg_rel = random.choice(candidates)

            if neg_rel is None or neg_rel == anchor_rel:
                while True:
                    neg_rel = random.choice(self.rel_paths)
                    if neg_rel != anchor_rel:
                        break

            neg_uint8 = self.cache.get_image(neg_rel)

            anchors.append(anchor_uint8.astype(np.float32) / 255.0)
            positives.append(pos_uint8.astype(np.float32) / 255.0)
            negatives.append(neg_uint8.astype(np.float32) / 255.0)

        return (
            np.array(anchors, dtype=np.float32),
            np.array(positives, dtype=np.float32),
            np.array(negatives, dtype=np.float32)
        )

if __name__ == "__main__":
    data_dir = "data"
    with open(os.path.join(data_dir, "train_split.json"), "r") as f:
        train_paths = json.load(f)

    gen = TripletDataGenerator(train_paths[:100], batch_size=4)
    a, p, n = gen.generate_batch()
    print("Batch test:", a.shape, p.shape, n.shape)
