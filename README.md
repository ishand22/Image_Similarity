# Visual Similarity Siamese Triplet Network for Movie Posters

A complete TensorFlow/Keras Siamese Triplet Network for learning **visual/appearance similarity** (not semantic similarity) of movie posters using a custom shared 256-dimensional CNN encoder.

## Features
- **Visual Appearance Focus**: Trained to be invariant to visual transformations (zoom, crop, translation, resizing, lighting changes) while keeping visually distinct posters separate.
- **Custom CNN Encoder**: Built strictly from scratch using TensorFlow/Keras matching exact architectural specifications. No Hugging Face, CLIP, or pre-trained semantic backbones.
- **Data Leakage Prevention**: Original posters are split (80% Train, 10% Validation, 10% Test) *before* triplet generation.
- **Hard-Negative Mining**: Multi-stage training pipeline leveraging offline nearest-neighbor mining to find and fine-tune on visually hard negatives.
- **Dynamic Thresholding**: Optimal similarity threshold computed on validation set via F1-score optimization.
- **Standalone Export**: Trained encoder saved as `visual_similarity.h5` for fast 256-d embedding inference.

---

## Directory Structure

```
project/
├── visual_similarity.h5      # Exported trained shared CNN encoder
├── download_and_prep.py     # Dataset download and 80/10/10 split
├── generate_triplets.py     # Triplet generation and appearance augmentations
├── model.py                 # Shared CNN encoder architecture & Triplet Loss
├── train.py                 # Siamese network training & fine-tuning script
├── hard_negative_mining.py  # Offline nearest-neighbor hard negative miner
├── evaluate.py              # Validation threshold selection & test evaluation
├── inference.py             # CLI inference script
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---

## Shared CNN Encoder Architecture

- **Input Shape**: `(224, 224, 3)`
- **Layers**:
  - `Conv2D(32, 3x3, ReLU) -> Conv2D(32, 3x3, ReLU) -> MaxPooling2D(2x2)`
  - `Conv2D(64, 3x3, ReLU) -> Conv2D(64, 3x3, ReLU) -> MaxPooling2D(2x2)`
  - `Conv2D(128, 3x3, ReLU) -> Conv2D(128, 3x3, ReLU) -> MaxPooling2D(2x2)`
  - `Conv2D(256, 3x3, ReLU) -> GlobalAveragePooling2D()`
  - `Dense(256) -> L2 Normalization`
- **Output Embedding**: 256-dimensional L2-normalized vector $\|E\|_2 = 1.0$.

---

## Installation & Setup

```bash
pip install -r requirements.txt
```

---

## Complete Pipeline Execution

### 1. Dataset Preparation & Split
Download movie posters and create 80% train / 10% validation / 10% test split:
```bash
python download_and_prep.py
```

### 2. Initial Siamese Triplet Training
Train the initial model using custom Triplet Loss ($L = \max(d(A, P) - d(A, N) + \text{margin}, 0)$):
```bash
python train.py --epochs 8 --steps 60 --batch_size 32 --margin 0.3 --lr 0.0005
```

### 3. Hard-Negative Mining
Extract 256-d embeddings for training posters, find top-5 nearest negative neighbors, and build hard negative dataset:
```bash
python hard_negative_mining.py
```

### 4. Fine-Tuning on Hard Triplets
Fine-tune model using hard negatives:
```bash
python train.py --epochs 6 --steps 50 --batch_size 32 --margin 0.35 --lr 0.0002 --hard_negatives data/hard_negatives.json
```

### 5. Evaluation & Dynamic Threshold Selection
Select optimal similarity threshold on validation set and evaluate test set performance:
```bash
python evaluate.py
```

---

## Inference Usage

To compare two movie poster images for visual similarity:

```bash
python inference.py poster1.jpg poster2.jpg
```

**Example Output:**
```
Similarity: 0.873 Prediction: SIMILAR
```
