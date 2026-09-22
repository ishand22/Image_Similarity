import os
import json
import argparse
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from model import build_encoder
from generate_triplets import TripletDataGenerator, FastImageCache

def train_siamese_network(
    data_dir="data",
    epochs=6,
    steps_per_epoch=40,
    batch_size=32,
    margin=0.3,
    learning_rate=0.0005,
    hard_negatives_file=None,
    save_encoder_path="visual_similarity.h5",
    save_weights_path=None
):
    print(f"--- Starting Training Siamese Triplet Network ---", flush=True)
    print(f"  Epochs:           {epochs}", flush=True)
    print(f"  Steps per Epoch:  {steps_per_epoch}", flush=True)
    print(f"  Batch Size:       {batch_size}", flush=True)
    print(f"  Margin:           {margin}", flush=True)
    print(f"  Learning Rate:    {learning_rate}", flush=True)

    # 1. Load train split paths
    train_split_path = os.path.join(data_dir, "train_split.json")
    with open(train_split_path, "r") as f:
        train_paths = json.load(f)

    # Load hard negatives if provided
    hard_negatives_dict = None
    if hard_negatives_file and os.path.exists(hard_negatives_file):
        print(f"Loading hard negatives from {hard_negatives_file}...", flush=True)
        with open(hard_negatives_file, "r") as f:
            hard_negatives_dict = json.load(f)

    # 2. Setup RAM Image Cache and Data Generator
    cache = FastImageCache(train_paths, base_dir=".")
    gen = TripletDataGenerator(
        train_paths,
        base_dir=".",
        batch_size=batch_size,
        hard_negatives_dict=hard_negatives_dict,
        image_cache=cache
    )

    # 3. Build Shared Encoder Architecture
    encoder = build_encoder(input_shape=(224, 224, 3), embedding_dim=256)
    
    if os.path.exists(save_encoder_path) and hard_negatives_file is not None:
        print(f"Loading existing encoder weights from {save_encoder_path} for fine-tuning...", flush=True)
        try:
            encoder.load_weights(save_encoder_path)
        except Exception as e:
            print(f"Could not load encoder weights directly: {e}", flush=True)

    optimizer = Adam(learning_rate=learning_rate)

    # Explicit input signature to prevent re-tracing on CPU
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
    ])
    def train_step(anchor_b, pos_b, neg_b):
        with tf.GradientTape() as tape:
            emb_a = encoder(anchor_b, training=True)
            emb_p = encoder(pos_b, training=True)
            emb_n = encoder(neg_b, training=True)

            d_pos = tf.sqrt(tf.reduce_sum(tf.square(emb_a - emb_p), axis=-1) + 1e-8)
            d_neg = tf.sqrt(tf.reduce_sum(tf.square(emb_a - emb_n), axis=-1) + 1e-8)

            loss = tf.reduce_mean(tf.maximum(d_pos - d_neg + margin, 0.0))

        grads = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

        return loss, tf.reduce_mean(d_pos), tf.reduce_mean(d_neg)

    print("\nStarting optimization loop...", flush=True)
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        running_loss, running_dp, running_dn = 0.0, 0.0, 0.0

        for step in range(1, steps_per_epoch + 1):
            a_batch, p_batch, n_batch = gen.generate_batch()
            a_t = tf.convert_to_tensor(a_batch, dtype=tf.float32)
            p_t = tf.convert_to_tensor(p_batch, dtype=tf.float32)
            n_t = tf.convert_to_tensor(n_batch, dtype=tf.float32)

            loss_val, d_p, d_n = train_step(a_t, p_t, n_t)

            running_loss += float(loss_val)
            running_dp += float(d_p)
            running_dn += float(d_n)

            if step % 10 == 0 or step == steps_per_epoch:
                print(f"  [Epoch {epoch}/{epochs}] Step {step}/{steps_per_epoch} - Loss: {float(loss_val):.4f} | d_pos: {float(d_p):.4f} | d_neg: {float(d_n):.4f}", flush=True)

        epoch_loss = running_loss / steps_per_epoch
        epoch_dp = running_dp / steps_per_epoch
        epoch_dn = running_dn / steps_per_epoch
        elapsed = time.time() - epoch_start

        print(
            f"--> Epoch {epoch:02d}/{epochs:02d} Summary ({elapsed:.1f}s) - Loss: {epoch_loss:.4f} | "
            f"Avg Pos Dist: {epoch_dp:.4f} | Avg Neg Dist: {epoch_dn:.4f}\n",
            flush=True
        )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds!", flush=True)

    # Save Exported Encoder Model
    encoder.save(save_encoder_path)
    print(f"Saved trained CNN encoder to {save_encoder_path} successfully!", flush=True)

    if save_weights_path:
        encoder.save_weights(save_weights_path)
        print(f"Saved weights to {save_weights_path}", flush=True)

    return encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Triplet Network")
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs")
    parser.add_argument("--steps", type=int, default=40, help="Steps per epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet margin")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--hard_negatives", type=str, default=None, help="Path to hard negatives JSON")
    parser.add_argument("--output", type=str, default="visual_similarity.h5", help="Encoder output path")

    args = parser.parse_args()

    train_siamese_network(
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        batch_size=args.batch_size,
        margin=args.margin,
        learning_rate=args.lr,
        hard_negatives_file=args.hard_negatives,
        save_encoder_path=args.output
    )
