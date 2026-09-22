import tensorflow as tf
from tensorflow.keras import layers, models, Model

@tf.keras.utils.register_keras_serializable(package="Custom")
class L2NormalizationLayer(layers.Layer):
    """Custom Keras Layer performing L2 normalization on embeddings."""
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
    Builds the shared CNN encoder as specified:
    Input: 224 × 224 × 3
    Conv2D(32, 3x3) ReLU -> Conv2D(32, 3x3) ReLU -> MaxPooling2D
    Conv2D(64, 3x3) ReLU -> Conv2D(64, 3x3) ReLU -> MaxPooling2D
    Conv2D(128, 3x3) ReLU -> Conv2D(128, 3x3) ReLU -> MaxPooling2D
    Conv2D(256, 3x3) ReLU -> GlobalAveragePooling2D -> Dense(256) -> L2 Normalize
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Output Embedding Layer
    x = layers.Dense(embedding_dim, name="dense_embedding")(x)
    outputs = L2NormalizationLayer(name="l2_norm_output")(x)

    encoder = Model(inputs=inputs, outputs=outputs, name="shared_cnn_encoder")
    return encoder

@tf.keras.utils.register_keras_serializable(package="Custom")
class TripletLossLayer(layers.Layer):
    """
    Custom Layer to compute Triplet Loss:
    L = max( distance(A, P) - distance(A, N) + margin, 0 )
    """
    def __init__(self, margin=0.3, distance_metric="euclidean", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.distance_metric = distance_metric

    def compute_distance(self, u, v):
        if self.distance_metric == "squared_euclidean":
            return tf.reduce_sum(tf.square(u - v), axis=-1)
        else: # euclidean
            return tf.sqrt(tf.reduce_sum(tf.square(u - v), axis=-1) + 1e-8)

    def call(self, inputs):
        anchor, positive, negative = inputs
        d_pos = self.compute_distance(anchor, positive)
        d_neg = self.compute_distance(anchor, negative)

        loss = tf.maximum(d_pos - d_neg + self.margin, 0.0)
        self.add_loss(tf.reduce_mean(loss))
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "margin": self.margin,
            "distance_metric": self.distance_metric,
        })
        return config

def build_siamese_triplet_network(encoder, margin=0.3, distance_metric="euclidean"):
    """
    Builds Siamese Triplet Network sharing the exact same encoder weights across Anchor, Positive, Negative branches.
    """
    input_shape = encoder.input_shape[1:]

    anchor_input = layers.Input(shape=input_shape, name="anchor_input")
    positive_input = layers.Input(shape=input_shape, name="positive_input")
    negative_input = layers.Input(shape=input_shape, name="negative_input")

    anchor_emb = encoder(anchor_input)
    positive_emb = encoder(positive_input)
    negative_emb = encoder(negative_input)

    loss_output = TripletLossLayer(margin=margin, distance_metric=distance_metric, name="triplet_loss")(
        [anchor_emb, positive_emb, negative_emb]
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_emb, positive_emb, negative_emb],
        name="siamese_triplet_network"
    )
    return siamese_network

if __name__ == "__main__":
    encoder = build_encoder()
    encoder.summary()
