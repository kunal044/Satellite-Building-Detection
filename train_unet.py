"""
train_unet.py
-------------
Defines a U-Net architecture for binary semantic segmentation (building vs.
background) and trains it on the satellite dataset.

U-Net overview
──────────────
  Encoder (contracting path) — four downsampling blocks, each consisting of
  two 3×3 Conv → BatchNorm → ReLU layers followed by a 2×2 MaxPool.

  Bottleneck — two Conv layers at the lowest spatial resolution.

  Decoder (expanding path) — four upsampling blocks, each consisting of a
  transposed convolution (up-conv) to double the spatial resolution, a skip
  connection from the mirrored encoder block (concatenation), and two 3×3
  Conv → BatchNorm → ReLU layers.

  Output — 1×1 convolution with sigmoid activation producing a
  (H, W, 1) probability map where each value represents the likelihood
  that the corresponding pixel belongs to a building.

Saved artefacts
───────────────
  unet_model.h5         — final trained model weights
  training_history.png  — loss and IoU curves plotted with Matplotlib
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Use non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from dataset_loader import load_dataset
from preprocessing  import split_dataset

# ── Hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE      = 256
BATCH_SIZE    = 8
EPOCHS        = 10
LEARNING_RATE = 1e-4
MODEL_PATH    = "unet_model.h5"
HISTORY_PLOT  = "training_history.png"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# U-Net building blocks
# ═══════════════════════════════════════════════════════════════════════════════

def conv_block(x: tf.Tensor, filters: int, kernel_size: int = 3) -> tf.Tensor:
    """
    Two consecutive Conv → BatchNorm → ReLU layers.

    Parameters
    ----------
    x           : Input feature map tensor.
    filters     : Number of output feature maps for each conv layer.
    kernel_size : Spatial size of the convolutional kernel (default 3).

    Returns
    -------
    Output tensor after two convolutions.
    """
    x = layers.Conv2D(
        filters, kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters, kernel_size,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def encoder_block(x: tf.Tensor, filters: int) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Encoder block: conv_block followed by MaxPooling.

    Returns
    -------
    skip : Feature map *before* pooling (used as a skip connection).
    pool : Feature map *after* 2×2 max-pooling (passed to the next block).
    """
    skip = conv_block(x, filters)
    pool = layers.MaxPooling2D(pool_size=(2, 2))(skip)
    return skip, pool


def decoder_block(
    x:      tf.Tensor,
    skip:   tf.Tensor,
    filters: int,
) -> tf.Tensor:
    """
    Decoder block: transposed conv (up-sampling) → concatenate skip → conv_block.

    Parameters
    ----------
    x       : Low-resolution feature map coming from the previous decoder layer.
    skip    : Skip-connection tensor from the mirrored encoder block.
    filters : Number of feature maps.

    Returns
    -------
    Output tensor at doubled spatial resolution.
    """
    # Transposed convolution doubles spatial dimensions
    x = layers.Conv2DTranspose(
        filters, (2, 2),
        strides=(2, 2),
        padding="same",
    )(x)

    # Concatenate along the channel axis
    x = layers.Concatenate()([x, skip])

    # Two consecutive convolutions to refine merged features
    x = conv_block(x, filters)
    return x


def build_unet(
    input_shape: tuple = (IMG_SIZE, IMG_SIZE, 3),
    num_classes: int   = 1,
) -> Model:
    """
    Build and return a U-Net Keras model for binary segmentation.

    Parameters
    ----------
    input_shape : Shape of one input image (H, W, C).
    num_classes : Number of output channels. Use 1 for binary segmentation
                  with sigmoid activation.

    Returns
    -------
    A compiled-ready Keras Model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ── Encoder ───────────────────────────────────────────────────────────────
    # Each block halves the spatial dimensions and doubles the feature depth.
    skip1, p1 = encoder_block(inputs, filters=64)    # 256 → 128
    skip2, p2 = encoder_block(p1,     filters=128)   # 128 →  64
    skip3, p3 = encoder_block(p2,     filters=256)   #  64 →  32
    skip4, p4 = encoder_block(p3,     filters=512)   #  32 →  16

    # ── Bottleneck ────────────────────────────────────────────────────────────
    bottleneck = conv_block(p4, filters=1024)         # 16 × 16 × 1024

    # ── Decoder ───────────────────────────────────────────────────────────────
    # Each block doubles the spatial dimensions and merges with the
    # corresponding encoder skip connection.
    d1 = decoder_block(bottleneck, skip4, filters=512)  # 16 →  32
    d2 = decoder_block(d1,         skip3, filters=256)  # 32 →  64
    d3 = decoder_block(d2,         skip2, filters=128)  # 64 → 128
    d4 = decoder_block(d3,         skip1, filters=64)   # 128 → 256

    # ── Output ────────────────────────────────────────────────────────────────
    # 1×1 conv collapses all feature maps into a single probability map.
    outputs = layers.Conv2D(
        num_classes, (1, 1),
        activation="sigmoid",
        name="output_mask",
    )(d4)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Custom metrics
# ═══════════════════════════════════════════════════════════════════════════════

def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """
    Intersection-over-Union (Jaccard Index) for binary masks.

    IoU = |A ∩ B| / |A ∪ B|

    A smooth term (1e-6) prevents division by zero on empty masks.
    """
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union        = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, save_path: str = HISTORY_PLOT) -> None:
    """
    Plot training and validation loss / IoU curves and save to *save_path*.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history.history["loss"],     label="Train loss",     linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val   loss",     linewidth=2, linestyle="--")
    axes[0].set_title("Binary Cross-Entropy Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # IoU curve
    iou_key     = [k for k in history.history if "iou" in k and "val" not in k][0]
    val_iou_key = [k for k in history.history if "iou" in k and "val" in k][0]
    axes[1].plot(history.history[iou_key],     label="Train IoU", linewidth=2)
    axes[1].plot(history.history[val_iou_key], label="Val   IoU", linewidth=2, linestyle="--")
    axes[1].set_title("Intersection-over-Union (IoU)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("U-Net Training History — Satellite Building Detection", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train_unet] Training history saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main training entry point
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    model_path: str = MODEL_PATH,
    epochs:     int = EPOCHS,
) -> None:
    """
    End-to-end training pipeline:
      1. Load dataset
      2. Preprocess and split
      3. Build U-Net
      4. Compile with Adam + BCE
      5. Train with callbacks
      6. Save model and history plot
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[train_unet] ── Step 1: Loading dataset ──────────────────────────")
    images, masks = load_dataset()

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print("\n[train_unet] ── Step 2: Splitting dataset ────────────────────────")
    splits = split_dataset(images, masks, augment_train=True)
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("\n[train_unet] ── Step 3: Building U-Net ──────────────────────────")
    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model.summary()

    # ── 4. Compile ────────────────────────────────────────────────────────────
    print("\n[train_unet] ── Step 4: Compiling model ─────────────────────────")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", iou_metric],
    )

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        # Save the best weights (lowest val_loss) during training
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Stop early if val_loss does not improve for 5 consecutive epochs
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Halve the learning rate when val_loss plateaus for 3 epochs
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print(f"\n[train_unet] ── Step 5: Training ({epochs} epochs) ──────────────")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    model.save(model_path)
    print(f"\n[train_unet] Model saved → {model_path}")

    plot_training_history(history)

    # Print final epoch metrics
    final_epoch = len(history.history["loss"])
    print(f"\n[train_unet] Training complete after {final_epoch} epoch(s).")
    print(f"  Final train loss : {history.history['loss'][-1]:.4f}")
    print(f"  Final val   loss : {history.history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    train()
