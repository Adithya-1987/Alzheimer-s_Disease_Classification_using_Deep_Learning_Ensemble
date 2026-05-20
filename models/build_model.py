"""
models/build_model.py
=====================
Ensemble model: ResNet50 + EfficientNetB3 with a shared classification head.

Architecture:
  Branch 1: ResNet50 (pretrained ImageNet, include_top=False) → GlobalAveragePooling2D
  Branch 2: EfficientNetB3 (pretrained ImageNet, include_top=False) → GlobalAveragePooling2D
  Fusion   : Concatenate([branch1, branch2])
  Head     : BN → Dense(512, relu, L1L2) → BN → Dropout(0.6)
             → Dense(256, relu) → Dropout(0.5)
             → Dense(128, relu) → Dropout(0.4)
             → Dense(4, softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import ResNet50, EfficientNetB3
import os


# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SHAPE  = (224, 224, 3)
NUM_CLASSES = 4


# ─────────────────────────────────────────────────────────────────────────────
# Branch builders
# ─────────────────────────────────────────────────────────────────────────────
def _build_resnet_branch(input_tensor: tf.Tensor) -> tf.Tensor:
    """
    ResNet50 branch.
    Returns GlobalAveragePooling2D feature vector (shape: 2048).
    """
    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=input_tensor,
    )
    base.trainable = False  # frozen during Phase 1
    x = base.output
    x = layers.GlobalAveragePooling2D(name="resnet_gap")(x)
    return x, base


def _build_efficientnet_branch(input_tensor: tf.Tensor):
    """
    EfficientNetB3 branch.
    Returns GlobalAveragePooling2D feature vector (shape: 1536).
    """
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=input_tensor,
    )
    base.trainable = False  # frozen during Phase 1
    x = base.output
    x = layers.GlobalAveragePooling2D(name="efficientnet_gap")(x)
    return x, base


# ─────────────────────────────────────────────────────────────────────────────
# Classification Head
# ─────────────────────────────────────────────────────────────────────────────
def _build_classification_head(features: tf.Tensor,
                                num_classes: int = NUM_CLASSES) -> tf.Tensor:
    """
    Shared classification head applied to fused features.

    Args:
        features   : concatenated feature tensor
        num_classes: number of output classes

    Returns:
        Output tensor with shape (None, num_classes)
    """
    # Block 1
    x = layers.BatchNormalization(name="bn_1")(features)
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        name="dense_512",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.6, name="drop_0_6")(x)

    # Block 2
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="drop_0_5")(x)

    # Block 3
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.4, name="drop_0_4")(x)

    # Output
    output = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Public API: build_ensemble_model
# ─────────────────────────────────────────────────────────────────────────────
def build_ensemble_model(input_shape: tuple = IMG_SHAPE,
                         num_classes: int = NUM_CLASSES):
    """
    Build and return the complete ensemble model (ResNet50 + EfficientNetB3).

    Args:
        input_shape : (H, W, C) — default (224, 224, 3)
        num_classes : number of output classes — default 4

    Returns:
        (model, resnet_base, efficientnet_base)
          model             : compiled Keras Model (Phase 1 config)
          resnet_base       : ResNet50 base model (for fine-tuning access)
          efficientnet_base : EfficientNetB3 base model (for fine-tuning access)
    """
    # Shared input
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Branch 1: ResNet50
    resnet_features, resnet_base = _build_resnet_branch(inputs)

    # Branch 2: EfficientNetB3
    efficient_features, efficient_base = _build_efficientnet_branch(inputs)

    # Feature Fusion: Concatenate
    fused = layers.Concatenate(name="feature_fusion")(
        [resnet_features, efficient_features]
    )

    # Classification Head
    outputs = _build_classification_head(fused, num_classes)

    # Assemble Model
    model = Model(inputs=inputs, outputs=outputs, name="AlzheimerEnsemble")

    # Compile for Phase 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model, resnet_base, efficient_base


# ─────────────────────────────────────────────────────────────────────────────
# Fine-Tuning helpers
# ─────────────────────────────────────────────────────────────────────────────
def prepare_fine_tuning(model: Model,
                        resnet_base,
                        efficient_base,
                        resnet_unfreeze_last: int = 60,
                        efficient_unfreeze_last: int = 50) -> Model:
    """
    Phase 2 fine-tuning: selectively unfreeze the last N layers of each base.

    Args:
        model                 : the ensemble model
        resnet_base           : ResNet50 base
        efficient_base        : EfficientNetB3 base
        resnet_unfreeze_last  : unfreeze last N layers of ResNet50
        efficient_unfreeze_last: unfreeze last N layers of EfficientNetB3

    Returns:
        Recompiled model ready for fine-tuning
    """
    # Freeze all first
    resnet_base.trainable    = True
    efficient_base.trainable = True

    # Freeze all but last N layers for ResNet
    total_resnet = len(resnet_base.layers)
    for layer in resnet_base.layers[: total_resnet - resnet_unfreeze_last]:
        layer.trainable = False

    # Freeze all but last N layers for EfficientNet
    total_eff = len(efficient_base.layers)
    for layer in efficient_base.layers[: total_eff - efficient_unfreeze_last]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    trainable_count = sum(
        1 for l in model.layers if l.trainable and len(l.get_weights()) > 0
    )
    print(f"[INFO] Fine-tuning: {trainable_count} trainable layers unlocked.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Building ensemble model …")
    model, resnet_base, eff_base = build_ensemble_model()
    model.summary(line_length=100)
    print(f"\n[OK] Output shape: {model.output_shape}")
    print(f"[OK] Total params: {model.count_params():,}")
