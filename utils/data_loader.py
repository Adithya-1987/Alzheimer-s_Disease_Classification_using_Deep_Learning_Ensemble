"""
utils/data_loader.py
====================
Custom Keras data generator with preprocessing + augmentation pipeline.
Expects the OASIS dataset in the structure:
    dataset/
        train/
            MildDemented/    *.jpg
            ModerateDemented/*.jpg
            NonDemented/     *.jpg
            VeryMildDemented/*.jpg
        val/   (same structure)
        test/  (same structure)
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from preprocessing.preprocess import preprocess_image

# ─── Class definitions ────────────────────────────────────────────────────────
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32


# ─── Augmentation generator (train only) ─────────────────────────────────────
def get_augmentation_generator() -> ImageDataGenerator:
    """
    Returns a Keras ImageDataGenerator with the specified augmentation params.
    Preprocessing is handled BEFORE the generator, so rescale=None here.
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )


# ─── Preprocessed generator wrapper ─────────────────────────────────────────
class PreprocessedGenerator(Sequence):
    """
    Wraps a Keras flow_from_directory generator and applies the custom
    preprocessing pipeline (CLAHE + Gaussian blur + normalize) on each batch.
    """

    def __init__(self, keras_gen):
        self.gen = keras_gen

    def __len__(self):
        return len(self.gen)

    def __getitem__(self, idx):
        batch_x, batch_y = self.gen[idx]
        # batch_x already in [0,255] float from Keras; convert back to uint8
        # then apply our pipeline
        processed = np.zeros_like(batch_x, dtype=np.float32)
        for i, img in enumerate(batch_x):
            uint8_img = np.clip(img, 0, 255).astype(np.uint8)
            processed[i] = preprocess_image(uint8_img, target_size=IMG_SIZE)
        return processed, batch_y

    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if hasattr(self.gen, 'on_epoch_end'):
            self.gen.on_epoch_end()

    @property
    def n(self):
        return self.gen.n

    @property
    def batch_size(self):
        return self.gen.batch_size

    @property
    def class_indices(self):
        return self.gen.class_indices


# ─── Build generators ─────────────────────────────────────────────────────────
def build_generators(dataset_dir: str, batch_size: int = BATCH_SIZE):
    """
    Build train / val / test generators.

    Args:
        dataset_dir : path to the root dataset folder (contains train/val/test)
        batch_size  : samples per batch

    Returns:
        (train_gen, val_gen, test_gen) — all PreprocessedGenerator instances
    """
    aug_gen = get_augmentation_generator()
    # No augmentation for val/test (rescale only; our pipeline handles it)
    plain_gen = ImageDataGenerator()

    train_dir = os.path.join(dataset_dir, "train")
    val_dir   = os.path.join(dataset_dir, "val")
    test_dir  = os.path.join(dataset_dir, "test")

    train_raw = aug_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_raw = plain_gen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    test_raw = plain_gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return (
        PreprocessedGenerator(train_raw),
        PreprocessedGenerator(val_raw),
        PreprocessedGenerator(test_raw),
    )


if __name__ == "__main__":
    # Quick test: build generators from dummy paths
    print("CLASS_NAMES:", CLASS_NAMES)
    print("data_loader module loaded OK")
