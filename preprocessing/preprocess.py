"""
preprocessing/preprocess.py
============================
Image preprocessing pipeline for Alzheimer's MRI classification.
Steps:
  1. Resize to 224x224
  2. CLAHE enhancement (improves local contrast in MRI scans)
  3. Gaussian blur (reduces noise)
  4. Normalize pixel values to [0, 1]
"""

import cv2
import numpy as np


def apply_clahe(image: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE to L channel of LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def apply_gaussian_blur(image: np.ndarray,
                        kernel_size: tuple = (3, 3),
                        sigma: float = 0.5) -> np.ndarray:
    """Apply light Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(image, kernel_size, sigma)


def preprocess_image(image: np.ndarray,
                     target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline: Resize → CLAHE → Gaussian Blur → Normalize

    Args:
        image       : Input numpy array (uint8, RGB)
        target_size : Target (width, height)

    Returns:
        float32 array of shape (224, 224, 3) in [0, 1]
    """
    # Ensure RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # 1. Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # 2. CLAHE enhancement
    image = apply_clahe(image)

    # 3. Gaussian blur
    image = apply_gaussian_blur(image)

    # 4. Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def load_and_preprocess(filepath: str,
                        target_size: tuple = (224, 224)) -> np.ndarray:
    """Load image from disk and preprocess."""
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return preprocess_image(img_rgb, target_size)


if __name__ == "__main__":
    dummy = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    result = preprocess_image(dummy)
    print(f"[OK] Output shape : {result.shape}")
    print(f"[OK] Value range  : [{result.min():.3f}, {result.max():.3f}]")
    print(f"[OK] dtype        : {result.dtype}")
