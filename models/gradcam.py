"""
models/gradcam.py
=================
Grad-CAM visualization for the ensemble model.
Uses the last convolutional layer of the ResNet50 branch to produce
a heatmap overlay on the input MRI image.

Reference:
  Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization"
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─── Last conv layer name for ResNet50 ───────────────────────────────────────
RESNET_LAST_CONV = "conv5_block3_out"   # output of ResNet50's last residual block


# ─────────────────────────────────────────────────────────────────────────────
# Core Grad-CAM computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_gradcam(model: tf.keras.Model,
                    img_array: np.ndarray,
                    class_index: int = None,
                    last_conv_layer_name: str = RESNET_LAST_CONV) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a given image and class.

    Args:
        model               : the full ensemble Keras model
        img_array           : preprocessed image, shape (1, 224, 224, 3), float32 [0,1]
        class_index         : index of the target class (if None, uses predicted class)
        last_conv_layer_name: name of the last conv layer in ResNet50 branch

    Returns:
        heatmap : float32 array of shape (H, W) normalized to [0, 1]
    """
    # Build a sub-model: input → [conv_output, final_predictions]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    # Record operations for gradient computation
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)

        if class_index is None:
            class_index = tf.argmax(predictions[0]).numpy()

        # Score for the target class
        class_score = predictions[:, class_index]

    # Gradients of class score w.r.t. conv feature maps
    grads = tape.gradient(class_score, conv_outputs)

    # Global average pooling of gradients → importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv feature maps by their importance
    conv_outputs = conv_outputs[0]                        # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────
def overlay_heatmap_on_image(original_img: np.ndarray,
                             heatmap: np.ndarray,
                             alpha: float = 0.45,
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Resize heatmap to image size and blend it with the original image.

    Args:
        original_img : uint8 RGB image (H, W, 3)
        heatmap      : float32 (h, w) in [0, 1]
        alpha        : blending weight for heatmap (0 = no overlay)
        colormap     : OpenCV colormap constant

    Returns:
        superimposed : uint8 RGB image with heatmap overlay
    """
    # Resize heatmap to match image dimensions
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # applyColorMap returns BGR → convert to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    superimposed = (alpha * heatmap_rgb.astype(np.float32) +
                    (1 - alpha) * original_img.astype(np.float32))
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return superimposed


# ─────────────────────────────────────────────────────────────────────────────
# High-level: generate Grad-CAM figure
# ─────────────────────────────────────────────────────────────────────────────
def generate_gradcam_figure(model: tf.keras.Model,
                            img_array: np.ndarray,
                            original_img: np.ndarray,
                            class_names: list,
                            class_index: int = None,
                            last_conv_layer_name: str = RESNET_LAST_CONV,
                            save_path: str = None) -> plt.Figure:
    """
    Full pipeline: compute Grad-CAM → overlay → create figure.

    Args:
        model               : ensemble model
        img_array           : preprocessed image (1, 224, 224, 3)
        original_img        : uint8 RGB image for display (any size)
        class_names         : list of class name strings
        class_index         : target class index (None → predicted)
        last_conv_layer_name: ResNet last conv layer name
        save_path           : if provided, save figure here

    Returns:
        matplotlib Figure with three panels:
        [Original | Heatmap | Overlay]
    """
    # 1. Compute heatmap
    heatmap = compute_gradcam(model, img_array, class_index,
                               last_conv_layer_name)

    # 2. Resize original to 224x224 for display
    display_img = cv2.resize(original_img, (224, 224))

    # 3. Create overlay
    overlay = overlay_heatmap_on_image(display_img, heatmap)

    # 4. Colormap for standalone heatmap panel
    heatmap_vis = np.uint8(255 * cv2.resize(heatmap, (224, 224)))
    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 5. Predict for label
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = class_names[pred_idx]
    pred_conf  = preds[pred_idx]

    # 6. Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Grad-CAM  |  Prediction: {pred_class}  ({pred_conf:.1%})",
        fontsize=14, fontweight="bold"
    )

    axes[0].imshow(display_img)
    axes[0].set_title("Original MRI", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap_colored)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Grad-CAM figure saved → {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helper: return overlay as numpy array (for Streamlit display)
# ─────────────────────────────────────────────────────────────────────────────
def get_gradcam_overlay(model: tf.keras.Model,
                        img_array: np.ndarray,
                        original_img: np.ndarray,
                        class_index: int = None,
                        last_conv_layer_name: str = RESNET_LAST_CONV) -> np.ndarray:
    """
    Lightweight version — returns overlay as uint8 RGB numpy array.
    Suitable for direct display in Streamlit (st.image).

    Args:
        model            : ensemble model
        img_array        : preprocessed (1, 224, 224, 3)
        original_img     : uint8 RGB image
        class_index      : target class index
        last_conv_layer_name: ResNet last conv layer

    Returns:
        overlay : uint8 RGB numpy array (224, 224, 3)
    """
    heatmap = compute_gradcam(model, img_array, class_index,
                               last_conv_layer_name)
    display_img = cv2.resize(original_img, (224, 224))
    return overlay_heatmap_on_image(display_img, heatmap)


if __name__ == "__main__":
    print("[OK] gradcam.py loaded — Grad-CAM utilities ready.")
