"""
training/evaluate.py
====================
Load a saved model and evaluate on the test set.
Produces:
  - Classification report
  - Confusion matrix plot
  - ROC curves plot
  - Grad-CAM samples for each class

Usage:
    cd alzheimer_ai
    python -m training.evaluate --model_path saved_models/final_model.keras \
                                 --dataset_dir dataset
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from utils.data_loader import build_generators, CLASS_NAMES, BATCH_SIZE
from utils.metrics import (
    compute_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curves,
)
from models.gradcam import generate_gradcam_figure


def evaluate(model_path: str, dataset_dir: str,
             batch_size: int = BATCH_SIZE,
             output_dir: str = "saved_models/eval"):
    """
    Evaluate a saved model on the test split.

    Args:
        model_path  : path to saved .keras model
        dataset_dir : root dataset directory
        batch_size  : batch size for inference
        output_dir  : directory to save result plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Model loaded OK")

    # ── Data ──────────────────────────────────────────────────────────────
    _, _, test_gen = build_generators(dataset_dir, batch_size)
    test_steps = max(1, test_gen.n // batch_size)

    # ── Predictions ───────────────────────────────────────────────────────
    print("\n[INFO] Running predictions on test set …")
    y_prob_list, y_true_list = [], []

    for step, (batch_x, batch_y) in enumerate(test_gen):
        if step >= test_steps:
            break
        preds = model.predict(batch_x, verbose=0)
        y_prob_list.append(preds)
        y_true_list.append(np.argmax(batch_y, axis=1))

    y_prob = np.concatenate(y_prob_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics)

    # ── Confusion Matrix ──────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    # ── ROC Curves ────────────────────────────────────────────────────────
    plot_roc_curves(
        y_true, y_prob,
        save_path=os.path.join(output_dir, "roc_curves.png"),
    )

    # ── Grad-CAM samples ──────────────────────────────────────────────────
    print("\n[INFO] Generating Grad-CAM samples …")
    _generate_gradcam_samples(model, test_gen, output_dir)

    print(f"\n[INFO] Evaluation complete. Results → {output_dir}/")
    return metrics


def _generate_gradcam_samples(model, test_gen, output_dir: str, n_samples: int = 4):
    """Generate one Grad-CAM figure per class from first test batch."""
    try:
        batch_x, batch_y = next(test_gen)
        for cls_idx in range(len(CLASS_NAMES)):
            # Find first sample of this class
            indices = np.where(np.argmax(batch_y, axis=1) == cls_idx)[0]
            if len(indices) == 0:
                continue
            idx = indices[0]
            img_array = batch_x[idx : idx + 1]          # (1,224,224,3)
            orig_img  = (batch_x[idx] * 255).astype(np.uint8)

            save_path = os.path.join(
                output_dir, f"gradcam_{CLASS_NAMES[cls_idx]}.png"
            )
            generate_gradcam_figure(
                model, img_array, orig_img,
                CLASS_NAMES, class_index=cls_idx,
                save_path=save_path,
            )
    except Exception as e:
        print(f"[WARN] Grad-CAM sample generation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Alzheimer MRI model")
    parser.add_argument("--model_path",  type=str, default="saved_models/final_model.keras")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--output_dir",  type=str, default="saved_models/eval")
    args = parser.parse_args()

    evaluate(args.model_path, args.dataset_dir, args.batch_size, args.output_dir)
