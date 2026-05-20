"""
utils/metrics.py
================
Evaluation metrics for multi-class classification:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix (with visualization)
  - ROC AUC (one-vs-rest, macro average)
  - Full classification report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]


# ─── Core metric computation ──────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true : 1-D array of true class indices
        y_pred : 1-D array of predicted class indices
        y_prob : 2-D array of predicted probabilities (N, num_classes)

    Returns:
        dict with keys: accuracy, precision, recall, f1, roc_auc, report
    """
    metrics = {}

    metrics["accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred,
                                           average="weighted",
                                           zero_division=0)
    metrics["recall"]    = recall_score(y_true, y_pred,
                                        average="weighted",
                                        zero_division=0)
    metrics["f1"]        = f1_score(y_true, y_pred,
                                    average="weighted",
                                    zero_division=0)

    # ROC AUC (requires probability scores)
    if y_prob is not None:
        y_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
        try:
            metrics["roc_auc"] = roc_auc_score(y_bin, y_prob,
                                                multi_class="ovr",
                                                average="macro")
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    metrics["report"] = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print computed metrics."""
    print("\n" + "=" * 55)
    print("  EVALUATION METRICS")
    print("=" * 55)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")
    print("=" * 55)
    print("\nClassification Report:")
    print(metrics["report"])


# ─── Confusion Matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: str = None) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true     : true class indices
        y_pred     : predicted class indices
        save_path  : if given, save figure to this path

    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] Confusion matrix saved → {save_path}")

    return fig


# ─── ROC Curves ──────────────────────────────────────────────────────────────
def plot_roc_curves(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    save_path: str = None) -> plt.Figure:
    """
    Plot one-vs-rest ROC curves for each class.

    Args:
        y_true    : true class indices
        y_prob    : predicted probabilities (N, num_classes)
        save_path : if given, save figure to this path

    Returns:
        matplotlib Figure
    """
    n_classes = len(CLASS_NAMES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – One vs Rest", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] ROC curves saved → {save_path}")

    return fig


# ─── Training History ─────────────────────────────────────────────────────────
def plot_training_history(history, save_dir: str = None):
    """
    Plot accuracy and loss curves from a Keras History object.

    Args:
        history  : Keras History object (or dict with 'accuracy', 'loss', etc.)
        save_dir : directory to save figures (optional)
    """
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(hist.get("accuracy", []), label="Train Acc", lw=2)
    axes[0].plot(hist.get("val_accuracy", []), label="Val Acc", lw=2)
    axes[0].set_title("Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(hist.get("loss", []), label="Train Loss", lw=2)
    axes[1].plot(hist.get("val_loss", []), label="Val Loss", lw=2)
    axes[1].set_title("Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "training_history.png")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Training history plot saved → {path}")

    return fig


if __name__ == "__main__":
    # Sanity check with random data
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 4, 100)
    y_pred = rng.integers(0, 4, 100)
    y_prob = rng.dirichlet(np.ones(4), 100)

    m = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(m)
    print("[OK] metrics.py loaded successfully")
