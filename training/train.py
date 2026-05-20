"""
training/train.py
=================
Two-phase training pipeline.
  Phase 1: Frozen base models, train classification head (20 epochs)
  Phase 2: Fine-tune last layers of ResNet50 & EfficientNetB3 (30 epochs)

Usage:
    cd alzheimer_ai
    python -m training.train --dataset_dir dataset --epochs_p1 20 --epochs_p2 30
"""

import argparse
import os
import sys
import json
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard,
)

from models.build_model import build_ensemble_model, prepare_fine_tuning
from utils.data_loader  import build_generators, BATCH_SIZE
from utils.metrics      import plot_training_history


def make_callbacks(phase: int, save_dir: str = "saved_models") -> list:
    """Build EarlyStopping, ReduceLR, ModelCheckpoint, TensorBoard callbacks."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"best_model_phase{phase}.keras")
    return [
        EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-7, verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        TensorBoard(log_dir=os.path.join("logs", f"phase{phase}"), histogram_freq=0),
    ]


def _save_history(hist_dict: dict, save_dir: str, filename: str):
    """Serialize Keras history to JSON."""
    serializable = {k: [float(v) for v in vals] for k, vals in hist_dict.items()}
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[INFO] History saved → {path}")


def train(dataset_dir: str, epochs_p1: int = 20, epochs_p2: int = 30,
          batch_size: int = BATCH_SIZE, save_dir: str = "saved_models"):
    """Full two-phase training routine."""
    os.makedirs(save_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[INFO] Loading data generators …")
    train_gen, val_gen, test_gen = build_generators(dataset_dir, batch_size)
    print(f"[INFO]   Train: {train_gen.n}  Val: {val_gen.n}  Test: {test_gen.n}")

    steps_per_epoch = max(1, train_gen.n // batch_size)
    val_steps       = max(1, val_gen.n   // batch_size)

    # ── Build model ───────────────────────────────────────────────────────
    print("\n[INFO] Building ensemble model …")
    model, resnet_base, eff_base = build_ensemble_model()
    print(f"[INFO]   Parameters: {model.count_params():,}")

    # ── Phase 1: frozen bases ─────────────────────────────────────────────
    print(f"\n{'='*55}\n  PHASE 1 — {epochs_p1} epochs (bases frozen)\n{'='*55}")
    history_p1 = model.fit(
        train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs_p1,
        validation_data=val_gen, validation_steps=val_steps,
        callbacks=make_callbacks(phase=1, save_dir=save_dir), verbose=1,
    )
    _save_history(history_p1.history, save_dir, "history_phase1.json")
    plot_training_history(history_p1, save_dir=os.path.join(save_dir, "plots"))

    # ── Phase 2: fine-tuning ──────────────────────────────────────────────
    print(f"\n{'='*55}\n  PHASE 2 — {epochs_p2} epochs (fine-tuning)\n{'='*55}")
    model = prepare_fine_tuning(model, resnet_base, eff_base,
                                resnet_unfreeze_last=60,
                                efficient_unfreeze_last=50)
    history_p2 = model.fit(
        train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs_p2,
        validation_data=val_gen, validation_steps=val_steps,
        callbacks=make_callbacks(phase=2, save_dir=save_dir), verbose=1,
    )
    _save_history(history_p2.history, save_dir, "history_phase2.json")

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "final_model.keras")
    model.save(final_path)
    print(f"\n[INFO] Final model → {final_path}")
    return model, history_p1, history_p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alzheimer MRI Ensemble")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--epochs_p1",   type=int, default=20)
    parser.add_argument("--epochs_p2",   type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--save_dir",    type=str, default="saved_models")
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train(args.dataset_dir, args.epochs_p1, args.epochs_p2,
          args.batch_size, args.save_dir)
