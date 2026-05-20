#!/usr/bin/env bash
# =============================================================================
# run_project.sh  —  Alzheimer's MRI Classification System
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  Alzheimer's MRI Classification — Project Runner"
echo "============================================================"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.9+"
    exit 1
fi
PYTHON=$(command -v python3)
echo "[INFO] Using Python: $PYTHON ($(python3 --version))"
echo ""

# ── Parse argument ────────────────────────────────────────────────────────────
MODE=${1:-"app"}    # default: launch Streamlit app

case "$MODE" in

  install)
    echo "[INFO] Installing dependencies …"
    pip install -r requirements.txt
    echo ""
    echo "[OK] All dependencies installed."
    ;;

  train)
    echo "[INFO] Starting training pipeline …"
    echo "       Dataset dir  : ${DATASET_DIR:-dataset}"
    echo "       Epochs Phase1: ${EPOCHS_P1:-20}"
    echo "       Epochs Phase2: ${EPOCHS_P2:-30}"
    echo ""
    python3 -m training.train \
        --dataset_dir "${DATASET_DIR:-dataset}" \
        --epochs_p1   "${EPOCHS_P1:-20}" \
        --epochs_p2   "${EPOCHS_P2:-30}" \
        --batch_size  "${BATCH_SIZE:-32}" \
        --save_dir    "saved_models"
    ;;

  evaluate)
    echo "[INFO] Running evaluation …"
    python3 -m training.evaluate \
        --model_path  "${MODEL_PATH:-saved_models/final_model.keras}" \
        --dataset_dir "${DATASET_DIR:-dataset}" \
        --output_dir  "saved_models/eval"
    ;;

  app)
    echo "[INFO] Launching Streamlit app …"
    echo "       → Open your browser at http://localhost:8501"
    echo ""
    streamlit run app/streamlit_app.py \
        --server.port 8501 \
        --server.address 0.0.0.0
    ;;

  test)
    echo "[INFO] Running preprocessing sanity tests …"
    python3 -m preprocessing.preprocess
    python3 -m utils.metrics
    python3 -m models.gradcam
    echo ""
    echo "[OK] All module tests passed."
    ;;

  *)
    echo "Usage: $0 [install|train|evaluate|app|test]"
    echo ""
    echo "  install   — Install all Python dependencies"
    echo "  train     — Train the ensemble model (Phase 1 + Phase 2)"
    echo "  evaluate  — Evaluate saved model on test set"
    echo "  app       — Launch Streamlit web application (default)"
    echo "  test      — Run module sanity tests"
    echo ""
    echo "Environment variables:"
    echo "  DATASET_DIR  — path to dataset root (default: dataset)"
    echo "  MODEL_PATH   — path to saved model (default: saved_models/final_model.keras)"
    echo "  EPOCHS_P1    — Phase 1 epochs (default: 20)"
    echo "  EPOCHS_P2    — Phase 2 epochs (default: 30)"
    echo "  BATCH_SIZE   — Batch size (default: 32)"
    exit 1
    ;;
esac
