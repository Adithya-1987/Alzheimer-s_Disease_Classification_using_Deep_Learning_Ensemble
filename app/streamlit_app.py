"""
app/streamlit_app.py
====================
Streamlit web application for Alzheimer's MRI Classification.

Features:
  1. Upload MRI image (JPG/PNG/JPEG)
  2. Preprocess image (CLAHE + Gaussian blur + normalize)
  3. Run ensemble model inference
  4. Display prediction + class probabilities bar chart
  5. Display Grad-CAM heatmap overlay
  6. Download PDF report

Usage:
    cd alzheimer_ai
    streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import preprocess_image
from models.gradcam import get_gradcam_overlay, compute_gradcam, overlay_heatmap_on_image

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
CLASS_COLORS = ["#FF6B6B", "#FF4444", "#4CAF50", "#FFA726"]
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "saved_models", "final_model.keras")
IMG_SIZE     = (224, 224)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
      text-align: center;
      background: linear-gradient(135deg, #1a1a2e, #16213e);
      color: white;
      padding: 2rem;
      border-radius: 12px;
      margin-bottom: 2rem;
  }
  .result-card {
      background: #f8f9fa;
      padding: 1.5rem;
      border-radius: 10px;
      border-left: 5px solid #2196F3;
      margin: 1rem 0;
  }
  .metric-card {
      text-align: center;
      background: white;
      padding: 1.2rem;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  .stProgress > div > div { background-color: #2196F3; }
</style>
""", unsafe_allow_html=True)


# ─── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# ─── Inference ────────────────────────────────────────────────────────────────
def run_inference(model, img_array: np.ndarray) -> tuple:
    """
    Run model inference.

    Args:
        model     : loaded Keras model
        img_array : preprocessed (1, 224, 224, 3) float32 array

    Returns:
        (predictions, predicted_class, confidence)
    """
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx])
    return preds, pred_class, confidence


# ─── Probability bar chart ────────────────────────────────────────────────────
def plot_probabilities(probabilities: np.ndarray) -> plt.Figure:
    """Create a horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors  = [CLASS_COLORS[i] for i in range(len(CLASS_NAMES))]
    y_pos   = range(len(CLASS_NAMES))

    bars = ax.barh(y_pos, probabilities * 100,
                   color=colors, edgecolor="white", linewidth=1.5, height=0.55)

    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", ha="left", fontsize=11, fontweight="bold")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Probability (%)", fontsize=11)
    ax.set_title("Class Probabilities", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, 115)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# ─── PDF Report generation ────────────────────────────────────────────────────
def generate_report(original_img: np.ndarray,
                    overlay_img: np.ndarray,
                    pred_class: str,
                    confidence: float,
                    probabilities: np.ndarray) -> bytes:
    """
    Generate a simple PDF-style report as a PNG image (multi-panel).
    Returns bytes for download.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")

    # Title
    fig.suptitle(
        "Alzheimer's MRI Analysis Report",
        color="white", fontsize=18, fontweight="bold", y=0.97
    )

    # Timestamp & prediction
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subtitle = f"Generated: {timestamp}  |  Prediction: {pred_class}  |  Confidence: {confidence:.1%}"
    fig.text(0.5, 0.92, subtitle, ha="center", color="#90CAF9", fontsize=11)

    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3,
                           left=0.05, right=0.95, top=0.88, bottom=0.05)

    # Panel 1: original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.resize(original_img, IMG_SIZE))
    ax1.set_title("Original MRI", color="white", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Panel 2: overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(overlay_img)
    ax2.set_title("Grad-CAM Overlay", color="white", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Panel 3: probability bars
    ax3 = fig.add_subplot(gs[0, 2])
    colors = CLASS_COLORS
    bars   = ax3.barh(CLASS_NAMES, probabilities * 100,
                       color=colors, edgecolor="white", linewidth=1)
    for bar, prob in zip(bars, probabilities):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{prob:.1%}", va="center", color="white", fontsize=9)
    ax3.set_facecolor("#16213e")
    ax3.spines[:].set_color("#444")
    ax3.tick_params(colors="white", labelsize=9)
    ax3.set_xlabel("Probability (%)", color="white", fontsize=10)
    ax3.set_title("Probabilities", color="white", fontsize=12, fontweight="bold")
    ax3.set_xlim(0, 120)

    # Panel 4: summary text
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor("#0f3460")
    ax4.axis("off")
    summary_lines = [
        f"DIAGNOSIS SUMMARY",
        f"",
        f"Predicted Class  :  {pred_class}",
        f"Confidence       :  {confidence:.2%}",
        f"",
        f"Class Probabilities:",
    ]
    for i, cls in enumerate(CLASS_NAMES):
        summary_lines.append(f"  {cls:<22} {probabilities[i]:.4f}  ({probabilities[i]:.1%})")
    summary_lines += [
        f"",
        f"DISCLAIMER: This AI-generated analysis is for research purposes only.",
        f"Always consult a qualified medical professional for clinical diagnosis.",
    ]
    ax4.text(0.05, 0.95, "\n".join(summary_lines),
             transform=ax4.transAxes, va="top", ha="left",
             color="white", fontsize=10, fontfamily="monospace",
             linespacing=1.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Alzheimer%27s_disease_brain_comparison.jpg/320px-Alzheimer%27s_disease_brain_comparison.jpg",
             caption="Alzheimer's Brain Comparison", use_column_width=True)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
This system uses a **ResNet50 + EfficientNetB3** ensemble model trained on the
OASIS Augmented Alzheimer MRI Dataset to classify brain MRI scans into:

- 🟢 Non-Demented
- 🟡 Very Mild Demented
- 🟠 Mild Demented
- 🔴 Moderate Demented
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_probs   = st.checkbox("Show probability chart", value=True)
    alpha_slider = st.slider("Grad-CAM overlay alpha", 0.2, 0.8, 0.45, 0.05)


# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Alzheimer's MRI Classification System</h1>
    <p>ResNet50 + EfficientNetB3 Ensemble · Grad-CAM Explainability</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()
if model is None:
    st.error(
        "⚠️ **Model not found.**  "
        "Please train the model first:\n\n"
        "```bash\npython -m training.train --dataset_dir dataset\n```"
    )
    st.stop()

st.success("✅ Model loaded successfully")

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI scan (JPG or PNG format)",
)

if uploaded_file is not None:
    # ── Read uploaded image ────────────────────────────────────────────
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Preprocess ────────────────────────────────────────────────────
    with st.spinner("🔄 Preprocessing image …"):
        preprocessed = preprocess_image(original_rgb, target_size=IMG_SIZE)
        img_array    = np.expand_dims(preprocessed, axis=0)  # (1,224,224,3)

    # ── Inference ─────────────────────────────────────────────────────
    with st.spinner("🤖 Running model inference …"):
        probabilities, pred_class, confidence = run_inference(model, img_array)

    # ── Display ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("📷 Input MRI")
        st.image(original_rgb, caption="Uploaded MRI", use_column_width=True)
        st.image(preprocessed, caption="Preprocessed (CLAHE + Blur)", use_column_width=True)

    with col2:
        st.subheader("🔍 Prediction")
        # Prediction badge
        color_map = {
            "NonDemented":       "green",
            "VeryMildDemented":  "orange",
            "MildDemented":      "darkorange",
            "ModerateDemented":  "red",
        }
        badge_color = color_map.get(pred_class, "blue")
        st.markdown(f"""
        <div style="background:{badge_color};color:white;padding:1.2rem;border-radius:10px;text-align:center;">
            <h2>{pred_class}</h2>
            <h3>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Metric cards
        pred_idx = CLASS_NAMES.index(pred_class)
        mcols = st.columns(2)
        mcols[0].metric("Top Class",   pred_class.replace("Demented",""))
        mcols[1].metric("Confidence", f"{confidence:.1%}")

        st.markdown("---")
        if show_probs:
            st.subheader("📊 Class Probabilities")
            fig_prob = plot_probabilities(probabilities)
            st.pyplot(fig_prob, use_container_width=True)
            plt.close(fig_prob)

    with col3:
        if show_gradcam:
            st.subheader("🌡️ Grad-CAM Heatmap")
            with st.spinner("Generating Grad-CAM …"):
                try:
                    overlay = get_gradcam_overlay(
                        model, img_array, original_rgb,
                        class_index=pred_idx,
                    )
                    st.image(overlay, caption="Grad-CAM Overlay (ResNet last conv)",
                             use_column_width=True)
                    st.caption(
                        "🔴 Red = high activation (most relevant regions for prediction)"
                    )
                except Exception as e:
                    st.warning(f"Grad-CAM unavailable: {e}")

    # ── Probability table ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Detailed Results")
    result_cols = st.columns(len(CLASS_NAMES))
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        with result_cols[i]:
            delta = f"+{prob - 0.25:.2%}" if prob > 0.25 else f"{prob - 0.25:.2%}"
            st.metric(
                label=cls,
                value=f"{prob:.4f}",
                delta=delta,
            )

    # ── Download report ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Download Report")
    with st.spinner("Generating report …"):
        try:
            overlay_for_report = get_gradcam_overlay(
                model, img_array, original_rgb, class_index=pred_idx
            )
        except Exception:
            overlay_for_report = cv2.resize(original_rgb, IMG_SIZE)

        report_bytes = generate_report(
            original_rgb, overlay_for_report,
            pred_class, confidence, probabilities
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="⬇️ Download Analysis Report (PNG)",
        data=report_bytes,
        file_name=f"alzheimer_report_{timestamp}.png",
        mime="image/png",
    )

    # ── Disclaimer ────────────────────────────────────────────────────
    st.warning(
        "⚠️ **Medical Disclaimer**: This AI tool is for research and educational "
        "purposes only. It does not constitute medical advice. Always consult a "
        "qualified neurologist or medical professional for diagnosis."
    )

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center;padding:3rem;background:#f8f9fa;border-radius:12px;">
        <h2>👆 Upload an MRI Image to Begin</h2>
        <p style="color:#666;font-size:1.1rem;">
            Supported formats: JPG, JPEG, PNG<br>
            Recommended: Axial brain MRI slice
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📌 How It Works")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**1. Upload**\nUpload your MRI brain scan image")
    c2.info("**2. Preprocess**\nCLAHE enhancement + Gaussian blur + normalization")
    c3.info("**3. Classify**\nResNet50 + EfficientNetB3 ensemble prediction")
    c4.info("**4. Explain**\nGrad-CAM heatmap shows model attention regions")
