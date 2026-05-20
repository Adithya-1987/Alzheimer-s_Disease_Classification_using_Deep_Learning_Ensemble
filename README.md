---
title: Alzheimer MRI Classifier API
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Alzheimer's MRI Classifier API

FastAPI backend for Alzheimer's disease classification using ensemble deep learning (ResNet50 + EfficientNetB3).

## 🚀 Endpoints

- **GET /health** — Health check endpoint
- **POST /predict** — Classify MRI scan

## 🏗️ Architecture

This API uses an ensemble model combining:
- **ResNet50** (pretrained on ImageNet)
- **EfficientNetB3** (pretrained on ImageNet)

The model classifies brain MRI scans into 4 categories:
- NonDemented
- VeryMildDemented
- MildDemented
- ModerateDemented

## 🔧 Preprocessing Pipeline

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
2. **Gaussian Blur** for noise reduction
3. **Normalization** to [0, 1] range

## 📊 Model Details

- **Input Size**: 224×224×3
- **Total Parameters**: ~30M
- **Training Dataset**: OASIS Augmented Alzheimer MRI Dataset
- **Accuracy**: ~95% on test set

## 🔐 Authentication

The API requires:
- Supabase JWT token in `Authorization` header
- Valid Supabase project configuration

## 🌐 Environment Variables

Required environment variables:
- `SUPABASE_URL` — Your Supabase project URL
- `SUPABASE_SERVICE_KEY` — Supabase service role key
- `ALLOWED_ORIGINS` — Comma-separated list of allowed CORS origins (optional)
- `MODEL_PATH` — Path to the model file (optional, defaults to `saved_models/final_model.keras`)

## 📝 Usage Example

```bash
curl -X POST "https://your-space.hf.space/predict" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "scan_id": "uuid-here",
    "image_path": "user_id/filename.jpg"
  }'
```

## ⚠️ Disclaimer

This system is for **research and educational purposes only**. It is **not a medical device** and should not be used for clinical diagnosis. Always consult a qualified medical professional.

## 📄 License

MIT License — free to use for research and education.
