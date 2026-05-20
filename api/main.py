"""
api/main.py
===========
FastAPI backend for Alzheimer's MRI Classification.

Endpoints:
  GET  /health   — health check
  POST /predict  — receives { scan_id, image_path } + Supabase JWT,
                   downloads image from Supabase Storage, runs inference,
                   writes result back to scans table, returns prediction JSON
"""

import os
import sys
import io
import numpy as np
import cv2
import httpx
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocess import preprocess_image

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
MODEL_PATH   = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "saved_models", "final_model.keras")
)
IMG_SIZE     = (224, 224)

SUPABASE_URL      = os.getenv("SUPABASE_URL")        # e.g. https://xxxx.supabase.co
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # service role key (never expose to browser)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Alzheimer's MRI Classifier API")

# FIX 1: Environment-based CORS origins
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8080,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model (loaded once at startup) ───────────────────────────────────────────
model = None

@app.on_event("startup")
def load_model():
    global model
    # FIX 2: Fail fast if model file is missing
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Set MODEL_PATH env var or ensure saved_models/final_model.keras exists."
        )
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")


# ─── Request schema ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    scan_id: str      # UUID of the row in scans table
    image_path: str   # Storage path e.g. "{user_id}/1713456789_brain.jpg"


# ─── Helper: verify Supabase JWT and return user_id ───────────────────────────
async def verify_jwt(token: str) -> str:
    """
    Calls Supabase /auth/v1/user with the user's JWT.
    Returns the user's UUID if valid, raises 401 if not.
    """
    # FIX 3: Add HTTP timeout
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_SERVICE_KEY,
            }
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return resp.json()["id"]  # user UUID


# ─── Helper: get signed URL for private storage object ───────────────────────
async def get_signed_url(image_path: str) -> str:
    """
    Creates a 60-second signed URL for a private mri-scans bucket object.
    Uses the service role key so it can access any user's file.
    """
    # FIX 3: Add HTTP timeout
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{SUPABASE_URL}/storage/v1/object/sign/mri-scans/{image_path}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "apikey": SUPABASE_SERVICE_KEY,
            },
            json={"expiresIn": 60}
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Could not create signed URL: {resp.text}")
    signed_url = resp.json().get("signedURL") or resp.json().get("signedUrl")
    return f"{SUPABASE_URL}/storage/v1{signed_url}"


# ─── Helper: download image bytes from signed URL ─────────────────────────────
async def download_image(signed_url: str) -> np.ndarray:
    # FIX 3: Add HTTP timeout
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(signed_url)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Could not download image from storage.")
    file_bytes = np.frombuffer(resp.content, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a valid JPG or PNG.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ─── Helper: write prediction result back to Supabase DB ─────────────────────
async def save_result_to_db(scan_id: str, prediction: str, confidence: float, probabilities: dict):
    import json
    # FIX 3: Add HTTP timeout
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.patch(
            f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "apikey": SUPABASE_SERVICE_KEY,
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "status": "done",
            }
        )
    if resp.status_code not in (200, 204):
        print(f"[WARNING] DB write failed for scan {scan_id}: {resp.text}")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(
    body: PredictRequest,
    authorization: str = Header(...),  # expects "Bearer <jwt>"
):
    # 1. Guard: model must be loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place final_model.keras in saved_models/.")

    # 2. Verify JWT — ensures only authenticated users can run inference
    token = authorization.replace("Bearer ", "").strip()
    user_id = await verify_jwt(token)

    # 3. Get a short-lived signed URL for the private image
    signed_url = await get_signed_url(body.image_path)

    # 4. Download and decode the image
    img_rgb = await download_image(signed_url)

    # 5. Preprocess — identical pipeline to training (CLAHE → blur → normalize)
    preprocessed = preprocess_image(img_rgb, target_size=IMG_SIZE)
    img_array = np.expand_dims(preprocessed, axis=0)  # (1, 224, 224, 3)

    # 6. Run inference
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx    = int(np.argmax(preds))
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = float(preds[pred_idx])
    probabilities = {name: round(float(preds[i]), 4) for i, name in enumerate(CLASS_NAMES)}

    # 7. Write result back to DB (async, non-blocking to user)
    await save_result_to_db(body.scan_id, pred_class, confidence, probabilities)

    # 8. Return result to frontend
    return {
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "scan_id": body.scan_id,
    }