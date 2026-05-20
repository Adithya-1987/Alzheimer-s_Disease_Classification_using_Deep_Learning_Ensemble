# 🚀 Hugging Face Spaces Deployment Guide

## ✅ All Fixes Applied

### FIX 1 — Environment-based CORS origins
✓ Replaced hardcoded origins with `ALLOWED_ORIGINS` environment variable

### FIX 2 — Configurable model path with fail-fast
✓ Model path now reads from `MODEL_PATH` env var
✓ Startup fails immediately if model file is missing

### FIX 3 — HTTP timeouts on all external requests
✓ Added `timeout=30.0` to all 4 httpx.AsyncClient() calls

### FIX 4 — Dockerfile for Hugging Face Spaces
✓ Created Dockerfile with port 7860 (HF Spaces requirement)
✓ Includes OpenCV system dependencies

### FIX 5 — Production-only requirements.txt
✓ Removed training-only dependencies (matplotlib, seaborn, streamlit, pandas, scikit-learn, tqdm)
✓ Kept only runtime essentials

### FIX 6 — Git LFS configuration
✓ Created .gitattributes for *.keras and *.h5 files

### FIX 7 — .gitignore
✓ Created .gitignore (excludes .env, __pycache__, dataset, logs)
✓ Model files are NOT ignored (needed for deployment)

### FIX 8 — Hugging Face README with metadata
✓ Created README.md with required HF Spaces YAML header

---

## 📦 Deployment Commands

### 1. Install Git LFS (if not already installed)
```bash
# macOS
brew install git-lfs

# Or download from: https://git-lfs.github.com/
```

### 2. Initialize Git LFS in your project
```bash
cd "/Users/adithya/Downloads/ML MODEL"
git lfs install
```

### 3. Initialize Git repository (if not already done)
```bash
git init
```

### 4. Track large model file with Git LFS
```bash
git lfs track "*.keras"
git lfs track "*.h5"
```

### 5. Add all files to Git
```bash
git add .gitattributes
git add .
```

### 6. Commit everything
```bash
git commit -m "Initial commit: FastAPI Alzheimer MRI classifier for HF Spaces"
```

### 7. Create a new Space on Hugging Face
1. Go to https://huggingface.co/new-space
2. Choose a name (e.g., `alzheimer-mri-api`)
3. Select **Docker** as the SDK
4. Set visibility (Public or Private)
5. Click "Create Space"

### 8. Add Hugging Face as remote and push
```bash
# Replace YOUR_USERNAME and YOUR_SPACE_NAME
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push to Hugging Face
git push hf main
```

**Note:** If your default branch is `master` instead of `main`, use:
```bash
git push hf master:main
```

---

## 🔐 Environment Variables to Set in HF Spaces

After pushing, go to your Space's **Settings → Variables and secrets** and add:

### Required Variables:
| Variable Name | Description | Example |
|---|---|---|
| `SUPABASE_URL` | Your Supabase project URL | `https://xxxxx.supabase.co` |
| `SUPABASE_SERVICE_KEY` | Supabase service role key | `eyJhbGc...` (long JWT) |

### Optional Variables:
| Variable Name | Description | Default |
|---|---|---|
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | `http://localhost:8080,http://localhost:5173` |
| `MODEL_PATH` | Custom model file path | `saved_models/final_model.keras` |

**Important:** Mark `SUPABASE_SERVICE_KEY` as a **Secret** (not a regular variable) to keep it hidden.

---

## 🧪 Testing Your Deployment

### 1. Check health endpoint
```bash
curl https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 2. Test prediction endpoint
```bash
curl -X POST "https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/predict" \
  -H "Authorization: Bearer YOUR_SUPABASE_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "scan_id": "your-scan-uuid",
    "image_path": "user_id/filename.jpg"
  }'
```

---

## 📊 Model File Size Note

Your `final_model.keras` file is **333MB**. Git LFS will handle this automatically:
- Regular Git has a 100MB file size limit
- Git LFS stores large files separately and uses pointers in the repo
- Hugging Face Spaces fully supports Git LFS

---

## 🔧 Troubleshooting

### Build fails with "Model file not found"
- Ensure `saved_models/final_model.keras` exists in your repo
- Check that `.gitattributes` is committed
- Verify Git LFS tracked the file: `git lfs ls-files`

### CORS errors from frontend
- Add your frontend URL to `ALLOWED_ORIGINS` in HF Spaces settings
- Format: `https://your-frontend.com,https://another-domain.com`

### Timeout errors
- All HTTP requests now have 30-second timeouts
- If Supabase is slow, consider increasing timeout in code

### Port issues
- HF Spaces REQUIRES port 7860 (already configured in Dockerfile)
- Do not change this port

---

## 📝 Next Steps

1. Update your frontend to point to the HF Spaces URL
2. Test with real MRI scans
3. Monitor logs in HF Spaces dashboard
4. Consider adding rate limiting for production use

---

## 🎉 You're Done!

Your FastAPI backend is now deployed on Hugging Face Spaces with:
- ✅ Docker containerization
- ✅ Environment-based configuration
- ✅ Git LFS for large model files
- ✅ Production-ready dependencies
- ✅ Proper error handling and timeouts
- ✅ Free hosting on HF Spaces

