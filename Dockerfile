FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    httpx>=0.27.0 \
    python-dotenv>=1.0.0 \
    python-multipart>=0.0.6 \
    tensorflow>=2.16.0 \
    keras>=3.0.0 \
    opencv-python-headless>=4.7.0 \
    numpy>=1.23.0 \
    Pillow>=9.5.0

# Copy the entire project
COPY . .

# Expose Hugging Face Spaces required port
EXPOSE 7860

# Run the FastAPI app on port 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
