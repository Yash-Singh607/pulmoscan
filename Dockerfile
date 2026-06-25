# Lightweight CPU image for serving the inference API.
# For GPU training, base off an nvidia/cuda or pytorch/pytorch image instead.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CXR_CHECKPOINT_PATH=/app/checkpoints/best_model.pth

# OpenCV runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first to keep the image small, then the package.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch>=2.0.0 torchvision>=0.15.0

COPY pyproject.toml README.md ./
COPY chestxray ./chestxray
RUN pip install ".[serve]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "chestxray.api:app", "--host", "0.0.0.0", "--port", "8000"]
