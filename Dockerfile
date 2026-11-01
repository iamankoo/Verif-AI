# Verif-AI - single image: React UI (built) served by the FastAPI backend. CPU inference, no external AI APIs.
#   docker build -t verifai .
#   docker run --rm -p 8000:8000 -v verifai-data:/app/storage verifai        ->  http://localhost:8000
# Model checkpoints are expected in backend/models/ (and/or backend/scripts/two_stream_model_epoch_10.pth).

# ---------- 1. build the frontend ----------
FROM node:22-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- 2. backend runtime ----------
FROM python:3.10-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VERIFAI_FRONTEND_DIR=/app/frontend_dist \
    VERIFAI_STORAGE_DIR=/app/storage \
    VERIFAI_MODEL_PATH=/app/models/verifai_v2_best.pth
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 verifai
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
# CPU-only PyTorch keeps the image small; use a CUDA base image + CUDA wheels for GPU deployments.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.0" "torchvision>=0.15" \
    && pip install -r requirements.txt
COPY backend/verifai ./verifai
COPY backend/models ./models
COPY backend/scripts/two_stream_model_epoch_10.pth ./scripts/two_stream_model_epoch_10.pth
COPY --from=frontend /fe/dist ./frontend_dist
RUN mkdir -p /app/storage && chown -R verifai:verifai /app
USER verifai
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health | grep -q '"model_loaded":true' || exit 1
CMD ["uvicorn", "verifai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
