# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-build.txt ./
COPY scripts ./scripts
COPY moderation ./moderation

RUN pip install --no-cache-dir -r requirements-build.txt \
    && python scripts/preload_nudenet.py \
    && python scripts/export_models.py

FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    TEXT_MODEL_DIR=/app/models/text_onnx

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY moderation ./moderation
COPY main.py ./
COPY --from=builder /app/models ./models
COPY --from=builder /root/.NudeNet /root/.NudeNet

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
