# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS builder

ARG TEXT_MODEL_HF_ID=oleksiizirka/xlm-roberta-toxicity-classifier
ENV TEXT_MODEL_HF_ID=${TEXT_MODEL_HF_ID}

ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    ORT_LOG_LEVEL=3 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get -y full-upgrade \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-build.txt ./
COPY scripts ./scripts
COPY moderation ./moderation

RUN python -m pip install --upgrade 'pip>=25.3' --root-user-action=ignore \
    && python -m pip install --no-cache-dir -r requirements-build.txt --root-user-action=ignore \
    && python scripts/export_models.py

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    TEXT_MODEL_DIR=/app/models/text_onnx \
    LOG_LEVEL=INFO \
    MODERATION_LOG_PREVIEW=true

# Patch Debian base packages (ncurses, zlib, util-linux, tar, etc.) before adding runtime deps.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get -y full-upgrade \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade 'pip>=25.3' --root-user-action=ignore \
    && python -m pip install --no-cache-dir 'numpy>=1.26.0,<2.0.0' --root-user-action=ignore \
    && python -m pip install --no-cache-dir -r requirements.txt --root-user-action=ignore \
    && python -m pip uninstall -y pip

COPY moderation ./moderation
COPY main.py ./
COPY --from=builder /app/models ./models

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
