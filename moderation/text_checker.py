from __future__ import annotations

import re
import threading
import unicodedata
from typing import Any

import numpy as np

from moderation.config import TEXT_MAX_CHARS, TEXT_MODEL_DIR, TOXIC_LABELS, TOXIC_THRESHOLD

_session: Any | None = None
_tokenizer: Any | None = None
_id2label: dict[int, str] | None = None
_load_lock = threading.Lock()
_load_error: str | None = None

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


def is_text_ready() -> bool:
    return _session is not None


def text_load_error() -> str | None:
    return _load_error


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return text.strip()


def _ensure_text_session() -> None:
    global _session, _tokenizer, _id2label, _load_error
    if _session is not None:
        return
    with _load_lock:
        if _session is not None:
            return
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer

            if not TEXT_MODEL_DIR.exists():
                raise FileNotFoundError(
                    f"Text ONNX model not found at {TEXT_MODEL_DIR}. "
                    "Run scripts/export_models.py first."
                )

            _tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
            _session = ORTModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
            _id2label = dict(_session.config.id2label)
            _load_error = None
        except Exception as exc:
            _load_error = str(exc)
            raise


def check_text(text: str) -> tuple[bool, dict[str, float]]:
    normalized = _normalize_text(text)
    if not normalized:
        raise ValueError("Text is required")

    truncated = normalized[:TEXT_MAX_CHARS]
    _ensure_text_session()

    inputs = _tokenizer(
        truncated,
        return_tensors="np",
        truncation=True,
        max_length=TEXT_MAX_CHARS,
    )
    outputs = _session(**inputs)
    logits = outputs.logits[0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    scores: dict[str, float] = {}
    for idx, prob in enumerate(probs):
        label = _id2label.get(idx, str(idx)) if _id2label else str(idx)
        scores[label] = round(float(prob), 4)

    is_toxic = any(scores.get(label, 0.0) > TOXIC_THRESHOLD for label in TOXIC_LABELS)
    return is_toxic, scores
