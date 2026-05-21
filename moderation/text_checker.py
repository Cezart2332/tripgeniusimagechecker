from __future__ import annotations

import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from moderation.config import TEXT_MAX_CHARS, TEXT_MODEL_DIR, TOXIC_LABELS, TOXIC_THRESHOLD

_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None
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


def _resolve_onnx_path(model_dir: Path) -> Path:
    for candidate in (
        model_dir / "model.onnx",
        model_dir / "model_quantized.onnx",
        model_dir / "onnx" / "model.onnx",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ONNX model found under {model_dir}")


def _load_id2label(model_dir: Path) -> dict[int, str]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    raw = config.get("id2label", {})
    return {int(key): str(value) for key, value in raw.items()}


def _ensure_text_session() -> None:
    global _session, _tokenizer, _id2label, _load_error
    if _session is not None:
        return
    with _load_lock:
        if _session is not None:
            return
        try:
            if not TEXT_MODEL_DIR.exists():
                raise FileNotFoundError(
                    f"Text ONNX model not found at {TEXT_MODEL_DIR}. "
                    "Run scripts/export_models.py first."
                )

            tokenizer_path = TEXT_MODEL_DIR / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_path}")

            onnx_path = _resolve_onnx_path(TEXT_MODEL_DIR)
            _tokenizer = Tokenizer.from_file(str(tokenizer_path))
            _tokenizer.enable_truncation(max_length=TEXT_MAX_CHARS)
            _session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            _id2label = _load_id2label(TEXT_MODEL_DIR)
            _load_error = None
        except Exception as exc:
            _load_error = str(exc)
            raise


def _build_feeds(encoded) -> dict[str, Any]:
    assert _session is not None
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    feeds: dict[str, Any] = {}
    for model_input in _session.get_inputs():
        name = model_input.name
        if name == "input_ids":
            feeds[name] = input_ids
        elif name == "attention_mask":
            feeds[name] = attention_mask
        elif name == "token_type_ids":
            feeds[name] = token_type_ids
    return feeds


def check_text(text: str) -> tuple[bool, dict[str, float]]:
    normalized = _normalize_text(text)
    if not normalized:
        raise ValueError("Text is required")

    truncated = normalized[:TEXT_MAX_CHARS]
    _ensure_text_session()

    assert _tokenizer is not None
    assert _session is not None

    encoded = _tokenizer.encode(truncated)
    logits = _session.run(None, _build_feeds(encoded))[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    scores: dict[str, float] = {}
    for idx, prob in enumerate(probs):
        label = _id2label.get(idx, str(idx)) if _id2label else str(idx)
        scores[label] = round(float(prob), 4)

    is_toxic = any(scores.get(label, 0.0) > TOXIC_THRESHOLD for label in TOXIC_LABELS)
    return is_toxic, scores
