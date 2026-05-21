from __future__ import annotations

import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from moderation.config import (
    SHORT_TEXT_LEN,
    SHORT_TEXT_TOXIC_THRESHOLD,
    TEXT_MAX_CHARS,
    TEXT_MODEL_DIR,
    TOXIC_LABELS,
    TOXIC_THRESHOLD,
)

_sp_processor: spm.SentencePieceProcessor | None = None
_token_config: dict[str, Any] | None = None
_session: ort.InferenceSession | None = None
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


def _load_tokenizer_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.exists():
        return {"bos_token_id": 0, "pad_token_id": 1, "eos_token_id": 2}
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _encode_text(text: str, max_length: int) -> tuple[list[int], list[int]]:
    assert _sp_processor is not None
    assert _token_config is not None

    bos_id = int(_token_config.get("bos_token_id", 0))
    pad_id = int(_token_config.get("pad_token_id", 1))
    eos_id = int(_token_config.get("eos_token_id", 2))

    body_ids = _sp_processor.encode(text, out_type=int)
    max_body = max(1, max_length - 2)
    body_ids = body_ids[:max_body]

    input_ids = [bos_id, *body_ids, eos_id]
    attention_mask = [1] * len(input_ids)

    while len(input_ids) < max_length:
        input_ids.append(pad_id)
        attention_mask.append(0)

    return input_ids, attention_mask


def _ensure_text_session() -> None:
    global _sp_processor, _token_config, _session, _id2label, _load_error
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

            spm_path = TEXT_MODEL_DIR / "sentencepiece.bpe.model"
            if not spm_path.exists():
                raise FileNotFoundError(f"sentencepiece.bpe.model not found at {spm_path}")

            processor = spm.SentencePieceProcessor()
            processor.load(str(spm_path))

            onnx_path = _resolve_onnx_path(TEXT_MODEL_DIR)
            _sp_processor = processor
            _token_config = _load_tokenizer_config(TEXT_MODEL_DIR)
            _session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            _id2label = _load_id2label(TEXT_MODEL_DIR)
            _load_error = None
        except Exception as exc:
            _load_error = str(exc)
            raise


def _scores_from_logits(logits: np.ndarray, id2label: dict[int, str]) -> dict[str, float]:
    """Map model outputs to label probabilities.

    unitary/multilingual-toxic-xlm-roberta exports a single toxic logit (sigmoid).
    Softmax on one logit is always 1.0 and incorrectly flags every message.
    """
    values = np.asarray(logits, dtype=np.float64).flatten()

    if values.size == 0:
        return {}

    if values.size == 1:
        label = id2label.get(0, "toxic")
        prob = float(1.0 / (1.0 + np.exp(-values[0])))
        return {label: round(prob, 4)}

    # Multi-label heads (e.g. Detoxify): independent sigmoid per logit.
    if len(id2label) > 1 and values.size == len(id2label):
        probs = 1.0 / (1.0 + np.exp(-values))
        return {
            id2label.get(i, str(i)): round(float(probs[i]), 4) for i in range(values.size)
        }

    # Mutually exclusive multi-class fallback.
    exp = np.exp(values - np.max(values))
    probs = exp / exp.sum()
    return {
        id2label.get(i, str(i)): round(float(probs[i]), 4) for i in range(values.size)
    }


def _build_feeds(input_ids: list[int], attention_mask: list[int]) -> dict[str, Any]:
    assert _session is not None
    ids = np.array([input_ids], dtype=np.int64)
    mask = np.array([attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(ids)

    feeds: dict[str, Any] = {}
    for model_input in _session.get_inputs():
        name = model_input.name
        if name == "input_ids":
            feeds[name] = ids
        elif name == "attention_mask":
            feeds[name] = mask
        elif name == "token_type_ids":
            feeds[name] = token_type_ids
    return feeds


def check_text(text: str) -> tuple[bool, dict[str, float]]:
    normalized = _normalize_text(text)
    if not normalized:
        raise ValueError("Text is required")

    truncated = normalized[:TEXT_MAX_CHARS]
    _ensure_text_session()

    assert _session is not None

    input_ids, attention_mask = _encode_text(truncated, TEXT_MAX_CHARS)
    logits = _session.run(None, _build_feeds(input_ids, attention_mask))[0][0]
    label_map = _id2label if _id2label else {0: "toxic"}
    scores = _scores_from_logits(logits, label_map)

    threshold = (
        SHORT_TEXT_TOXIC_THRESHOLD
        if len(truncated) < SHORT_TEXT_LEN
        else TOXIC_THRESHOLD
    )
    is_toxic = any(scores.get(label, 0.0) > threshold for label in TOXIC_LABELS)
    return is_toxic, scores
