"""Image NSFW classification via Falconsai/nsfw_image_detection (ONNX)."""
from __future__ import annotations

import io
import json
import logging
from typing import Any

import numpy as np
from PIL import Image

from moderation.config import IMAGE_MODEL_DIR, NSFW_THRESHOLD

logger = logging.getLogger(__name__)

_session: Any | None = None
_label_map: dict[int, str] = {}
_image_size: int = 224
_image_mean: list[float] = [0.485, 0.456, 0.406]
_image_std: list[float] = [0.229, 0.224, 0.225]

_NSFW_LABEL_ID: int | None = None


def init_image_detector() -> None:
    global _session, _label_map, _image_size, _image_mean, _image_std, _NSFW_LABEL_ID

    if _session is not None:
        return

    import onnxruntime as ort

    model_dir = IMAGE_MODEL_DIR
    onnx_path = model_dir / "model_quantized.onnx"
    if not onnx_path.exists():
        onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"No ONNX image model in {model_dir}")

    logger.info("Loading image ONNX model from %s", onnx_path)
    _session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        id2label = config.get("id2label", {})
        _label_map = {int(k): v for k, v in id2label.items()}
    if not _label_map:
        _label_map = {0: "normal", 1: "nsfw"}

    _NSFW_LABEL_ID = None
    for idx, label in _label_map.items():
        if label.lower() == "nsfw":
            _NSFW_LABEL_ID = idx
            break
    if _NSFW_LABEL_ID is None:
        _NSFW_LABEL_ID = 1
        logger.warning("Could not find 'nsfw' in id2label; defaulting to index %d", _NSFW_LABEL_ID)

    preprocess_path = model_dir / "preprocessor_config.json"
    if preprocess_path.exists():
        with open(preprocess_path) as f:
            pp = json.load(f)
        if "size" in pp:
            size_val = pp["size"]
            if isinstance(size_val, dict):
                _image_size = size_val.get("height", size_val.get("shortest_edge", 224))
            elif isinstance(size_val, int):
                _image_size = size_val
        if "image_mean" in pp:
            _image_mean = pp["image_mean"]
        if "image_std" in pp:
            _image_std = pp["image_std"]

    logger.info(
        "Image model ready: labels=%s, nsfw_id=%d, size=%d",
        _label_map,
        _NSFW_LABEL_ID,
        _image_size,
    )


def is_image_ready() -> bool:
    return _session is not None


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((_image_size, _image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(_image_mean, dtype=np.float32)
    std = np.array(_image_std, dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return np.expand_dims(arr, axis=0)  # add batch dim


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def check_image_bytes(data: bytes) -> tuple[bool, float]:
    """Classify image bytes. Returns (is_nsfw, nsfw_probability)."""
    if _session is None:
        raise RuntimeError("Image detector not initialized")

    image = Image.open(io.BytesIO(data))
    pixel_values = _preprocess(image)

    input_name = _session.get_inputs()[0].name
    (logits,) = _session.run(None, {input_name: pixel_values})

    probs = _softmax(logits[0])
    nsfw_prob = float(probs[_NSFW_LABEL_ID])

    logger.info(
        "Falconsai classification: %s (nsfw=%.4f, threshold=%.2f)",
        {_label_map.get(i, str(i)): round(float(p), 4) for i, p in enumerate(probs)},
        nsfw_prob,
        NSFW_THRESHOLD,
    )

    return nsfw_prob >= NSFW_THRESHOLD, round(nsfw_prob, 4)
