from __future__ import annotations

import io
import logging
import os
from typing import Any

import numpy as np
from PIL import Image

from moderation.config import IMAGE_MAX_EDGE, NSFW_LABELS, NSFW_THRESHOLD

logger = logging.getLogger(__name__)

_detector: Any | None = None

# Labels returned by nudenet 3.x (see nudenet.nudenet.__labels).
_NUDENET_EXPOSED_LABELS = frozenset(
    {
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED",
    }
)


def init_image_detector() -> None:
    global _detector
    if _detector is not None:
        return
    from nudenet import NudeDetector

    model_path = os.getenv("NUDENET_MODEL_PATH")
    inference_size = int(os.getenv("NUDENET_INFERENCE_SIZE", "320"))

    if model_path and os.path.isfile(model_path):
        logger.info("Loading NudeNet from %s (inference %spx)", model_path, inference_size)
        _detector = NudeDetector(model_path=model_path, inference_resolution=inference_size)
    else:
        if model_path:
            logger.warning("NUDENET_MODEL_PATH=%s not found; using bundled 320n.onnx", model_path)
        _detector = NudeDetector(inference_resolution=inference_size)


def is_image_ready() -> bool:
    return _detector is not None


def _resize_for_inference(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    longest = max(w, h)
    if longest <= IMAGE_MAX_EDGE:
        return image
    scale = IMAGE_MAX_EDGE / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _is_nsfw_label(label: str) -> bool:
    if label in NSFW_LABELS or label in _NUDENET_EXPOSED_LABELS:
        return True
    return "EXPOSED" in label.upper() or label.upper().startswith("EXPOSED_")


def check_image_bytes(data: bytes) -> tuple[bool, float]:
    if _detector is None:
        raise RuntimeError("Image detector not initialized")

    image = Image.open(io.BytesIO(data))
    image = _resize_for_inference(image)

    # NudeNet expects file bytes / BGR paths — not a bare RGB ndarray (wrong cv2 color path).
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    detections = _detector.detect(buffer.getvalue())

    nsfw_score = 0.0
    hits: list[str] = []
    for item in detections:
        label = str(item.get("class") or item.get("label") or "")
        score = float(item.get("score", 0.0))
        if _is_nsfw_label(label) and score > nsfw_score:
            nsfw_score = score
            hits.append(f"{label}={score:.3f}")

    if hits:
        logger.info("NudeNet detections (nsfw): %s", ", ".join(hits))
    elif detections:
        logger.debug(
            "NudeNet detections below nsfw filter: %s",
            ", ".join(
                f"{item.get('class')}={float(item.get('score', 0)):.3f}" for item in detections
            ),
        )

    return nsfw_score >= NSFW_THRESHOLD, round(nsfw_score, 4)
