from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from moderation.config import IMAGE_MAX_EDGE, NSFW_LABELS, NSFW_THRESHOLD

_detector: Any | None = None


def init_image_detector() -> None:
    global _detector
    if _detector is not None:
        return
    from nudenet import NudeDetector

    _detector = NudeDetector()


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


def check_image_bytes(data: bytes) -> tuple[bool, float]:
    if _detector is None:
        raise RuntimeError("Image detector not initialized")

    image = Image.open(io.BytesIO(data))
    image = _resize_for_inference(image)
    detections = _detector.detect(np.array(image))

    nsfw_score = 0.0
    for item in detections:
        label = str(item.get("class") or item.get("label") or "")
        score = float(item.get("score", 0.0))
        if label in NSFW_LABELS and score > nsfw_score:
            nsfw_score = score

    return nsfw_score >= NSFW_THRESHOLD, round(nsfw_score, 4)
