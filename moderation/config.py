import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Falconsai softmax probabilities (0.0–1.0); 0.5 is the natural decision boundary.
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.5"))
TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.72"))
SHORT_TEXT_LEN = int(os.getenv("SHORT_TEXT_LEN", "16"))
SHORT_TEXT_TOXIC_THRESHOLD = float(os.getenv("SHORT_TEXT_TOXIC_THRESHOLD", "0.88"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
TEXT_MAX_CHARS = int(os.getenv("TEXT_MAX_CHARS", "256"))
TEXT_MAX_TOKENS = int(os.getenv("TEXT_MAX_TOKENS", "256"))
TEXT_MODEL_DIR = Path(os.getenv("TEXT_MODEL_DIR", str(BASE_DIR / "models" / "text_onnx")))
IMAGE_MODEL_DIR = Path(os.getenv("IMAGE_MODEL_DIR", str(BASE_DIR / "models" / "image_onnx")))
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "10"))

# Labels that can trigger a block (any score above threshold). "none" is never included.
_DEFAULT_TOXIC_LABELS = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
_toxic_labels_env = os.getenv("TOXIC_LABELS", "").strip()
if _toxic_labels_env:
    TOXIC_LABELS = tuple(
        label.strip()
        for label in _toxic_labels_env.split(",")
        if label.strip() and label.strip().lower() != "none"
    )
else:
    TOXIC_LABELS = _DEFAULT_TOXIC_LABELS
