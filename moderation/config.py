import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# NudeNet box scores are typically 0.3–0.8; 0.85 blocked almost nothing in practice.
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.45"))
TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.72"))
SHORT_TEXT_LEN = int(os.getenv("SHORT_TEXT_LEN", "16"))
SHORT_TEXT_TOXIC_THRESHOLD = float(os.getenv("SHORT_TEXT_TOXIC_THRESHOLD", "0.88"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
TEXT_MAX_CHARS = int(os.getenv("TEXT_MAX_CHARS", "256"))
TEXT_MODEL_DIR = Path(os.getenv("TEXT_MODEL_DIR", str(BASE_DIR / "models" / "text_onnx")))
IMAGE_MAX_EDGE = int(os.getenv("IMAGE_MAX_EDGE", "384"))
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "10"))

# Block only core toxicity; insult/obscene fire often on benign short multilingual text.
TOXIC_LABELS = ("toxic", "severe_toxic")

# NudeNet / detector labels treated as NSFW (any match above threshold).
NSFW_LABELS = frozenset(
    {
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "ANUS_EXPOSED",
        "EXPOSED_GENITALIA_F",
        "EXPOSED_GENITALIA_M",
        "EXPOSED_BUTTOCKS",
        "EXPOSED_BREAST_F",
        "EXPOSED_BREAST_M",
        "EXPOSED_ANUS",
        "EXPOSED_GENITALIA",
    }
)
