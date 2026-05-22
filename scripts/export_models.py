"""Bake ONNX models (text + image) into models/ at Docker build time. See requirements-build.txt."""
import os
import shutil
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("PIP_ROOT_USER_ACTION", "ignore")
os.environ.setdefault("ORT_LOG_LEVEL", "3")  # ERROR
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from huggingface_hub import hf_hub_download

# Default: single multilingual toxic logit. Multi-label override:
# TEXT_MODEL_HF_ID=oleksiizirka/xlm-roberta-toxicity-classifier
TEXT_MODEL_ID = os.getenv(
    "TEXT_MODEL_HF_ID",
    "unitary/multilingual-toxic-xlm-roberta",
)
IMAGE_MODEL_ID = os.getenv(
    "IMAGE_MODEL_HF_ID",
    "Falconsai/nsfw_image_detection",
)

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
TEXT_OUT_DIR = MODELS_ROOT / "text_onnx"
IMAGE_OUT_DIR = MODELS_ROOT / "image_onnx"

TOKENIZER_HUB_FILES = (
    "config.json",
    "sentencepiece.bpe.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

ONNX_ARTIFACTS = (
    "model.onnx",
    "model_quantized.onnx",
    "ort_config.json",
)

IMAGE_CONFIG_FILES = (
    "config.json",
    "preprocessor_config.json",
)


def _copy_onnx_artifacts(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_ARTIFACTS:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dest_dir / name)
            print(f"  Copied {name}")

    onnx_subdir = src_dir / "onnx"
    if onnx_subdir.is_dir():
        for src in onnx_subdir.glob("*.onnx"):
            shutil.copy2(src, dest_dir / src.name)
            print(f"  Copied {src.name}")


# ── Text model ────────────────────────────────────────────────────────────────


def export_tokenizer_assets(out_dir: Path) -> None:
    from transformers import XLMRobertaTokenizer

    for filename in TOKENIZER_HUB_FILES:
        downloaded = hf_hub_download(repo_id=TEXT_MODEL_ID, filename=filename)
        shutil.copy2(downloaded, out_dir / filename)
        print(f"  Copied {filename}")

    print("  Exporting tokenizer.json via slow XLMRobertaTokenizer ...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    tokenizer_json = out_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        from transformers.convert_slow_tokenizer import XLMRobertaConverter

        print("  save_pretrained did not emit tokenizer.json; converting slow tokenizer ...")
        XLMRobertaConverter(tokenizer).converted().save(str(tokenizer_json))

    if not tokenizer_json.exists():
        raise RuntimeError(f"tokenizer.json missing in {out_dir}. Cannot run text moderation.")


def export_text_onnx(out_dir: Path) -> None:
    import torch

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    from optimum.onnxruntime import ORTModelForSequenceClassification

    with tempfile.TemporaryDirectory(prefix="tg-text-onnx-") as tmp:
        work = Path(tmp)
        print(f"  Exporting text ONNX to temp dir {work} ...")
        model = ORTModelForSequenceClassification.from_pretrained(TEXT_MODEL_ID, export=True)
        model.save_pretrained(work)

        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            print("  Applying dynamic INT8 quantization (text) ...")
            qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
            quantizer = ORTQuantizer.from_pretrained(work)
            quantizer.quantize(save_dir=work, quantization_config=qconfig)
        except Exception as exc:
            print(f"  Quantization skipped ({exc}); using full-precision ONNX.")

        _copy_onnx_artifacts(work, out_dir)


# ── Image model (Falconsai ViT) ──────────────────────────────────────────────


def export_image_onnx(out_dir: Path) -> None:
    import torch

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    from optimum.onnxruntime import ORTModelForImageClassification

    with tempfile.TemporaryDirectory(prefix="tg-image-onnx-") as tmp:
        work = Path(tmp)
        print(f"  Exporting image ONNX to temp dir {work} ...")
        model = ORTModelForImageClassification.from_pretrained(IMAGE_MODEL_ID, export=True)
        model.save_pretrained(work)

        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            print("  Applying dynamic INT8 quantization (image) ...")
            qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
            quantizer = ORTQuantizer.from_pretrained(work)
            quantizer.quantize(save_dir=work, quantization_config=qconfig)
        except Exception as exc:
            print(f"  Quantization skipped ({exc}); using full-precision ONNX.")

        _copy_onnx_artifacts(work, out_dir)

    for cfg_name in IMAGE_CONFIG_FILES:
        downloaded = hf_hub_download(repo_id=IMAGE_MODEL_ID, filename=cfg_name)
        shutil.copy2(downloaded, out_dir / cfg_name)
        print(f"  Copied {cfg_name}")


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    # Text
    TEXT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[text] Exporting {TEXT_MODEL_ID} → {TEXT_OUT_DIR} ...")
    export_text_onnx(TEXT_OUT_DIR)
    export_tokenizer_assets(TEXT_OUT_DIR)
    text_onnx = TEXT_OUT_DIR / "model.onnx"
    text_quantized = TEXT_OUT_DIR / "model_quantized.onnx"
    if not text_onnx.exists() and not text_quantized.exists():
        raise RuntimeError(f"No text ONNX model found in {TEXT_OUT_DIR}")
    print("[text] Export complete.\n")

    # Image
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[image] Exporting {IMAGE_MODEL_ID} → {IMAGE_OUT_DIR} ...")
    export_image_onnx(IMAGE_OUT_DIR)
    image_onnx = IMAGE_OUT_DIR / "model.onnx"
    image_quantized = IMAGE_OUT_DIR / "model_quantized.onnx"
    if not image_onnx.exists() and not image_quantized.exists():
        raise RuntimeError(f"No image ONNX model found in {IMAGE_OUT_DIR}")
    print("[image] Export complete.")


if __name__ == "__main__":
    main()
