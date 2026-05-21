"""Bake ONNX text model into models/text_onnx (Docker build). See requirements-build.txt."""
import os
import shutil
import tempfile
import warnings
from pathlib import Path

# Quiet known-noisy build-time logs (Docker/CI has no GPU; runtime uses SentencePiece, not HF tokenizer).
os.environ.setdefault("PIP_ROOT_USER_ACTION", "ignore")
os.environ.setdefault("ORT_LOG_LEVEL", "3")  # ERROR
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from huggingface_hub import hf_hub_download

MODEL_ID = "unitary/multilingual-toxic-xlm-roberta"
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "text_onnx"

HUB_FILES = (
    "sentencepiece.bpe.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)

ONNX_ARTIFACTS = (
    "model.onnx",
    "model_quantized.onnx",
    "ort_config.json",
)


def export_tokenizer_assets(out_dir: Path) -> None:
    for filename in HUB_FILES:
        downloaded = hf_hub_download(repo_id=MODEL_ID, filename=filename)
        shutil.copy2(downloaded, out_dir / filename)
        print(f"Copied {filename}")

    if not (out_dir / "sentencepiece.bpe.model").exists():
        raise RuntimeError(
            f"sentencepiece.bpe.model missing in {out_dir}. Cannot run text moderation."
        )


def _copy_onnx_artifacts(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_ARTIFACTS:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dest_dir / name)
            print(f"Copied {name}")

    onnx_subdir = src_dir / "onnx"
    if onnx_subdir.is_dir():
        for src in onnx_subdir.glob("*.onnx"):
            shutil.copy2(src, dest_dir / src.name)
            print(f"Copied {src.name}")


def export_onnx(out_dir: Path) -> None:
    import torch

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    from optimum.onnxruntime import ORTModelForSequenceClassification

    # Export in an isolated dir so optimum/transformers never load a half-written tokenizer tree.
    with tempfile.TemporaryDirectory(prefix="tg-text-onnx-") as tmp:
        work = Path(tmp)
        print(f"Exporting ONNX to temp dir {work} ...")
        model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
        model.save_pretrained(work)

        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            print("Applying dynamic INT8 quantization ...")
            qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
            quantizer = ORTQuantizer.from_pretrained(work)
            quantizer.quantize(save_dir=work, quantization_config=qconfig)
        except Exception as exc:
            print(f"Quantization skipped ({exc}); using full-precision ONNX.")

        _copy_onnx_artifacts(work, out_dir)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {MODEL_ID} to {OUT_DIR} ...")

    export_onnx(OUT_DIR)
    export_tokenizer_assets(OUT_DIR)

    if not (OUT_DIR / "model.onnx").exists() and not (OUT_DIR / "model_quantized.onnx").exists():
        raise RuntimeError(f"No ONNX model found in {OUT_DIR}")

    print("Text model export complete.")


if __name__ == "__main__":
    main()
