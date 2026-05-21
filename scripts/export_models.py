"""Bake ONNX text model into models/text_onnx (Docker build). See requirements-build.txt."""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_ID = "unitary/multilingual-toxic-xlm-roberta"
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "text_onnx"

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


def export_tokenizer(out_dir: Path) -> None:
    for filename in TOKENIZER_FILES:
        try:
            downloaded = hf_hub_download(repo_id=MODEL_ID, filename=filename)
            shutil.copy2(downloaded, out_dir / filename)
            print(f"Copied {filename}")
        except Exception as exc:
            print(f"Could not download {filename}: {exc}")

    if not (out_dir / "tokenizer.json").exists():
        print("Falling back to slow tokenizer (use_fast=False) ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
        tokenizer.save_pretrained(out_dir)

    if not (out_dir / "tokenizer.json").exists():
        raise RuntimeError(
            f"tokenizer.json missing in {out_dir}. Cannot run text moderation at runtime."
        )


def export_onnx(out_dir: Path) -> None:
    from optimum.onnxruntime import ORTModelForSequenceClassification

    model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
    model.save_pretrained(out_dir)

    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        print("Applying dynamic INT8 quantization ...")
        qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
        quantizer = ORTQuantizer.from_pretrained(out_dir)
        quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)
    except Exception as exc:
        print(f"Quantization skipped ({exc}); using full-precision ONNX.")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {MODEL_ID} to {OUT_DIR} ...")

    export_tokenizer(OUT_DIR)
    export_onnx(OUT_DIR)

    onnx_path = OUT_DIR / "model.onnx"
    if not onnx_path.exists() and not (OUT_DIR / "model_quantized.onnx").exists():
        raise RuntimeError(f"No ONNX model found in {OUT_DIR}")

    print("Text model export complete.")


if __name__ == "__main__":
    main()
