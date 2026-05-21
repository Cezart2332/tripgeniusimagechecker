"""Bake ONNX text model into models/text_onnx (Docker build). See requirements-build.txt."""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "unitary/multilingual-toxic-xlm-roberta"
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "text_onnx"

HUB_FILES = (
    "sentencepiece.bpe.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
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

    export_tokenizer_assets(OUT_DIR)
    export_onnx(OUT_DIR)

    if not (OUT_DIR / "model.onnx").exists() and not (OUT_DIR / "model_quantized.onnx").exists():
        raise RuntimeError(f"No ONNX model found in {OUT_DIR}")

    print("Text model export complete.")


if __name__ == "__main__":
    main()
