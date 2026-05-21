"""Bake ONNX text model into models/text_onnx (Docker build). See requirements-build.txt."""
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "unitary/multilingual-toxic-xlm-roberta"
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "text_onnx"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {MODEL_ID} to {OUT_DIR} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(OUT_DIR)

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
        model.save_pretrained(OUT_DIR)

        print("Applying dynamic INT8 quantization ...")
        qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
        quantizer = ORTQuantizer.from_pretrained(OUT_DIR)
        quantizer.quantize(save_dir=OUT_DIR, quantization_config=qconfig)
    except Exception as exc:
        print(f"Quantization skipped ({exc}); using full-precision ONNX export.")
        from optimum.onnxruntime import ORTModelForSequenceClassification

        model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
        model.save_pretrained(OUT_DIR)

    print("Text model export complete.")


if __name__ == "__main__":
    main()
