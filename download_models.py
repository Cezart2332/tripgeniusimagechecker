"""Local dev helper: export both text + image ONNX models. Docker build uses scripts/export_models.py."""
from scripts.export_models import main as export_all

if __name__ == "__main__":
    export_all()
    print("All models ready.")
