"""Local dev helper: preload NudeNet + export text ONNX. Docker build uses scripts/ directly."""
from nudenet import NudeDetector

from scripts.export_models import main as export_text

if __name__ == "__main__":
    print("Preloading NudeNet ...")
    NudeDetector()
    export_text()
    print("All models ready.")
