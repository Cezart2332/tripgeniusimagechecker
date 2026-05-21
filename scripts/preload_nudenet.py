"""Pre-download NudeNet ONNX weights at Docker build."""
from nudenet import NudeDetector

print("Preloading NudeNet detector ...")
NudeDetector()
print("NudeNet ready.")
