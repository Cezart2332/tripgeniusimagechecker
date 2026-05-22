import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


@pytest.fixture
def client():
    """API tests without loading NudeNet / ONNX models."""
    with patch.object(main, "init_image_detector"), patch.object(main, "init_text_model"):
        with TestClient(main.app) as test_client:
            yield test_client
