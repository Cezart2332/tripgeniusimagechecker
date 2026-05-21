from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """API tests without loading NudeNet / ONNX models."""
    with patch("main.init_image_detector"):
        from main import app

        with TestClient(app) as test_client:
            yield test_client
