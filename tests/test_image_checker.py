import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from moderation.image_checker import check_image_bytes, init_image_detector


def _jpeg_bytes_for_test() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _make_mock_session(logits: list[float]) -> MagicMock:
    session = MagicMock()
    inp = MagicMock()
    inp.name = "pixel_values"
    session.get_inputs.return_value = [inp]
    session.run.return_value = [np.array([logits], dtype=np.float32)]
    return session


@patch("moderation.image_checker._session")
@patch("moderation.image_checker._label_map", {0: "normal", 1: "nsfw"})
@patch("moderation.image_checker._NSFW_LABEL_ID", 1)
def test_check_image_bytes_blocks_nsfw(mock_session: MagicMock):
    mock_session.__bool__ = lambda _: True
    mock_session.get_inputs.return_value = [MagicMock(name="pixel_values")]
    mock_session.get_inputs.return_value[0].name = "pixel_values"
    mock_session.run.return_value = [np.array([[-2.0, 3.0]], dtype=np.float32)]

    is_nsfw, score = check_image_bytes(_jpeg_bytes_for_test())

    assert is_nsfw is True
    assert score > 0.9
    mock_session.run.assert_called_once()


@patch("moderation.image_checker._session")
@patch("moderation.image_checker._label_map", {0: "normal", 1: "nsfw"})
@patch("moderation.image_checker._NSFW_LABEL_ID", 1)
def test_check_image_bytes_allows_normal(mock_session: MagicMock):
    mock_session.__bool__ = lambda _: True
    mock_session.get_inputs.return_value = [MagicMock(name="pixel_values")]
    mock_session.get_inputs.return_value[0].name = "pixel_values"
    mock_session.run.return_value = [np.array([[3.0, -2.0]], dtype=np.float32)]

    is_nsfw, score = check_image_bytes(_jpeg_bytes_for_test())

    assert is_nsfw is False
    assert score < 0.1


@patch("moderation.image_checker._session", None)
def test_check_image_bytes_raises_when_not_initialized():
    import pytest

    with pytest.raises(RuntimeError, match="not initialized"):
        check_image_bytes(_jpeg_bytes_for_test())


@patch("moderation.image_checker._session", None)
def test_init_loads_from_model_dir(tmp_path: Path):
    import moderation.image_checker as module

    config = {"id2label": {"0": "normal", "1": "nsfw"}}
    preprocess = {
        "size": {"height": 224, "width": 224},
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    }

    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(preprocess))
    (tmp_path / "model.onnx").write_bytes(b"placeholder")

    with patch.object(module, "IMAGE_MODEL_DIR", tmp_path), \
         patch("onnxruntime.InferenceSession") as mock_ort:
        mock_ort.return_value = MagicMock()
        module._session = None
        init_image_detector()

        mock_ort.assert_called_once_with(
            str(tmp_path / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        assert module._label_map == {0: "normal", 1: "nsfw"}
        assert module._NSFW_LABEL_ID == 1
        assert module._image_size == 224
