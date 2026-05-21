import io
from unittest.mock import MagicMock, patch

from moderation.image_checker import _is_nsfw_label, check_image_bytes, init_image_detector


def test_is_nsfw_label_matches_exposed_classes():
    assert _is_nsfw_label("FEMALE_BREAST_EXPOSED")
    assert _is_nsfw_label("BUTTOCKS_EXPOSED")
    assert not _is_nsfw_label("FACE_FEMALE")
    assert not _is_nsfw_label("FEMALE_BREAST_COVERED")


def _jpeg_bytes_for_test() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


@patch("moderation.image_checker._detector")
def test_check_image_bytes_blocks_high_exposed_score(mock_detector: MagicMock):
    mock_detector.detect.return_value = [
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.72, "box": [0, 0, 10, 10]},
    ]

    is_nsfw, score = check_image_bytes(_jpeg_bytes_for_test())

    assert is_nsfw is True
    assert score == 0.72
    mock_detector.detect.assert_called_once()
    assert isinstance(mock_detector.detect.call_args[0][0], (bytes, bytearray))


@patch("moderation.image_checker._detector")
def test_check_image_bytes_allows_face_only(mock_detector: MagicMock):
    mock_detector.detect.return_value = [
        {"class": "FACE_FEMALE", "score": 0.9, "box": [0, 0, 10, 10]},
    ]

    is_nsfw, score = check_image_bytes(_jpeg_bytes_for_test())

    assert is_nsfw is False
    assert score == 0.0


@patch("moderation.image_checker._detector", None)
@patch("nudenet.NudeDetector")
def test_init_loads_custom_model_path(mock_nude_cls: MagicMock, tmp_path):
    model_file = tmp_path / "640m.onnx"
    model_file.write_bytes(b"onnx-placeholder")

    import moderation.image_checker as module

    module._detector = None
    with patch.dict("os.environ", {"NUDENET_MODEL_PATH": str(model_file), "NUDENET_INFERENCE_SIZE": "640"}):
        init_image_detector()

    mock_nude_cls.assert_called_once_with(
        model_path=str(model_file),
        inference_resolution=640,
    )
