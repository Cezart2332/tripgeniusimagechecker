from unittest.mock import patch

from fastapi.testclient import TestClient


def test_health_without_models(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert body["image_ready"] is False
    assert body["text_ready"] is False


@patch("main.check_image_bytes", return_value=(False, 0.12, [], []))
def test_image_check_pass(_mock_check, client: TestClient):
    response = client.post(
        "/image-check",
        files={"file": ("photo.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_nsfw"] is False
    assert body["nsfw_score"] == 0.12


@patch("main.check_image_bytes", return_value=(True, 0.91, ["FEMALE_BREAST_EXPOSED=0.91"], []))
def test_image_check_block(_mock_check, client: TestClient):
    response = client.post(
        "/image-check",
        files={"file": ("photo.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["is_nsfw"] is True


def test_image_check_rejects_non_image(client: TestClient):
    response = client.post(
        "/image-check",
        files={"file": ("doc.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


@patch(
    "main.check_text",
    return_value=(False, {"toxic": 0.01, "obscene": 0.02, "insult": 0.01, "severe_toxic": 0.0}),
)
def test_text_check_pass(_mock_check, client: TestClient):
    response = client.post("/text-check", json={"text": "Beautiful trail in the mountains"})
    assert response.status_code == 200
    body = response.json()
    assert body["is_toxic"] is False
    assert "toxic" in body["scores"]


@patch(
    "main.check_text",
    return_value=(True, {"toxic": 0.9, "obscene": 0.1, "insult": 0.2, "severe_toxic": 0.05}),
)
def test_text_check_block(_mock_check, client: TestClient):
    response = client.post("/text-check", json={"text": "offensive content"})
    assert response.status_code == 200
    assert response.json()["is_toxic"] is True


def test_text_check_rejects_empty(client: TestClient):
    response = client.post("/text-check", json={"text": "   "})
    assert response.status_code == 400
