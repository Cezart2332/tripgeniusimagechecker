import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from moderation import (
    check_image_bytes,
    check_text,
    init_image_detector,
    is_image_ready,
    is_text_ready,
    text_load_error,
)
from moderation.config import INFERENCE_TIMEOUT_SECONDS, MAX_UPLOAD_BYTES

_inference_semaphore = asyncio.Semaphore(1)
_queue_depth = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_image_detector()
    yield


app = FastAPI(lifespan=lifespan)


class TextRequest(BaseModel):
    text: str


class ImageResult(BaseModel):
    is_nsfw: bool
    nsfw_score: float


class TextResult(BaseModel):
    is_toxic: bool
    scores: dict[str, float]


class HealthResult(BaseModel):
    status: str
    image_ready: bool
    text_ready: bool
    text_load_error: str | None
    queue_depth: int


async def _run_with_semaphore(func, *args):
    global _queue_depth
    async with _inference_semaphore:
        _queue_depth += 1
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(func, *args),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )
        finally:
            _queue_depth -= 1


@app.get("/health", response_model=HealthResult)
def health():
    return HealthResult(
        status="ok" if is_image_ready() else "degraded",
        image_ready=is_image_ready(),
        text_ready=is_text_ready(),
        text_load_error=text_load_error(),
        queue_depth=_queue_depth,
    )


@app.post("/image-check", response_model=ImageResult)
async def image_check(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        is_nsfw, nsfw_score = await _run_with_semaphore(check_image_bytes, data)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Image check timed out")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error processing image: {exc}")

    return ImageResult(is_nsfw=is_nsfw, nsfw_score=nsfw_score)


@app.post("/text-check", response_model=TextResult)
async def text_check(request: TextRequest):
    try:
        is_toxic, scores = await _run_with_semaphore(check_text, request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Text check timed out")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Text model unavailable: {exc}")

    return TextResult(is_toxic=is_toxic, scores=scores)
