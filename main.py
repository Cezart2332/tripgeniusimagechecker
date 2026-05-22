import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from moderation import (
    check_image_bytes,
    check_text,
    init_image_detector,
    init_text_model,
    is_image_ready,
    is_text_ready,
    text_load_error,
)
from moderation.audit import log_text_preview_enabled, moderation_audit
from moderation.config import INFERENCE_TIMEOUT_SECONDS, MAX_UPLOAD_BYTES

_inference_semaphore = asyncio.Semaphore(1)
_queue_depth = 0


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("moderation.audit").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    logger.info("Loading moderation models (image + text ONNX) ...")
    await asyncio.gather(
        asyncio.to_thread(init_image_detector),
        asyncio.to_thread(init_text_model),
    )
    moderation_audit(
        "startup",
        image_ready=is_image_ready(),
        text_ready=is_text_ready(),
        text_error=text_load_error(),
    )
    if not is_text_ready():
        logger.error("Text model failed to load: %s", text_load_error())
    yield


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger(__name__)


class TextRequest(BaseModel):
    text: str


class ImageResult(BaseModel):
    is_nsfw: bool
    nsfw_score: float
    decision: str = Field(description='"block" or "allow"')
    debug: dict | None = None


class TextResult(BaseModel):
    is_toxic: bool
    scores: dict[str, float]
    decision: str = Field(description='"block" or "allow"')
    debug: dict | None = None


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
        is_nsfw, nsfw_score, nsfw_hits, all_labels = await _run_with_semaphore(
            check_image_bytes, data
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Image check timed out")
    except Exception as exc:
        logger.exception("image-check failed")
        raise HTTPException(status_code=400, detail=f"Error processing image: {exc}")

    decision = "block" if is_nsfw else "allow"
    debug = {
        "bytes": len(data),
        "content_type": file.content_type,
        "nsfw_hits": nsfw_hits,
        "all_detections": all_labels,
    }
    moderation_audit(
        "image-check",
        decision=decision,
        is_nsfw=is_nsfw,
        nsfw_score=nsfw_score,
        bytes=len(data),
        nsfw_hits=nsfw_hits or "none",
        all_detections=all_labels or "none",
    )
    return ImageResult(
        is_nsfw=is_nsfw,
        nsfw_score=nsfw_score,
        decision=decision,
        debug=debug,
    )


@app.post("/text-check", response_model=TextResult)
async def text_check(request: TextRequest):
    try:
        is_toxic, scores = await _run_with_semaphore(check_text, request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Text check timed out")
    except Exception as exc:
        logger.exception("text-check failed")
        raise HTTPException(status_code=503, detail=f"Text model unavailable: {exc}")

    decision = "block" if is_toxic else "allow"
    debug: dict = {"scores": scores}
    audit_fields: dict = {
        "decision": decision,
        "is_toxic": is_toxic,
        "scores": scores,
        "text_len": len(request.text),
    }
    if log_text_preview_enabled():
        from moderation.audit import _preview

        preview = _preview(request.text)
        debug["preview"] = preview
        audit_fields["preview"] = preview

    moderation_audit("text-check", **audit_fields)
    return TextResult(
        is_toxic=is_toxic,
        scores=scores,
        decision=decision,
        debug=debug,
    )
