"""Stdout audit lines for Docker/Coolify logs (uvicorn access logs only show HTTP 200)."""
from __future__ import annotations

import logging
import os
from typing import Any

_audit_logger = logging.getLogger("moderation.audit")


def _preview(text: str, max_len: int = 48) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def moderation_audit(event: str, **fields: Any) -> None:
    parts = " ".join(f"{k}={v!r}" for k, v in fields.items())
    line = f"[MODERATION] {event} {parts}"
    print(line, flush=True)
    _audit_logger.info(line)


def log_text_preview_enabled() -> bool:
    return os.getenv("MODERATION_LOG_PREVIEW", "true").lower() in ("1", "true", "yes")
