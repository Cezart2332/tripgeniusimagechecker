"""High-confidence blocklist for obvious slurs (ML often misses short/out-of-context words)."""
from __future__ import annotations

import re

# Word-boundary match; extend as needed. Kept small and high-precision.
_BLOCKLIST = (
    "fuck",
    "fucking",
    "motherfucker",
    "shit",
    "bullshit",
    "bitch",
    "asshole",
    "cunt",
    "dick",
    "pussy",
    "cock",
    "whore",
    "slut",
    "nigger",
    "nigga",
    "faggot",
    "retard",
    "pula",
    "muie",
    "fut",
    "curve",
    "pizda",
)

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in _BLOCKLIST) + r")\b",
    re.IGNORECASE | re.UNICODE,
)


def contains_profanity(text: str) -> bool:
    if not text or not text.strip():
        return False
    return _PATTERN.search(text) is not None
