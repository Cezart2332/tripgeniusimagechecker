"""High-confidence phrase blocklist (ML models miss some Romanian/English slang)."""
from __future__ import annotations

import re
import unicodedata

# Normalized substrings (lowercase, no diacritics). Phrase-style entries reduce false positives.
_BLOCKED_PHRASES: tuple[str, ...] = (
    # Romanian
    "sugi pula",
    "sugeti pula",
    "suge pula",
    "suge-ti pula",
    "muie la",
    "muie ",
    " muie",
    "fututi mortii",
    "fututi ma",
    "futu-ti",
    "dute in pula",
    "du-te in pula",
    "laba mare",
    # English
    "suck my dick",
    "suck my cock",
    "go fuck yourself",
    "motherfucker",
    "piece of shit",
)

# Whole-word patterns (regex after normalization).
_BLOCKED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bf+u+c+k+\b",
        r"\bsh+i+t+\b",
        r"\bc+u+n+t+\b",
        r"\bmuie\b",
        r"\bpul[ae]\b",
    )
)


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_for_profanity(text: str) -> str:
    lowered = _strip_diacritics(text.lower())
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def contains_profanity(text: str) -> bool:
    norm = normalize_for_profanity(text)
    if not norm:
        return False

    for phrase in _BLOCKED_PHRASES:
        if phrase in norm:
            return True

    return any(pattern.search(norm) for pattern in _BLOCKED_PATTERNS)
