"""Lightweight, dependency-free language detection.

For a 3-language assignment (EN/AR/HI) a heuristic detector is more than
enough and adds zero latency. For production you'd swap this for
fasttext-lid or a small NLLB classifier — the interface stays the same.
"""
from __future__ import annotations

import re
import unicodedata

# Unicode blocks: Arabic (0600–06FF), Devanagari (0900–097F)
_AR_RE = re.compile(r"[؀-ۿ]")
_HI_RE = re.compile(r"[ऀ-ॿ]")
# Latin script with little/no diacritics ⇒ English (covers most inputs)
_LAT_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Return 'ar', 'hi', or 'en' (default) using script + char counts."""
    if not text:
        return "en"
    norm = unicodedata.normalize("NFC", text)
    ar = len(_AR_RE.findall(norm))
    hi = len(_HI_RE.findall(norm))
    lat = len(_LAT_RE.findall(norm))

    # Mixed-script (transliteration) → decide by non-Latin majority
    if ar > hi and ar >= lat * 0.3:
        return "ar"
    if hi > ar and hi >= lat * 0.3:
        return "hi"
    # Romanized Hindi often has no Devanagari; heuristic Latin fallback to EN
    return "en"


def is_supported(lang: str) -> bool:
    return lang in {"en", "ar", "hi"}
