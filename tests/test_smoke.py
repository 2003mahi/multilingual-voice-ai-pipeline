"""Smoke test: ensure the pipeline imports, routes, and falls back cleanly.

Run:  pytest tests/   (or)   python -m tests.test_smoke
Designed to pass even WITHOUT a GPU by exercising the MMS-TTS CPU backend.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.config import load_config
from src.core.language import detect_language
from src.core.router import TTSRouter
from src.tts.base import TTSRequest


def test_language_detection():
    assert detect_language("مرحبا بالعالم") == "ar"
    assert detect_language("नमस्ते दुनिया") == "hi"
    assert detect_language("Hello world") == "en"


def test_router_fallback_to_mmsts():
    # Force only the CPU fallback enabled → proves resilience path works
    cfg = load_config()
    cfg.backends = {k: v for k, v in cfg.backends.items() if k == "mmsts"}
    cfg.routing.language_map = {"en": ["mmsts"], "ar": ["mmsts"], "hi": ["mmsts"]}
    cfg.routing.fallback_chain = ["mmsts"]
    router = TTSRouter(cfg)
    res = router.synthesize(TTSRequest(text="Hello from the fallback engine.", language="en"))
    assert res.audio is not None and len(res.audio) > 0
    assert res.backend == "mmsts"


def test_health_shape():
    cfg = load_config()
    router = TTSRouter(cfg)
    h = router.health()
    assert "device" in h and "backends" in h


if __name__ == "__main__":
    test_language_detection()
    test_router_fallback_to_mmsts()
    test_health_shape()
    print("SMOKE TESTS PASSED ✅")
