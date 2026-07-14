#!/usr/bin/env python
"""Streaming demo: measure time-to-first-chunk (real streaming latency).

Run:  python benchmarks/stream_demo.py
Requires a streaming backend (CosyVoice2). Others return a clear message.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import CONFIG
from src.core.router import TTSRouter
from src.tts.base import TTSRequest

# Use a streaming-capable backend explicitly.
BACKEND = "cosyvoice2"
TEXT = "Streaming lets us return the first audio chunk before the sentence ends."


def main() -> None:
    router = TTSRouter(CONFIG)
    be = router._backends.get(BACKEND)
    if be is None or not be.supports_stream:
        print(f"Backend '{BACKEND}' not available/streaming — skipping.")
        return
    be.load()
    t0 = time.perf_counter()
    first = None
    n = 0
    for chunk in be.stream(TTSRequest(text=TEXT, language="en")):
        if first is None:
            first = time.perf_counter() - t0
        n += 1
        # In production you'd push chunk.audio to the client here.
    total = time.perf_counter() - t0
    print(f"backend={BACKEND} chunks={n} time_to_first_chunk={first:.3f}s total={total:.3f}s")


if __name__ == "__main__":
    main()
