"""Backend registry + per-language router with automatic fallback.

Why a router instead of 3 endpoints? A single /tts endpoint that detects
language and picks the best available backend is the production-grade answer:
one serving surface, one Docker image, no client-side language logic.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from src.core.config import AppConfig
from src.core.language import detect_language
from src.tts.base import TTSBackend, TTSRequest, TTSResult
from src.utils.gpu import resolve_device

logger = logging.getLogger("router")


class TTSRouter:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.devices.preferred)
        self._backends: dict[str, TTSBackend] = {}
        self._build()

    def _build(self) -> None:
        # Lazy import to avoid importing all heavy libs at startup
        from src.tts.cosyvoice2 import CosyVoice2Backend
        from src.tts.chatterbox import ChatterboxBackend
        from src.tts.xtts import XTTSBackend
        from src.tts.mmsts import MMSTSBackend
        from src.tts.fishspeech import FishSpeechBackend

        registry = {
            "cosyvoice2": CosyVoice2Backend,
            "chatterbox": ChatterboxBackend,
            "xtts": XTTSBackend,
            "mmsts": MMSTSBackend,
            "fishspeech": FishSpeechBackend,
        }
        for name, cls in registry.items():
            bcfg = self.cfg.backends.get(name)
            if bcfg and bcfg.enabled:
                try:
                    self._backends[name] = cls(bcfg, self.device)
                    logger.info("Registered backend: %s", name)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Backend %s unavailable: %s", name, e)

    def _load_on_demand(self, name: str) -> Optional[TTSBackend]:
        be = self._backends.get(name)
        if be is None:
            return None
        if not be.ready:
            try:
                be.load()
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to load %s: %s", name, e)
                return None
        return be if be.ready else None

    def choose_backend(self, language: str, requested: Optional[str] = None) -> list[str]:
        """Ordered list of backend names to try for this language."""
        if requested and requested in self._backends:
            return [requested] + self.cfg.routing.fallback_chain
        return self.cfg.routing.language_map.get(
            language, [self.cfg.routing.default_backend]
        ) + self.cfg.routing.fallback_chain

    def synthesize(self, req: TTSRequest, requested: Optional[str] = None) -> TTSResult:
        if not req.language or req.language == "auto":
            req.language = detect_language(req.text)
        from src.utils.audio import duration_seconds
        from src.utils.gpu import gpu_memory_mb, reset_peak_memory

        for name in self.choose_backend(req.language, requested):
            be = self._load_on_demand(name)
            if be is None:
                continue
            try:
                logger.info("Serving %s via %s", req.language, name)
                reset_peak_memory()
                t0 = time.perf_counter()
                res = be.synthesize(req)
                dt = time.perf_counter() - t0
                res.gen_time_s = dt
                dur = duration_seconds(res.audio, res.sample_rate)
                res.rtf = dt / dur if dur > 0 else 0.0
                res._gpu_mem_mb = gpu_memory_mb()  # type: ignore[attr-defined]
                return res
            except Exception as e:  # noqa: BLE001
                logger.error("Backend %s failed: %s", name, e)
                continue
        raise RuntimeError("All TTS backends failed to serve the request.")

    def health(self) -> dict:
        return {
            "device": self.device,
            "backends": {n: b.health() for n, b in self._backends.items()},
        }

    def list_backends(self) -> list[str]:
        return list(self._backends.keys())
