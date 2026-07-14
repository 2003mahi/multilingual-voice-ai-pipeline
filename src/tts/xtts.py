"""XTTS-v2 backend — multilingual fallback with zero-shot voice cloning.

Author: Coqui. License: Apache-2.0.
Used as a fallback when CosyVoice2 is unavailable. Covers EN/AR/HI with
clone, but naturalness/RTF trail CosyVoice2, so it is NOT the default.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from src.core.config import BackendSpecific
from src.tts.base import TTSBackend, TTSRequest, TTSResult
from src.utils.audio import load_wav

logger = logging.getLogger("xtts")

_LANG_MAP = {"en": "en", "ar": "ar", "hi": "hi"}


class XTTSBackend(TTSBackend):
    name = "xtts"
    languages = ["en", "ar", "hi"]
    supports_clone = True
    supports_stream = False

    def __init__(self, cfg: BackendSpecific, device: str):
        super().__init__(cfg, device)
        self.model = None

    def load(self) -> None:
        if self._loaded:
            return
        from TTS.api import TTS  # type: ignore

        logger.info("Loading XTTS-v2 (%s)", self.cfg.model_id)
        self.model = TTS(self.cfg.model_id).to(self.device)
        self._loaded = True

    def synthesize(self, req: TTSRequest) -> TTSResult:
        self.load()
        lang = _LANG_MAP.get(req.language, "en")
        wav = self.model.tts(
            text=req.text,
            speaker_wav=req.voice_ref,
            language=lang,
        )
        audio = np.asarray(wav, dtype=np.float32)
        # XTTS outputs at 24 kHz natively
        return TTSResult(
            audio=audio,
            sample_rate=24000,
            backend=self.name,
            language=req.language,
            char_count=len(req.text),
            gen_time_s=0.0,
            rtf=0.0,
            voice_id=req.voice_ref,
        )
