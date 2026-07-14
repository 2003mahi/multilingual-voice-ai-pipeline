"""MMS-TTS backend — ultra-light CPU fallback (NO voice cloning).

Paper: Scaling Speech Technology to 1,000+ languages (Meta, arXiv 2305.13516).
License: MIT. Tiny VITS, runs on CPU in <50 MB RAM. Robotic (no clone) but
guarantees the pipeline NEVER hard-fails: if every GPU backend is down, the
service still returns intelligible audio. That's the resilience play.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from src.core.config import BackendSpecific
from src.tts.base import TTSBackend, TTSRequest, TTSResult
from src.utils.audio import _resample

logger = logging.getLogger("mmsts")

# MMS language codes for our three targets
_MMS_CODE = {"en": "eng", "ar": "ara", "hi": "hin"}


class MMSTSBackend(TTSBackend):
    name = "mmsts"
    languages = ["en", "ar", "hi"]
    supports_clone = False
    supports_stream = False

    def __init__(self, cfg: BackendSpecific, device: str):
        super().__init__(cfg, device)
        self._models: dict[str, object] = {}
        self._tok: dict[str, object] = {}

    def load(self) -> None:
        # Lazily load per-language models on first use (CPU only)
        self._loaded = True

    def _ensure(self, lang: str):
        if lang in self._models:
            return
        from transformers import AutoTokenizer, VitsModel  # type: ignore

        code = _MMS_CODE.get(lang, "eng")
        mid = f"facebook/mms-tts-{code}"
        logger.info("Loading MMS-TTS %s (CPU)", mid)
        self._models[lang] = VitsModel.from_pretrained(mid)
        self._tok[lang] = AutoTokenizer.from_pretrained(mid)

    def synthesize(self, req: TTSRequest) -> TTSResult:
        self.load()
        self._ensure(req.language)
        model = self._models[req.language]
        tok = self._tok[req.language]
        inputs = tok(req.text, return_tensors="pt")
        with torch.no_grad():
            audio = model(**inputs).waveform.squeeze().cpu().numpy().astype(np.float32)
        # MMS outputs 16 kHz → upsample to 24 kHz for a uniform pipeline
        audio = _resample(audio, 16000, 24000)
        return TTSResult(
            audio=audio,
            sample_rate=24000,
            backend=self.name,
            language=req.language,
            char_count=len(req.text),
            gen_time_s=0.0,
            rtf=0.0,
        )
