"""Chatterbox backend — ENGLISH quality specialist (benchmark contender).

Author: Resemble AI (2025). License: MIT.
Why it's the EN leader: aligned TTS (Llama + WavTokenizer) dramatically
reduces word skips / hallucinations, and built-in voice cloning + an
`exaggeration` knob give expressive, human-like EN. We benchmark it against
CosyVoice2 for the "most human-like English" claim.
"""
from __future__ import annotations

import logging

import numpy as np

from src.core.config import BackendSpecific
from src.tts.base import TTSBackend, TTSRequest, TTSResult

logger = logging.getLogger("chatterbox")


class ChatterboxBackend(TTSBackend):
    name = "chatterbox"
    languages = ["en"]  # clones other langs but shines on EN
    supports_clone = True
    supports_stream = False

    def __init__(self, cfg: BackendSpecific, device: str):
        super().__init__(cfg, device)
        self.model = None

    def load(self) -> None:
        if self._loaded:
            return
        from chatterbox.tts import ChatterboxTTS  # type: ignore

        logger.info("Loading Chatterbox on %s", self.device)
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self._loaded = True

    def synthesize(self, req: TTSRequest) -> TTSResult:
        self.load()
        kwargs = dict(
            text=req.text,
            exaggeration=req.extra.get("exaggeration", self.cfg.exaggeration),
            temperature=req.extra.get("temperature", self.cfg.temperature),
        )
        if req.voice_ref:
            kwargs["audio_prompt_path"] = req.voice_ref
        wav = self.model.generate(**kwargs)  # tensor [1, T] @ 24 kHz
        audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
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
