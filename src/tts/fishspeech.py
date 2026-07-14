"""Fish Speech 1.5 backend — multilingual alternative for AR / HI (+ EN).

Paper: Fish-Speech (fishaudio, arXiv 2411.01144). License: Apache-2.0.
Supports 13+ languages incl. Arabic & Hindi, streaming, voice cloning via a
reference voice. Used as an *alternative* AR/HI engine to CosyVoice2; both
implement the same TTSBackend interface so the router treats them uniformly.

NOTE: Fish Speech is fundamentally a voice-cloning model → a reference voice is
required (same as CosyVoice2). If `fish-speech` is not installed or its API
differs by version, this backend simply fails to load and the router falls
through to the next engine — it never blocks the pipeline.
"""
from __future__ import annotations

import logging

import numpy as np

from src.core.config import BackendSpecific, CONFIG
from src.tts.base import TTSBackend, TTSRequest, TTSResult
from src.utils.audio import load_wav, _resample

logger = logging.getLogger("fishspeech")

_FS_LANG = {"en": "en", "ar": "ar", "hi": "hi"}


class FishSpeechBackend(TTSBackend):
    name = "fishspeech"
    languages = ["en", "ar", "hi"]
    supports_clone = True
    supports_stream = False

    def __init__(self, cfg: BackendSpecific, device: str):
        super().__init__(cfg, device)
        self.model = None

    def load(self) -> None:
        if self._loaded:
            return
        # Heavy import kept local so a missing package never breaks startup.
        from fish_speech.models import FishSpeech  # type: ignore
        from fish_speech.utils import auto_load  # type: ignore

        model_id = self.cfg.model_id or "fishaudio/fish-speech-1.5"
        logger.info("Loading Fish Speech (%s) on %s", model_id, self.device)
        self.model = auto_load(model_id, device=self.device)
        self._loaded = True

    def _ref_32k(self, voice_ref: str | None, language: str):
        path = voice_ref or getattr(CONFIG.voices, language, "") or ""
        if not path:
            raise RuntimeError(
                "FishSpeech requires a reference voice. Set voices.<lang> in "
                "config.yaml or pass voice_ref in the request."
            )
        wav, _ = load_wav(path, target_sr=32000)  # Fish Speech expects 32 kHz
        return np.expand_dims(wav, 0)

    def synthesize(self, req: TTSRequest) -> TTSResult:
        self.load()
        import torch

        lang = _FS_LANG.get(req.language, "en")
        ref_audio = self._ref_32k(req.voice_ref, req.language)
        ref_audio_t = torch.from_numpy(ref_audio).to(self.device)

        # Tokenize input (and optional reference transcription).
        text_tokens = self.model.tokenize(req.text, lang)
        ref_text = self.cfg.clone_prompt_text
        ref_tokens = self.model.tokenize(ref_text, lang) if ref_text else None

        with torch.no_grad():
            codes = self.model.forward(
                text_tokens,
                reference_audio=ref_audio_t,
                reference_text=ref_tokens,
            )
            audio, sr = self.model.decode(codes)

        audio = np.asarray(audio).squeeze().astype(np.float32)
        if int(sr) != 24000:
            audio = _resample(audio, int(sr), 24000)
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
