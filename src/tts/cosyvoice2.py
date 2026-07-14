"""CosyVoice2 backend — the UNIFIED engine for EN / AR / HI.

Paper: CosyVoice2: Scalable Multi-lingual Large Speech Generation
       (FunAudioLLM, arXiv 2412.10117). License: Apache-2.0.
Strengths: one model, 30+ languages, streaming, zero-shot + cross-lingual
voice cloning, strong naturalness. This is the production default because it
removes multi-model ops sprawl while covering all three target languages.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch

from src.core.config import BackendSpecific, CONFIG
from src.tts.base import TTSBackend, TTSRequest, TTSResult
from src.utils.audio import load_wav

logger = logging.getLogger("cosyvoice2")


class CosyVoice2Backend(TTSBackend):
    name = "cosyvoice2"
    languages = ["en", "ar", "hi"]
    supports_clone = True
    supports_stream = True

    def __init__(self, cfg: BackendSpecific, device: str):
        super().__init__(cfg, device)
        self.model = None

    def load(self) -> None:
        if self._loaded:
            return
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore

        logger.info("Loading CosyVoice2 (%s) on %s", self.cfg.model_id, self.device)
        self.model = CosyVoice2(
            self.cfg.model_id,
            load_jit=self.cfg.load_jit,
            load_trt=self.cfg.load_trt,
            fp16=(self.device.startswith("cuda") and self.cfg.fp16),
        )
        self._loaded = True

    def _ref_16k(self, voice_ref: str | None, language: str):
        # CosyVoice2 needs a reference voice for EVERY mode → fall back to the
        # configured per-language default when the request supplies none.
        path = voice_ref or getattr(CONFIG.voices, language, "") or ""
        if not path:
            raise RuntimeError(
                "CosyVoice2 requires a reference voice. Set voices.<lang> in "
                "config.yaml or pass voice_ref in the request."
            )
        wav, _ = load_wav(path, target_sr=16000)
        return torch.from_numpy(wav).unsqueeze(0)

    def synthesize(self, req: TTSRequest) -> TTSResult:
        self.load()
        prompt_speech = self._ref_16k(req.voice_ref, req.language)
        # cross-lingual mode clones timbre regardless of prompt language
        stream = self.model.inference_cross_lingual(
            req.text, prompt_speech, stream=False
        )
        chunks = []
        for item in stream:
            chunks.append(item["tts_speech"].squeeze(0).cpu().numpy())
        audio = np.concatenate(chunks).astype(np.float32)
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

    def stream(self, req: TTSRequest) -> "iterator[TTSResult]":  # type: ignore
        self.load()
        prompt_speech = self._ref_16k(req.voice_ref, req.language)
        stream = self.model.inference_cross_lingual(
            req.text, prompt_speech, stream=True
        )
        for i, item in enumerate(stream):
            chunk = item["tts_speech"].squeeze(0).cpu().numpy().astype(np.float32)
            yield TTSResult(
                audio=chunk,
                sample_rate=24000,
                backend=self.name,
                language=req.language,
                char_count=len(req.text),
                gen_time_s=0.0,
                rtf=0.0,
                voice_id=req.voice_ref,
                is_final=False,
                chunk_index=i,
            )
