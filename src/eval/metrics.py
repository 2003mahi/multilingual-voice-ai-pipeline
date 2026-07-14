"""Automatic evaluation metrics.

SENIOR NOTE: True MOS requires human raters. We use UTMOS22 (a strong
SSL-MOS predictor from the VoiceMOS Challenge 2022) as a *proxy* and label
it "predicted MOS". Speaker similarity uses a WavLM-large speaker encoder
(cosine). WER uses Whisper transcription + jiwer. All metrics degrade
gracefully if a dependency is missing so the pipeline never crashes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("eval")


@dataclass
class MetricResult:
    utmos: Optional[float] = None          # predicted MOS (1–5)
    similarity: Optional[float] = None     # cosine sim to reference voice
    wer: Optional[float] = None            # word error rate (0–1)
    char_count: int = 0
    gen_time_s: float = 0.0
    rtf: float = 0.0
    gpu_mem_mb: float = 0.0
    backend: str = ""
    language: str = ""
    text: str = ""
    extra: dict = field(default_factory=dict)


# ── UTMOS (predicted MOS) ────────────────────────────────────────────────
_utmos_predictor = None


def compute_utmos(audio: np.ndarray, sr: int) -> Optional[float]:
    global _utmos_predictor
    try:
        import torch
        from utmos22_strong import UTMOS22StrongPredictor  # type: ignore

        if _utmos_predictor is None:
            _utmos_predictor = UTMOS22StrongPredictor()
        wav = torch.from_numpy(audio.astype(np.float32))
        with torch.no_grad():
            score = _utmos_predictor(wav, sr)
        return float(score)
    except Exception as e:  # noqa: BLE001
        logger.warning("UTMOS unavailable: %s", e)
        return None


# ── Speaker similarity (WavLM-large) ─────────────────────────────────────
_spk_encoder = None


def _get_spk_encoder():
    global _spk_encoder
    if _spk_encoder is None:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

        _spk_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-wavlm-large",
            savedir=".cache/spkrec-wavlm-large",
            run_opts={"device": "cpu"},
        )
    return _spk_encoder


def compute_speaker_similarity(
    gen_audio: np.ndarray, ref_audio: np.ndarray, sr: int
) -> Optional[float]:
    try:
        import torch
        from src.utils.audio import _resample

        enc = _get_spk_encoder()
        g = torch.from_numpy(_resample(gen_audio, sr, 16000)).unsqueeze(0)
        r = torch.from_numpy(_resample(ref_audio, sr, 16000)).unsqueeze(0)
        eg = enc.encode_batch(g, normalize=True)[0, 0].cpu().numpy()
        er = enc.encode_batch(r, normalize=True)[0, 0].cpu().numpy()
        sim = float(np.dot(eg, er) / (np.linalg.norm(eg) * np.linalg.norm(er)))
        return max(0.0, min(1.0, sim))
    except Exception as e:  # noqa: BLE001
        logger.warning("Speaker similarity unavailable: %s", e)
        return None


# ── WER (Whisper + jiwer) ────────────────────────────────────────────────
_whisper_model = None


def _get_whisper(model_name: str):
    global _whisper_model
    if _whisper_model is None or _whisper_model[0] != model_name:
        import whisper  # type: ignore

        _whisper_model = (model_name, whisper.load_model(model_name))
    return _whisper_model[1]


def _normalize(text: str) -> str:
    import re

    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def compute_wer(
    reference_text: str, gen_audio: np.ndarray, sr: int, asr_model: str = "medium"
) -> Optional[float]:
    try:
        import jiwer  # type: ignore
        from src.utils.audio import _resample

        model = _get_whisper(asr_model)
        audio_16k = _resample(gen_audio, sr, 16000).astype(np.float32)
        hyp = model.transcribe(audio_16k, fp16=False, language=None)["text"]
        return float(jiwer.wer(_normalize(reference_text), _normalize(hyp)))
    except Exception as e:  # noqa: BLE001
        logger.warning("WER unavailable: %s", e)
        return None


def evaluate(
    text: str,
    audio: np.ndarray,
    sr: int,
    *,
    ref_audio: Optional[np.ndarray] = None,
    gen_time_s: float = 0.0,
    rtf: float = 0.0,
    gpu_mem_mb: float = 0.0,
    backend: str = "",
    language: str = "",
    enable_utmos: bool = True,
    enable_sim: bool = True,
    enable_wer: bool = True,
    asr_model: str = "medium",
) -> MetricResult:
    """Run all enabled metrics for one synthesized sample."""
    res = MetricResult(
        char_count=len(text),
        gen_time_s=gen_time_s,
        rtf=rtf,
        gpu_mem_mb=gpu_mem_mb,
        backend=backend,
        language=language,
        text=text,
    )
    if enable_utmos:
        res.utmos = compute_utmos(audio, sr)
    if enable_sim and ref_audio is not None:
        res.similarity = compute_speaker_similarity(audio, ref_audio, sr)
    if enable_wer:
        res.wer = compute_wer(text, audio, sr, asr_model=asr_model)
    return res
