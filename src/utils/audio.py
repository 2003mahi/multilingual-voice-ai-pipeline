"""Audio I/O helpers: load, resample, save, duration."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf


def load_wav(path: str | Path, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Load any audio file, return (float32 mono @ target_sr, sample_rate)."""
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # to mono
    data = data.astype(np.float32)
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
    return data, target_sr


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    try:
        import librosa

        return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out).astype(np.float32)
    except Exception:
        # Fallback: linear interp (rare path)
        n = int(round(len(x) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(
            np.float32
        )


def save_wav(
    audio: np.ndarray, sr: int, path: str | Path | None = None
) -> bytes | None:
    """Save to path if given, else return WAV bytes (for HTTP responses)."""
    audio = np.asarray(audio, dtype=np.float32)
    if path is not None:
        sf.write(str(path), audio, sr)
        return None
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def duration_seconds(audio: np.ndarray, sr: int) -> float:
    return len(audio) / float(sr) if sr else 0.0
