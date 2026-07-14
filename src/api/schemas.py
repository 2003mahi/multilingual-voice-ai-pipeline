"""Pydantic models for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel


class TTSRequestJSON(BaseModel):
    text: str
    language: str = "auto"          # auto | en | ar | hi
    backend: str | None = None       # force a backend (else router decides)
    voice_ref: str | None = None     # path to reference wav for cloning
    speed: float = 1.0
    return_base64: bool = True


class TTSResponseJSON(BaseModel):
    audio_base64: str | None = None
    sample_rate: int
    backend: str
    language: str
    char_count: int
    gen_time_s: float
    rtf: float
    gpu_mem_mb: float
    voice_id: str | None = None


class HealthResponse(BaseModel):
    device: str
    backends: dict


class CloneUploadResponse(BaseModel):
    voice_id: str
    path: str
    duration_s: float
