"""Abstract TTS backend interface.

Every engine (CosyVoice2, Chatterbox, XTTS, MMS) implements this so the
router, API, and benchmark harness stay backend-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np


@dataclass
class TTSRequest:
    text: str
    language: str = "en"
    voice_ref: Optional[str] = None      # path to 16k/24k reference wav for cloning
    speed: float = 1.0
    streaming: bool = False
    # backend-specific extras (exaggeration, instruct text, ...) land here
    extra: dict = field(default_factory=dict)


@dataclass
class TTSResult:
    audio: np.ndarray                     # float32, mono
    sample_rate: int
    backend: str
    language: str
    char_count: int
    gen_time_s: float                    # wall-clock generation time
    rtf: float                            # gen_time / audio_duration
    voice_id: Optional[str] = None
    # streaming chunk metadata
    is_final: bool = True
    chunk_index: int = 0


class TTSBackend(ABC):
    name: str = "base"
    languages: list[str] = ["en"]
    supports_clone: bool = False
    supports_stream: bool = False

    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.device = device
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Lazily import heavy deps and load weights. Idempotent."""

    @abstractmethod
    def synthesize(self, req: TTSRequest) -> TTSResult:
        """Generate full audio for the request."""

    def stream(self, req: TTSRequest) -> Iterator[TTSResult]:
        """Default: synthesize then yield once. Override for true streaming."""
        yield self.synthesize(req)

    @property
    def ready(self) -> bool:
        return self._loaded

    def health(self) -> dict:
        return {
            "backend": self.name,
            "ready": self._loaded,
            "languages": self.languages,
            "clone": self.supports_clone,
            "stream": self.supports_stream,
        }
