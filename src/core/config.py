"""Central configuration loading from config.yaml + .env overrides."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (src/core/ -> src -> repo root). Makes config/.env resolution
# independent of the current working directory.
ROOT = Path(__file__).resolve().parents[2]


class ServiceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"
    default_language: str = "en"


class DeviceConfig(BaseModel):
    preferred: str = "auto"
    fp16: bool = False


class BackendSpecific(BaseModel):
    enabled: bool = True
    model_id: str = ""
    stream: bool = False
    load_jit: bool = False
    load_trt: bool = False
    clone_prompt_text: str = ""
    exaggeration: float = 0.5
    temperature: float = 0.8


class RoutingConfig(BaseModel):
    default_backend: str = "cosyvoice2"
    language_map: dict[str, list[str]] = Field(default_factory=dict)
    fallback_chain: list[str] = Field(default_factory=list)


class VoicesConfig(BaseModel):
    en: str = ""
    ar: str = ""
    hi: str = ""


class EvalConfig(BaseModel):
    utmos: bool = True
    speaker_similarity: bool = True
    wer: bool = True
    asr_model: str = "openai/whisper-medium"
    sample_rate: int = 24000


class BenchmarkConfig(BaseModel):
    output_dir: str = "benchmarks/results"
    num_samples: int = 20
    warmup: int = 3
    reference_prompts: str = "benchmarks/prompts.json"


class AppConfig(BaseModel):
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    backends: dict[str, BackendSpecific] = Field(default_factory=dict)
    voices: VoicesConfig = Field(default_factory=VoicesConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


class _Env(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")
    device: str = "auto"
    cosyvoice_source: str = "modelscope"
    cosyvoice_model_id: str = "iic/CosyVoice2-0.5B"
    asr_model: str = "openai/whisper-medium"
    enable_cosyvoice2: bool = True
    enable_chatterbox: bool = True
    enable_xtts: bool = True
    enable_mmsts: bool = True


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config, then apply .env overrides.

    Defaults to <repo_root>/config.yaml so it works regardless of CWD.
    """
    # Populate os.environ from .env so that os.getenv(...) picks up its values.
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass

    path = Path(path) if path else ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = AppConfig(**raw)
    env = _Env()

    # Apply env-driven toggles
    cfg.service.host = os.getenv("SERVICE_HOST", cfg.service.host)
    cfg.service.port = int(os.getenv("SERVICE_PORT", cfg.service.port))
    cfg.devices.preferred = env.device
    cfg.evaluation.asr_model = env.asr_model
    if "cosyvoice2" in cfg.backends:
        cfg.backends["cosyvoice2"].model_id = env.cosyvoice_model_id

    # Disable backends flagged off in env
    if not env.enable_cosyvoice2:
        _disable(cfg, "cosyvoice2")
    if not env.enable_chatterbox:
        _disable(cfg, "chatterbox")
    if not env.enable_xtts:
        _disable(cfg, "xtts")
    if not env.enable_mmsts:
        _disable(cfg, "mmsts")
    return cfg


def _disable(cfg: AppConfig, name: str) -> None:
    if name in cfg.backends:
        cfg.backends[name].enabled = False
    for langs in cfg.routing.language_map.values():
        if name in langs:
            langs.remove(name)
    if name in cfg.routing.fallback_chain:
        cfg.routing.fallback_chain.remove(name)


# Singleton used across the app
CONFIG = load_config()
