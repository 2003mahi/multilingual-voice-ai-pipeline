"""FastAPI service for the multilingual voice AI pipeline.

Run:  uvicorn src.api.server:app --host 0.0.0.0 --port 8000
Or:   docker compose up   (see Dockerfile / docker-compose.yml)
"""
from __future__ import annotations

import base64
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.api.schemas import (
    CloneUploadResponse,
    HealthResponse,
    TTSRequestJSON,
    TTSResponseJSON,
)
from src.core.config import CONFIG
from src.core.language import detect_language
from src.core.router import TTSRouter
from src.tts.base import TTSRequest
from src.utils.audio import duration_seconds, load_wav, save_wav

logging.basicConfig(level=getattr(logging, CONFIG.service.log_level))
logger = logging.getLogger("api")

app = FastAPI(
    title="Multilingual Voice AI Pipeline",
    version="1.0.0",
    description="Open-source TTS for EN / AR / HI with voice cloning & streaming.",
)
router = TTSRouter(CONFIG)
VOICE_DIR = Path("assets/voices")
VOICE_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def _startup() -> None:
    logger.info("Device: %s | Backends: %s", router.device, router.list_backends())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(device=router.device, backends=router.health()["backends"])


@app.get("/backends")
def backends() -> list[str]:
    return router.list_backends()


@app.post("/tts", summary="Text → speech (WAV bytes + metrics header)")
def tts_audio(req: TTSRequestJSON) -> Response:
    """Returns audio/wav. Metrics are returned in the X-TTS-Metrics header."""
    language = req.language if req.language != "auto" else detect_language(req.text)
    tts_req = TTSRequest(
        text=req.text,
        language=language,
        voice_ref=req.voice_ref,
        speed=req.speed,
    )
    try:
        res = router.synthesize(tts_req, requested=req.backend)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"TTS failed: {e}")

    wav = save_wav(res.audio, res.sample_rate)
    metrics = {
        "backend": res.backend,
        "language": res.language,
        "gen_time_s": round(res.gen_time_s, 3),
        "rtf": round(res.rtf, 3),
        "char_count": res.char_count,
        "gpu_mem_mb": round(getattr(res, "_gpu_mem_mb", 0.0), 1),
    }
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"X-TTS-Metrics": base64.b64encode(str(metrics).encode()).decode()},
    )


@app.post("/tts/json", response_model=TTSResponseJSON)
def tts_json(req: TTSRequestJSON) -> TTSResponseJSON:
    language = req.language if req.language != "auto" else detect_language(req.text)
    tts_req = TTSRequest(
        text=req.text, language=language, voice_ref=req.voice_ref, speed=req.speed
    )
    try:
        res = router.synthesize(tts_req, requested=req.backend)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"TTS failed: {e}")

    b64 = (
        base64.b64encode(save_wav(res.audio, res.sample_rate)).decode()
        if req.return_base64
        else None
    )
    return TTSResponseJSON(
        audio_base64=b64,
        sample_rate=res.sample_rate,
        backend=res.backend,
        language=res.language,
        char_count=res.char_count,
        gen_time_s=round(res.gen_time_s, 3),
        rtf=round(res.rtf, 3),
        gpu_mem_mb=round(getattr(res, "_gpu_mem_mb", 0.0), 1),
        voice_id=res.voice_id,
    )


@app.post("/tts/stream", summary="Streaming TTS (WAV chunks, low latency)")
def tts_stream(req: TTSRequestJSON) -> Response:
    """Stream audio chunks as they are generated (CosyVoice2 supports this).

    Returns a sequence of WAV blobs; the first arrives after the model emits
    its first chunk (measure time-to-first-chunk for latency). Backends without
    streaming return HTTP 400. For browser playback use /tts instead.
    """
    language = req.language if req.language != "auto" else detect_language(req.text)
    be = router._backends.get(req.backend) or router._load_on_demand(
        router.choose_backend(language, req.backend)[0]
    )
    if be is None or not be.supports_stream:
        raise HTTPException(
            status_code=400,
            detail=f"Backend '{req.backend or 'auto'}' does not support streaming.",
        )
    tts_req = TTSRequest(text=req.text, language=language, voice_ref=req.voice_ref)

    def chunk_gen():
        for chunk in be.stream(tts_req):
            yield save_wav(chunk.audio, chunk.sample_rate)

    return StreamingResponse(chunk_gen(), media_type="audio/wav")


@app.post("/clone/upload", response_model=CloneUploadResponse, summary="Upload a reference voice")
def upload_voice(file: UploadFile = File(...), language: str = "en") -> CloneUploadResponse:
    vid = f"{language}_{uuid.uuid4().hex[:8]}"
    path = VOICE_DIR / f"{vid}.wav"
    data = file.file.read()
    path.write_bytes(data)
    wav, _ = load_wav(path)
    return CloneUploadResponse(
        voice_id=vid, path=str(path), duration_s=round(duration_seconds(wav, 24000), 2)
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=CONFIG.service.host,
        port=CONFIG.service.port,
        workers=CONFIG.service.workers,
    )
