# Multilingual Voice AI Pipeline — production image
# Base: CUDA 12.1 + cuDNN (matches torch 2.4.1). For CPU-only, swap to
# python:3.10-slim and set DEVICE=cpu + ENABLE_* accordingly.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_CACHE=/app/.cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt

# ── Heavy engines (pin to your runtime CUDA). Comment out what you don't use.
# CosyVoice2 (unified EN/AR/HI)
RUN pip install modelscope cosyvoice
# Chatterbox (EN quality specialist)
RUN pip install chatterbox
# XTTS-v2 fallback
RUN pip install TTS
# MMS-TTS fallback (CPU) ships with transformers (already in requirements)

COPY . /app
RUN mkdir -p logs assets/voices benchmarks/results

EXPOSE 8000
HEALTHCHECK CMD python3 -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
