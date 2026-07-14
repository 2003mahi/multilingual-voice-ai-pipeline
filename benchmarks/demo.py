#!/usr/bin/env python
"""Demo: generate one sample per language, save WAVs, print metrics.

Run:  python benchmarks/demo.py
Outputs: assets/demo_en.wav, demo_ar.wav, demo_hi.wav
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from src.core.config import CONFIG
from src.core.router import TTSRouter
from src.eval.metrics import evaluate
from src.tts.base import TTSRequest
from src.utils.audio import save_wav

SAMPLES = {
    "en": "Welcome to our multilingual voice assistant, built with open-source models.",
    "ar": "مرحباً بك في مساعدنا الصوتي متعدد اللغات المبني على نماذج مفتوحة المصدر.",
    "hi": "नमस्ते, यह हमारा बहुभाषी आवाज़ सहायक है जो खुले स्रोत मॉडलों पर आधारित है।",
}


def main() -> None:
    router = TTSRouter(CONFIG)
    out = Path("assets")
    out.mkdir(exist_ok=True)
    for lang, text in SAMPLES.items():
        res = router.synthesize(TTSRequest(text=text, language=lang))
        save_wav(res.audio, res.sample_rate, out / f"demo_{lang}.wav")
        m = evaluate(
            text=text,
            audio=res.audio,
            sr=res.sample_rate,
            gen_time_s=res.gen_time_s,
            rtf=res.rtf,
            gpu_mem_mb=getattr(res, "_gpu_mem_mb", 0.0),
            backend=res.backend,
            language=lang,
            enable_sim=False,  # no reference voice uploaded in demo
        )
        print(
            f"[{lang}] backend={res.backend} MOS={m.utmos:.2f} "
            f"WER={m.wer:.3f} RTF={res.rtf:.3f} "
            f"latency={res.gen_time_s:.2f}s -> assets/demo_{lang}.wav"
        )


if __name__ == "__main__":
    main()
