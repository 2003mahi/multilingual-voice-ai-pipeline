"""Gradio demo UI for the multilingual voice AI pipeline.

Run:
    python -m src.ui.gradio_app          # (project root on PYTHONPATH)
    python src/ui/gradio_app.py          # path bootstrap handles import
Then open http://localhost:7860

Reuses the same TTSRouter as the FastAPI service, so cloning, routing, and
fallback all behave identically. No separate server required.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import gradio as gr  # type: ignore
from src.core.config import CONFIG
from src.core.language import detect_language
from src.core.router import TTSRouter
from src.tts.base import TTSRequest
from src.utils.audio import save_wav

router = TTSRouter(CONFIG)
OUT_WAV = os.path.join(ROOT, "assets", "demo_ui.wav")


def generate(text: str, language: str, backend: str, ref_file: str | None):
    if not text or not text.strip():
        return None, "⚠️ Please enter some text."
    lang = language if language != "auto" else detect_language(text)
    try:
        res = router.synthesize(
            TTSRequest(text=text, language=lang, voice_ref=ref_file or None),
            requested=(backend or None),
        )
    except Exception as e:  # noqa: BLE001
        return None, f"❌ Error: {e}"
    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    save_wav(res.audio, res.sample_rate, OUT_WAV)
    metrics = (
        f"backend={res.backend}  lang={res.language}\n"
        f"RTF={res.rtf:.3f}  latency={res.gen_time_s:.2f}s  "
        f"chars={res.char_count}  gpu={getattr(res, '_gpu_mem_mb', 0.0):.0f}MB"
    )
    return OUT_WAV, metrics


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Multilingual Voice AI (EN / AR / HI)") as demo:
        gr.Markdown("# 🌍 Multilingual Voice AI — EN / AR / HI\nOpen-source TTS with voice cloning & automatic fallback.")
        with gr.Row():
            text = gr.Textbox(label="Text", lines=3, placeholder="Type in English, Arabic, or Hindi…")
            lang = gr.Dropdown(["auto", "en", "ar", "hi"], value="auto", label="Language (auto-detect)")
        with gr.Row():
            backend = gr.Dropdown(
                ["", "cosyvoice2", "fishspeech", "chatterbox", "xtts", "mmsts"],
                value="", label="Backend (blank = router picks best)",
            )
            ref = gr.Audio(source="upload", type="filepath", label="Reference voice (for cloning)")
        btn = gr.Button("🔊 Speak", variant="primary")
        out_audio = gr.Audio(label="Generated speech", type="filepath")
        out_metrics = gr.Textbox(label="Metrics", lines=3)
        btn.click(generate, [text, lang, backend, ref], [out_audio, out_metrics])
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
