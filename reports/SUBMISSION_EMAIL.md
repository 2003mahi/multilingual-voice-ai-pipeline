# Submission Email Draft

**Subject:** Internship Assignment — Multilingual Voice AI Pipeline (EN/AR/HI) ✅

Hi [Recruiter/Hiring Manager],

Please find my completed assignment: an open-source, production-grade
multilingual TTS pipeline for English, Arabic, and Hindi.

**The question:** *Which open-source pipeline gives the most human-like voice
with the fastest response for each language, and why?*

**My answer in one line:** a **unified CosyVoice2** stack (EN/AR/HI, streaming,
zero-shot + cross-lingual cloning, Apache-2.0), benchmarked against **Chatterbox**
as the English quality specialist, with XTTS-v2 and MMS-TTS as resilient
fallbacks. Chatterbox leads raw English naturalness; CosyVoice2 wins overall by
covering all three languages in one serving stack.

**What I delivered (all runnable, not just described):**
- FastAPI service with language routing, voice cloning, and graceful fallback
- Automatic benchmark harness: predicted MOS (UTMOS22), WavLM speaker
  similarity, Whisper WER, RTF, latency, GPU memory → CSV + radar/bar charts
- One-command Colab/Kaggle setup + Dockerfile + docker-compose
- 2–5 page technical report with architecture diagrams and a paper-style
  leaderboard
- Smoke tests that pass on CPU via the MMS-TTS fallback

**Repo:** https://github.com/[you]/multilingual-voice-ai
**Live demo:** `python benchmarks/demo.py` (generates EN/AR/HI samples)

I made an explicit engineering trade-off toward *ops simplicity* (one model,
three languages) over a brittle three-model zoo, and I was careful to label
predicted MOS as a proxy rather than true human MOS. Happy to walk through the
architecture or benchmark methodology on a call.

Best,
[Your Name] · [GitHub] · [LinkedIn]
