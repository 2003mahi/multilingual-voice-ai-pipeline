# Presentation Outline (PPT — 8 to 10 slides)

**Title:** *Most Human-Like, Fastest Open-Source Voice AI for EN / AR / HI*

1. **Title + hook** — the assignment question; one-line answer up front.
2. **What we measured** — naturalness (pred. MOS), similarity, WER, RTF,
   latency, GPU mem. "Why MOS needs humans; we use UTMOS proxy + label it."
3. **Candidate landscape** — table of 8 models, licenses, clone/stream.
4. **The senior insight** — unified engine vs. 3-model zoo; ops sprawl.
5. **Architecture diagram** — Mermaid flow + sequence (routing/fallback).
6. **Winners per language** — EN: Chatterbox/CosyVoice2; AR/HI: CosyVoice2;
   with reasoning.
7. **Benchmark leaderboard** — the results table + radar chart.
8. **Reproducibility** — Colab one-click, Docker, `requirements.txt`, tests.
9. **Risks & backups** — fallback chain; honesty about predicted MOS.
10. **Demo + thanks** — play `assets/demo_en/ar/hi.wav`; QR to GitHub.

**Design tips:** dark theme, one chart per slide, big numbers (MOS 4.4, RTF 0.2),
cite papers in footnotes. Keep text ≤ 6 bullets/slide.
