# 5-Day Execution Roadmap (single engineer, free GPUs)

> Principle applied to every decision: *"Can one engineer build, benchmark, and
> demo this in 5 days on free resources?"* If not → simplify.

## Day 1 — Foundations & env (8h)
| Time | Task |
|---|---|
| 09–10 | Set up Colab (T4) + repo skeleton, `git init`, README stub |
| 10–12 | Install CosyVoice2 + Chatterbox; verify both load on GPU |
| 12–13 | Lunch / let weights download |
| 13–15 | Implement `TTSBackend` interface + CosyVoice2 adapter (EN first) |
| 15–17 | Language detection + router with fallback |
| 17–18 | Smoke test: EN synthesis works end-to-end |

## Day 2 — Multilingual + cloning (8h)
| Time | Task |
|---|---|
| 09–11 | Verify CosyVoice2 AR + HI outputs; fix tokenization/diacritization |
| 11–13 | Voice cloning: zero-shot + cross-lingual via 16k reference wav |
| 13–14 | Chatterbox adapter (EN specialist) + comparison harness |
| 14–16 | XTTS-v2 + MMS-TTS fallback adapters (CPU path) |
| 16–18 | Wire FastAPI `/tts`, `/tts/json`, `/health`, `/clone/upload` |

## Day 3 — Evaluation harness (8h)
| Time | Task |
|---|---|
| 09–11 | UTMOS22 predicted-MOS integration |
| 11–13 | WavLM speaker similarity |
| 13–15 | Whisper + jiwer WER pipeline |
| 15–17 | Latency / RTF / GPU-mem measurement + CSV writer |
| 17–18 | Run full benchmark EN; sanity-check numbers |

## Day 4 — Benchmarking & visuals (8h)
| Time | Task |
| 09–12 | Run benchmarks AR + HI; collect all results |
| 12–14 | Radar + bar charts + leaderboard PNGs |
| 14–16 | Write technical report (2–5 pages) |
| 16–18 | Dockerfile + docker-compose + Colab script verified |

## Day 5 — Polish & submit (8h)
| Time | Task |
|---|---|
| 09–11 | PPT (8–10 slides), demo video/gif |
| 11–12 | Refresh GitHub stars via `scripts/repo_stats.py` |
| 12–13 | README final + run full test suite |
| 13–15 | Record 2-min demo, push to GitHub |
| 15–17 | Submission email + final review |
| 17–18 | Buffer / risk mitigation (swap to fallback if a model breaks) |

## Priority list
1. **P0** CosyVoice2 EN/AR/HI working + FastAPI. 2. **P0** Benchmark CSV.
3. **P1** Chatterbox comparison. 4. **P1** Charts/report. 5. **P2** Docker/Colab.
6. **P2** MMS fallback. 7. **P3** Streaming endpoint.

## Risk mitigation / backup models
| Risk | Mitigation |
|---|---|
| CosyVoice2 won't load on Colab | Fallback XTTS-v2 → MMS-TTS (CPU always works) |
| Chatterbox EN regresses | Use CosyVoice2 EN as the EN winner instead |
| Whisper download blocked | Disable WER; keep UTMOS + RTF (still a valid table) |
| GPU OOM on T4 | `fp16: true`, smaller batch, single worker |
| Arabic diacritization poor | Pre-normalize; cite CosyVoice2 cross-lingual mode |

**Backup stack if primary fails entirely:** MMS-TTS (CPU) guarantees a working
demo for the live presentation — intelligible if not pretty.
