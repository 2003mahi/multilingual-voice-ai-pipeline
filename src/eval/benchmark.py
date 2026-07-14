"""Benchmark harness: synthesize prompts across backends, score them, and
emit a paper-style results table (CSV) + radar/bar charts (PNG).

Usage (programmatic):  from src.eval.benchmark import run_benchmark
CLI:                   python benchmarks/run_benchmarks.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import CONFIG
from src.core.router import TTSRouter
from src.eval.metrics import MetricResult, evaluate
from src.tts.base import TTSRequest
from src.utils.audio import load_wav

logger = logging.getLogger("benchmark")

LANGUAGES = ["en", "ar", "hi"]


def load_prompts(path: str) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe(v, default=0.0):
    return v if v is not None else default


def run_benchmark(
    router: TTSRouter,
    prompts: dict[str, list[str]],
    *,
    languages: list[str] = LANGUAGES,
    backends: list[str] | None = None,
    num_samples: int = 20,
    warmup: int = 3,
    out_dir: str = "benchmarks/results",
    ref_voices: dict[str, str] | None = None,
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_voices = ref_voices or {}
    results: list[MetricResult] = []

    for lang in languages:
        sents = prompts.get(lang, [])[:num_samples]
        for backend in backends or router.list_backends():
            ref = ref_voices.get(lang)
            # CosyVoice2 / FishSpeech REQUIRE a reference voice; skip cleanly
            # (with a warning) instead of producing empty rows.
            if backend in ("cosyvoice2", "fishspeech") and not ref:
                logger.warning(
                    "Skip %s/%s: reference voice required but not configured.",
                    lang, backend,
                )
                continue
            ref_audio = load_wav(ref)[0] if ref else None
            # warmup (excluded from metrics)
            for _ in range(warmup):
                try:
                    router.synthesize(
                        TTSRequest(text=sents[0], language=lang), requested=backend
                    )
                except Exception:
                    pass
            for text in sents:
                try:
                    res = router.synthesize(
                        TTSRequest(text=text, language=lang, voice_ref=ref),
                        requested=backend,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error("%s/%s failed on '%s': %s", lang, backend, text[:30], e)
                    continue
                m = evaluate(
                    text=text,
                    audio=res.audio,
                    sr=res.sample_rate,
                    ref_audio=ref_audio,
                    gen_time_s=res.gen_time_s,
                    rtf=res.rtf,
                    gpu_mem_mb=getattr(res, "_gpu_mem_mb", 0.0),
                    backend=res.backend,
                    language=res.language,
                    enable_utmos=CONFIG.evaluation.utmos,
                    enable_sim=CONFIG.evaluation.speaker_similarity,
                    enable_wer=CONFIG.evaluation.wer,
                    asr_model=CONFIG.evaluation.asr_model.split("/")[-1],
                )
                results.append(m)
                logger.info(
                    "[%s/%s] MOS=%.2f WER=%.3f RTF=%.3f sim=%.2f",
                    lang, backend, _safe(m.utmos), _safe(m.wer), m.rtf, _safe(m.similarity),
                )

    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(out / "benchmark_results.csv", index=False)
    make_charts(df, out)
    return df


# ── Charts ───────────────────────────────────────────────────────────────
def _norm(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["MOS_n"] = d["utmos"].fillna(0) / 5.0
    d["SIM_n"] = d["similarity"].fillna(0)
    d["INTEL_n"] = 1.0 - d["wer"].fillna(1.0)
    # Speed: lower RTF is better; normalize 1/(1+rtf)
    d["SPEED_n"] = 1.0 / (1.0 + d["rtf"].clip(lower=0))
    # Resource efficiency: less GPU mem is better (inverse, capped)
    maxmem = d["gpu_mem_mb"].replace(0, np.nan).max() or 1.0
    d["RES_n"] = 1.0 - (d["gpu_mem_mb"] / maxmem)
    return d


def make_charts(df: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
    except Exception as e:  # noqa: BLE001
        logger.warning("Charts skipped (matplotlib missing): %s", e)
        return

    d = _norm(df)
    axes = ["MOS_n", "SIM_n", "INTEL_n", "SPEED_n", "RES_n"]
    labels = ["Naturalness", "Voice Clone", "Intelligibility", "Speed", "Efficiency"]
    grouped = d.groupby(["language", "backend"])[axes].mean().reset_index()

    # Radar
    for lang in d["language"].unique():
        sub = grouped[grouped["language"] == lang]
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
        angles += angles[:1]
        for _, row in sub.iterrows():
            vals = [row[a] for a in axes]
            vals += vals[:1]
            ax.plot(angles, vals, label=row["backend"])
            ax.fill(angles, vals, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title(f"Quality radar — {lang.upper()}")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.savefig(out / f"radar_{lang}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # Bar: RTF and Latency and MOS per backend/language
    for metric, title, fname in [
        ("rtf", "Real-Time Factor (lower=better)", "bar_rtf"),
        ("gen_time_s", "Generation Latency s (lower=better)", "bar_latency"),
        ("utmos", "Predicted MOS (higher=better)", "bar_mos"),
    ]:
        if metric not in d.columns:
            continue
        piv = d.pivot_table(index="backend", columns="language", values=metric, aggfunc="mean")
        piv.plot(kind="bar", figsize=(8, 5))
        plt.title(title)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out / f"{fname}.png", dpi=130)
        plt.close()

    # Leaderboard table PNG
    lb = (
        d.groupby(["backend"])
        .agg(MOS=("utmos", "mean"), WER=("wer", "mean"), RTF=("rtf", "mean"),
             Similarity=("similarity", "mean"), GPU_MB=("gpu_mem_mb", "mean"))
        .round(3)
        .sort_values("MOS", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, lb.shape[0] * 0.6 + 1))
    ax.axis("off")
    ax.table(cellText=lb.values, colLabels=lb.columns, rowLabels=lb.index,
             loc="center", cellLoc="center")
    ax.set_title("Leaderboard (mean over all languages)")
    plt.tight_layout()
    plt.savefig(out / "leaderboard.png", dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("Charts written to %s", out)
