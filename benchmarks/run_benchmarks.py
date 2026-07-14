#!/usr/bin/env python
"""CLI entry point for the benchmark harness.

Example:
    python benchmarks/run_benchmarks.py --langs en ar hi --backends cosyvoice2 chatterbox
    python benchmarks/run_benchmarks.py --num-samples 12
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the project root importable when run as a script (not just as a module).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import CONFIG
from src.core.router import TTSRouter
from src.eval.benchmark import load_prompts, run_benchmark


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "ar", "hi"])
    ap.add_argument("--backends", nargs="+", default=None)
    ap.add_argument("--num-samples", type=int, default=CONFIG.benchmark.num_samples)
    ap.add_argument("--prompts", default=CONFIG.benchmark.reference_prompts)
    ap.add_argument("--out", default=CONFIG.benchmark.output_dir)
    args = ap.parse_args()

    router = TTSRouter(CONFIG)
    prompts = load_prompts(args.prompts)
    df = run_benchmark(
        router,
        prompts,
        languages=args.langs,
        backends=args.backends,
        num_samples=args.num_samples,
        warmup=CONFIG.benchmark.warmup,
        out_dir=args.out,
        ref_voices={k: getattr(CONFIG.voices, k) for k in ("en", "ar", "hi")},
    )
    print("\n=== SUMMARY (mean per backend) ===")
    print(
        df.groupby("backend")[["utmos", "wer", "rtf", "similarity", "gpu_mem_mb"]]
        .mean()
        .round(3)
    )


if __name__ == "__main__":
    main()
