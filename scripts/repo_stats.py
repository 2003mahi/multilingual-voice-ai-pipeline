#!/usr/bin/env python
"""Fetch LIVE GitHub stars for every model in the report.

Run this on submission day so the comparison table reflects current stars:
    python scripts/repo_stats.py

Output: a markdown snippet you can paste into README / report.
"""
from __future__ import annotations

import json
import urllib.request

REPOS = {
    "CosyVoice2 (FunAudioLLM/CosyVoice)": "FunAudioLLM/CosyVoice",
    "Chatterbox (resemble-ai/chatterbox)": "resemble-ai/chatterbox",
    "Fish Speech (fishaudio/fish-speech)": "fishaudio/fish-speech",
    "IndexTTS-2 (IndexTeam/IndexTTS-2)": "IndexTeam/IndexTTS-2",
    "XTTS-v2 (coqui-ai/TTS)": "coqui-ai/TTS",
    "MMS-TTS (facebookresearch/fairseq)": "facebookresearch/fairseq",
    "OpenVoice (myshell-ai/OpenVoice)": "myshell-ai/OpenVoice",
    "Bark (suno-ai/bark)": "suno-ai/bark",
}


def fetch_stars(full_name: str) -> int | None:
    url = f"https://api.github.com/repos/{full_name}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tts-report"})
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
            return json.load(r).get("stargazers_count")
    except Exception as e:  # noqa: BLE001
        print(f"  ! {full_name}: {e}")
        return None


if __name__ == "__main__":
    print("\n| Model | GitHub Stars |")
    print("|---|---|")
    for label, repo in REPOS.items():
        stars = fetch_stars(repo)
        print(f"| {label} | {stars if stars is not None else 'n/a'} |")
