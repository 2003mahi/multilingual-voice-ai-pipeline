#!/usr/bin/env python
"""Convenience launcher for the Streamlit demo UI.

Run:  python streamlit_demo.py   →  http://localhost:8501

Mirrors demo_ui.py (Gradio). Prefers the `streamlit` console script and
falls back to `python -m streamlit` so it works regardless of PATH. Passing
the script path here (instead of typing it by hand) also avoids the classic
"file has no extension" error from a stray character in the path.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(ROOT, "src", "ui", "streamlit_app.py")


def main() -> int:
    if shutil.which("streamlit"):
        cmd = ["streamlit", "run", SCRIPT]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", SCRIPT]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
