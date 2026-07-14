#!/usr/bin/env python
"""Convenience launcher for the Gradio demo UI.

Run:  python demo_ui.py   →  http://localhost:7860
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui.gradio_app import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
