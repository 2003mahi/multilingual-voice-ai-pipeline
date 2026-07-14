# =============================================================================
# Colab / Kaggle one-click setup for the Multilingual Voice AI Pipeline
# Paste into a Colab cell (Runtime → GPU → T4) and run top-to-bottom.
# =============================================================================
# 1) Clone repo (or just upload these files)
# !git clone https://github.com/<you>/multilingual-voice-ai.git
# %cd multilingual-voice-ai

# 2) Install core deps + engines (runs ~3-5 min on T4)
import subprocess, sys, os

def run(cmd):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True)

run("pip install -q torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121")
run("pip install -q -r requirements.txt")
run("pip install -q modelscope cosyvoice")      # unified EN/AR/HI
run("pip install -q chatterbox")                # EN quality specialist
run("pip install -q TTS")                        # XTTS-v2 fallback
run("pip install -q utmos22-strong speechbrain jiwer whisper")

# 3) Set env (Colab has no .env file by default)
os.environ["DEVICE"] = "cuda"
os.environ["COSYVOICE_SOURCE"] = "modelscope"

# 4) Smoke test
from src.core.router import TTSRouter
from src.core.config import CONFIG
from src.tts.base import TTSRequest

router = TTSRouter(CONFIG)
res = router.synthesize(TTSRequest(text="Hello from open-source TTS!", language="en"))
print(f"Backend={res.backend} RTF={res.rtf:.3f} dur={len(res.audio)/24000:.2f}s")

# 5) Launch the API (background). Open the proxy URL to interact.
# from src.api.server import app  # import uvicorn and serve, or use /tts endpoints.

# ── Kaggle note ────────────────────────────────────────────────────────────
# On Kaggle: Settings → Accelerator = GPU (P100/T4). The same pip lines work.
# Kaggle blocks some outbound model downloads; set COSYVOICE_SOURCE=huggingface
# and use HF_HUB_CACHE to persist weights across sessions.
