"""Streamlit demo UI for the multilingual voice AI pipeline.

Run:
    streamlit run src/ui/streamlit_app.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import streamlit as st
from src.core.config import CONFIG
from src.core.language import detect_language
from src.core.router import TTSRouter
from src.tts.base import TTSRequest
from src.utils.audio import save_wav

# Set page config for beautiful layout
st.set_page_config(
    page_title="Multilingual Voice AI (EN / AR / HI)",
    page_icon="🌍",
    layout="wide",
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #ff4b4b, #ff7676, #ff9f9f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize router once using Streamlit cache
@st.cache_resource
def get_router():
    return TTSRouter(CONFIG)

router = get_router()
# Separate output path from the Gradio demo so both UIs can run side-by-side
# without clobbering each other's generated audio.
OUT_WAV = os.path.join(ROOT, "assets", "demo_ui_streamlit.wav")

st.title("🌍 Multilingual Voice AI Pipeline")
st.markdown("### Open-source TTS with voice cloning & automatic fallback (EN / AR / HI)")

col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area("Input Text", placeholder="Type in English, Arabic, or Hindi...", height=150)
    
    with st.expander("Advanced Settings", expanded=True):
        col_lang, col_backend = st.columns(2)
        with col_lang:
            lang = st.selectbox("Language Detection Mode", options=["auto", "en", "ar", "hi"], index=0)
        with col_backend:
            backend = st.selectbox(
                "Engine Backend (leave blank for Auto-Routing)",
                options=["", "cosyvoice2", "fishspeech", "chatterbox", "xtts", "mmsts"],
                index=0
            )

with col2:
    st.write("🎙️ **Voice Cloning (Optional)**")
    ref_file = st.file_uploader("Upload reference audio file (.wav, .mp3)", type=["wav", "mp3"])

if st.button("🔊 Generate Speech", type="primary", use_container_width=True):
    if not text or not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Synthesizing audio..."):
            # Save uploaded file to a temporary file if provided
            temp_ref_path = None
            if ref_file is not None:
                ext = os.path.splitext(ref_file.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                    f.write(ref_file.read())
                    temp_ref_path = f.name
            
            lang_to_use = lang if lang != "auto" else detect_language(text)
            
            try:
                res = router.synthesize(
                    TTSRequest(text=text, language=lang_to_use, voice_ref=temp_ref_path),
                    requested=(backend or None),
                )
                
                os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
                save_wav(res.audio, res.sample_rate, OUT_WAV)
                
                st.success("🎉 Synthesis Completed Successfully!")
                st.audio(OUT_WAV)
                
                # Show Metrics
                st.markdown("### 📊 Inference Metrics")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Selected Backend", res.backend)
                m_col1.metric("Detected/Used Language", res.language)
                m_col2.metric("RTF (Real-Time Factor)", f"{res.rtf:.3f}")
                m_col2.metric("Generation Time", f"{res.gen_time_s:.2f}s")
                m_col3.metric("Character Count", res.char_count)
                gpu_mem = getattr(res, '_gpu_mem_mb', 0.0)
                m_col3.metric("GPU Memory Usage", f"{gpu_mem:.0f} MB" if gpu_mem > 0 else "N/A")
                
            except Exception as e:
                st.error(f"❌ Error during synthesis: {e}")
            finally:
                # Cleanup temp file
                if temp_ref_path and os.path.exists(temp_ref_path):
                    try:
                        os.remove(temp_ref_path)
                    except Exception:
                        pass
