"""
app.py
------
Lecture Notes Converter — Streamlit front-end.

Pipeline:
  Upload audio/video  →  Transcribe (faster-whisper)
                      →  Generate Notes + Summary (Gemini 2.5 Flash)
                      →  Export PDF (ReportLab)
                      →  Download

All heavy objects (WhisperModel, GenAI client) are cached in
st.session_state so that UI interactions never re-run expensive
inference steps.
"""

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env for local development (no-op on Streamlit Cloud)
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config  (must be the FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Lecture Notes Converter",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 100 * 1024 * 1024          # 100 MB
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mp4"}
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Custom CSS — academic dark-navy + crimson accent
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global typography ── */
    html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* ── App background ── */
    .stApp { background-color: #F7F8FC; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #16213E 0%, #0F3460 100%);
    }
    section[data-testid="stSidebar"] * { color: #E8EAF6 !important; }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #E94560 !important;
    }

    /* ── Main header banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #16213E 0%, #0F3460 60%, #1A1A2E 100%);
        border-radius: 14px;
        padding: 2.2rem 2.5rem 1.8rem;
        margin-bottom: 1.6rem;
    }
    .hero-banner h1 { color: #FFFFFF; font-size: 2rem; margin: 0 0 0.3rem; }
    .hero-banner p  { color: #B0B8D4; margin: 0; font-size: 0.95rem; }
    .hero-accent    { color: #E94560; }

    /* ── Stage status cards ── */
    .stage-card {
        border-left: 4px solid #E94560;
        background: #FFFFFF;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stage-card.done  { border-color: #2ECC71; }
    .stage-card.error { border-color: #E74C3C; }

    /* ── Tab content area ── */
    .stTabs [data-baseweb="tab-panel"] {
        background: #FFFFFF;
        border-radius: 0 10px 10px 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #E94560, #C0392B) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.4rem !important;
        font-size: 0.95rem !important;
    }
    .stDownloadButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px);
    }

    /* ── Upload area ── */
    .stFileUploader label { font-weight: 600; color: #16213E; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "transcriber": None,
    "genai_client": None,
    "transcript": None,
    "notes": None,
    "summary": None,
    "pdf_path": None,
    "last_filename": None,
    "pipeline_done": False,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Lazy model / client loaders (cached in session_state)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_transcriber():
    """Load and cache the AudioTranscriber (loads Whisper model once)."""
    from transcriber import AudioTranscriber
    return AudioTranscriber()


@st.cache_resource(show_spinner=False)
def _load_genai_client():
    """Build and cache the Google GenAI client once per session."""
    from ai_generator import _build_client
    return _build_client()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_upload(uploaded_file) -> str | None:
    """
    Validate the uploaded file's extension and size.
    Returns an error message string, or None if validation passes.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return (
            f"Unsupported file type '{suffix}'. "
            f"Please upload one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        mb = uploaded_file.size / (1024 * 1024)
        return f"File too large ({mb:.1f} MB). Maximum allowed size is 100 MB."
    return None


def _save_upload_to_temp(uploaded_file) -> str:
    """
    Write the Streamlit UploadedFile to a named temp file on disk and
    return the absolute path.  The caller is responsible for cleanup.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=str(UPLOADS_DIR),
        prefix="lecture_",
    )
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _reset_pipeline_outputs() -> None:
    """Clear previously computed pipeline results from session state."""
    for key in ("transcript", "notes", "summary", "pdf_path", "pipeline_done"):
        st.session_state[key] = None if key != "pipeline_done" else False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎓 Lecture Notes Converter")
    st.markdown(
        "Transform any lecture recording into polished study notes with "
        "AI-powered transcription and summarisation."
    )
    st.markdown("---")

    st.markdown("### ⚙️ Pipeline Stages")
    stages = [
        ("🎙️", "Transcription", "faster-whisper (local)"),
        ("🧠", "Notes Generation", "Gemini 2.5 Flash"),
        ("📋", "Summary Generation", "Gemini 2.5 Flash"),
        ("📄", "PDF Export", "ReportLab"),
    ]
    for icon, name, engine in stages:
        st.markdown(f"{icon} **{name}** — *{engine}*")

    st.markdown("---")

    st.markdown("### 📁 Accepted Formats")
    st.markdown("- MP3 (audio/mpeg)\n- WAV (audio/wav)\n- MP4 (video/mp4)")
    st.markdown("**Max file size:** 100 MB")

    st.markdown("---")

    st.markdown("### ℹ️ About")
    st.markdown(
        "Built with [Streamlit](https://streamlit.io), "
        "[faster-whisper](https://github.com/SYSTRAN/faster-whisper), "
        "and [Gemini 2.5 Flash](https://ai.google.dev)."
    )

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

# Hero banner
st.markdown(
    """
    <div class="hero-banner">
        <h1>🎓 Lecture Notes <span class="hero-accent">Converter</span></h1>
        <p>Upload a lecture recording · Transcribe locally · Generate AI study notes · Export PDF</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── File upload ─────────────────────────────────────────────────────────────
upload_col, info_col = st.columns([3, 2], gap="large")

with upload_col:
    st.markdown("#### 📤 Upload Lecture Recording")
    uploaded_file = st.file_uploader(
        label="Drop your audio or video file here",
        type=["mp3", "wav", "mp4"],
        help="Maximum file size: 100 MB. Supported formats: MP3, WAV, MP4.",
        label_visibility="collapsed",
    )

with info_col:
    st.markdown("#### 📊 Session Metrics")
    m1, m2, m3 = st.columns(3)
    transcript_len = len(st.session_state.transcript.split()) if st.session_state.transcript else 0
    notes_len = len(st.session_state.notes.split()) if st.session_state.notes else 0
    summary_len = len(st.session_state.summary.split()) if st.session_state.summary else 0

    m1.metric("Transcript", f"{transcript_len:,} words" if transcript_len else "—")
    m2.metric("Notes", f"{notes_len:,} words" if notes_len else "—")
    m3.metric("Summary", f"{summary_len:,} words" if summary_len else "—")

st.markdown("---")

# ── Process button & pipeline ────────────────────────────────────────────────

if uploaded_file is not None:
    # Detect if a new file has been uploaded and clear stale results
    if st.session_state.last_filename != uploaded_file.name:
        _reset_pipeline_outputs()
        st.session_state.last_filename = uploaded_file.name

    validation_error = _validate_upload(uploaded_file)
    if validation_error:
        st.error(f"⚠️ {validation_error}")
    else:
        file_mb = uploaded_file.size / (1024 * 1024)
        st.success(
            f"✅ **{uploaded_file.name}** loaded "
            f"({file_mb:.2f} MB) — ready to process."
        )

        run_pipeline = st.button(
            "🚀 Generate Study Notes",
            type="primary",
            disabled=st.session_state.pipeline_done,
            use_container_width=True,
        )

        if run_pipeline and not st.session_state.pipeline_done:
            temp_audio_path: str | None = None

            try:
                # ── Stage 1: Save file to disk ──────────────────────────
                with st.spinner("💾 Saving uploaded file…"):
                    temp_audio_path = _save_upload_to_temp(uploaded_file)
                    logger.info("Saved upload to %s", temp_audio_path)

                # ── Stage 2: Transcription ──────────────────────────────
                with st.spinner(
                    "🎙️ Transcribing audio locally with faster-whisper… "
                    "(this may take a few minutes)"
                ):
                    transcriber = _load_transcriber()
                    transcript = transcriber.transcribe(temp_audio_path)
                    st.session_state.transcript = transcript
                    logger.info(
                        "Transcription complete: %d words.", len(transcript.split())
                    )

                if not st.session_state.transcript:
                    st.error(
                        "Transcription produced no output. "
                        "Please check the audio quality and try again."
                    )
                    st.stop()

                # ── Stage 3: AI Notes ───────────────────────────────────
                with st.spinner(
                    "🧠 Generating structured study notes with Gemini 2.5 Flash…"
                ):
                    from ai_generator import generate_notes, generate_summary

                    client = _load_genai_client()
                    st.session_state.notes = generate_notes(
                        st.session_state.transcript, client=client
                    )
                    logger.info(
                        "Notes generated: %d words.",
                        len(st.session_state.notes.split()),
                    )

                # ── Stage 4: AI Summary ─────────────────────────────────
                with st.spinner(
                    "📋 Writing executive summary with Gemini 2.5 Flash…"
                ):
                    st.session_state.summary = generate_summary(
                        st.session_state.transcript, client=client
                    )
                    logger.info(
                        "Summary generated: %d words.",
                        len(st.session_state.summary.split()),
                    )

                # ── Stage 5: PDF Export ─────────────────────────────────
                with st.spinner("📄 Typesetting PDF with ReportLab…"):
                    from pdf_export import export_to_pdf

                    stem = Path(uploaded_file.name).stem[:40]
                    pdf_path = export_to_pdf(
                        notes=st.session_state.notes,
                        summary=st.session_state.summary,
                        transcript=st.session_state.transcript,
                        filename_prefix=stem,
                    )
                    st.session_state.pdf_path = pdf_path
                    logger.info("PDF exported to %s", pdf_path)

                st.session_state.pipeline_done = True
                st.success(
                    "🎉 Pipeline complete!  "
                    "Review your notes in the tabs below and download the PDF."
                )
                st.rerun()

            except EnvironmentError as e:
                st.error(
                    f"🔑 API key error: {e}\n\n"
                    "Add `GOOGLE_API_KEY` to your `.env` file (local) "
                    "or Streamlit secrets (cloud)."
                )
            except FileNotFoundError as e:
                st.error(f"❌ File error: {e}")
            except RuntimeError as e:
                st.error(f"❌ Processing error: {e}")
            except Exception as e:
                logger.exception("Unexpected pipeline error.")
                st.error(f"❌ Unexpected error: {e}")

            finally:
                # Always clean up the temporary audio file from disk
                if temp_audio_path and os.path.isfile(temp_audio_path):
                    try:
                        os.remove(temp_audio_path)
                        logger.info("Cleaned up temp file: %s", temp_audio_path)
                    except OSError as cleanup_err:
                        logger.warning(
                            "Could not remove temp file %s: %s",
                            temp_audio_path,
                            cleanup_err,
                        )

else:
    st.info(
        "👆 Upload a lecture recording (MP3, WAV, or MP4, up to 100 MB) "
        "to begin."
    )

# ---------------------------------------------------------------------------
# Results workspace — shown only after a successful pipeline run
# ---------------------------------------------------------------------------

if st.session_state.pipeline_done:
    st.markdown("### 📚 Your Study Workspace")

    # ── PDF Download button ──────────────────────────────────────────────
    if st.session_state.pdf_path and os.path.isfile(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        pdf_filename = Path(st.session_state.pdf_path).name
        dl_col, spacer = st.columns([2, 5])
        with dl_col:
            st.download_button(
                label="⬇️  Download PDF Notes",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True,
            )

    st.markdown("")

    # ── Three-tab output workspace ────────────────────────────────────────
    tab_notes, tab_summary, tab_transcript = st.tabs(
        ["📖 Structured Study Notes", "💡 Executive Summary", "📝 Raw Transcript"]
    )

    with tab_notes:
        if st.session_state.notes:
            st.markdown(st.session_state.notes)
        else:
            st.info("No notes available.")

    with tab_summary:
        if st.session_state.summary:
            st.markdown(
                f"""
                <div style="
                    background:#FFFFFF;
                    border-left: 5px solid #E94560;
                    border-radius: 8px;
                    padding: 1.4rem 1.6rem;
                    font-size: 1.0rem;
                    line-height: 1.75;
                    color: #1A1A2E;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                ">
                    {st.session_state.summary.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No summary available.")

    with tab_transcript:
        if st.session_state.transcript:
            word_count = len(st.session_state.transcript.split())
            char_count = len(st.session_state.transcript)
            c1, c2 = st.columns(2)
            c1.metric("Word Count", f"{word_count:,}")
            c2.metric("Character Count", f"{char_count:,}")
            st.markdown("")
            st.text_area(
                label="Full transcript",
                value=st.session_state.transcript,
                height=400,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.info("No transcript available.")

    # ── Reset button ────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Process a New Recording", use_container_width=False):
        _reset_pipeline_outputs()
        st.session_state.last_filename = None
        st.rerun()
