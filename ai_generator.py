"""
ai_generator.py
---------------
Cognitive layer: uses Google's Gemini 2.5 Flash model (via the
official google-genai SDK) to transform raw transcripts into
structured study notes and concise executive summaries.

API key resolution order:
  1. st.secrets["GOOGLE_API_KEY"]  — for Streamlit Community Cloud
  2. os.getenv("GOOGLE_API_KEY")   — for local .env / shell exports
"""

import os
import logging
import textwrap

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifier
# ---------------------------------------------------------------------------
_MODEL_ID = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_NOTES_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert academic note-taker and educator.
    Your task is to convert a raw lecture transcript into well-structured,
    comprehensive study notes suitable for university-level students.

    Follow these formatting rules EXACTLY:
    - Use **bold** for key terms and important concepts.
    - Use numbered lists (1. 2. 3.) for sequential processes or steps.
    - Use bullet points (- ) for non-sequential details, examples, or elaborations.
    - Use ### for major section headings and ## for the overall topic heading.
    - Include a "Key Takeaways" section at the end with the 3-5 most critical points.
    - Maintain an academic but approachable tone.
    - Preserve all factual content; do NOT invent information not present in the transcript.
    """
).strip()

_NOTES_USER_TEMPLATE = textwrap.dedent(
    """
    Please convert the following lecture transcript into structured study notes.

    TRANSCRIPT:
    {transcript}
    """
).strip()

_SUMMARY_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a professional academic summariser.
    Your task is to produce a concise, high-density executive summary of a
    lecture transcript.

    Requirements:
    - Maximum 250 words.
    - Written in clear, formal prose (no bullet points).
    - Cover: the main topic, core arguments or concepts, and practical
      implications or conclusions.
    - Begin with a single sentence that captures the lecture's central thesis.
    """
).strip()

_SUMMARY_USER_TEMPLATE = textwrap.dedent(
    """
    Please write an executive summary for the following lecture transcript.

    TRANSCRIPT:
    {transcript}
    """
).strip()


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _build_client() -> genai.Client:
    """
    Construct and return a google.genai.Client, resolving the API key
    from Streamlit secrets or environment variables.

    Raises
    ------
    EnvironmentError
        If no API key can be found from either source.
    """
    api_key: str | None = None

    # Attempt 1 — Streamlit secrets (works on Streamlit Community Cloud)
    try:
        import streamlit as st  # imported lazily to keep this module usable outside Streamlit
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        # st.secrets is unavailable (e.g. running in a plain Python script)
        pass

    # Attempt 2 — environment variable / .env loaded by python-dotenv
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. "
            "Set it in .streamlit/secrets.toml (Streamlit Cloud) or "
            "in your .env file / shell environment (local development)."
        )

    logger.info("Initialising Google GenAI client.")
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------

def generate_notes(transcript: str, client: genai.Client | None = None) -> str:
    """
    Generate structured, Markdown-formatted study notes from *transcript*.

    Parameters
    ----------
    transcript : str
        The raw text produced by the transcription pipeline.
    client : genai.Client | None
        Optional pre-built client.  A new client is created if omitted.

    Returns
    -------
    str
        Markdown-formatted study notes as a plain string.

    Raises
    ------
    ValueError
        If *transcript* is empty or blank.
    RuntimeError
        If the Gemini API call fails.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Cannot generate notes from an empty transcript.")

    client = client or _build_client()

    user_message = _NOTES_USER_TEMPLATE.format(transcript=transcript.strip())

    logger.info("Requesting study notes from Gemini (%s).", _MODEL_ID)
    try:
        response = client.models.generate_content(
            model=_MODEL_ID,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=_NOTES_SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=8192,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API error while generating notes: {exc}") from exc

    notes = response.text
    if not notes or not notes.strip():
        raise RuntimeError("Gemini returned an empty response for notes generation.")

    logger.info("Study notes generated successfully (%d chars).", len(notes))
    return notes.strip()


def generate_summary(transcript: str, client: genai.Client | None = None) -> str:
    """
    Generate a concise executive summary from *transcript*.

    Parameters
    ----------
    transcript : str
        The raw text produced by the transcription pipeline.
    client : genai.Client | None
        Optional pre-built client.  A new client is created if omitted.

    Returns
    -------
    str
        Plain-prose executive summary (≤ 250 words).

    Raises
    ------
    ValueError
        If *transcript* is empty or blank.
    RuntimeError
        If the Gemini API call fails.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Cannot generate a summary from an empty transcript.")

    client = client or _build_client()

    user_message = _SUMMARY_USER_TEMPLATE.format(transcript=transcript.strip())

    logger.info("Requesting executive summary from Gemini (%s).", _MODEL_ID)
    try:
        response = client.models.generate_content(
            model=_MODEL_ID,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=_SUMMARY_SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API error while generating summary: {exc}") from exc

    summary = response.text
    if not summary or not summary.strip():
        raise RuntimeError("Gemini returned an empty response for summary generation.")

    logger.info("Executive summary generated successfully (%d chars).", len(summary))
    return summary.strip()
