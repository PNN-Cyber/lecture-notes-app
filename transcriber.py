"""
transcriber.py
--------------
Local speech-to-text transcription using faster-whisper.
Automatically selects GPU (CUDA) when available, gracefully falls
back to CPU with int8 quantisation for efficient on-device inference.
"""

import os
import logging
from typing import Optional

import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Wraps faster-whisper's WhisperModel to provide a simple,
    high-level transcription interface.

    The model is loaded once at construction time and reused
    across multiple calls, making it safe to cache in
    st.session_state for the lifetime of a Streamlit session.
    """

    MODEL_SIZE: str = "base"

    def __init__(self) -> None:
        device, compute_type = self._resolve_device()
        logger.info(
            "Loading Whisper '%s' model on device='%s' compute_type='%s'",
            self.MODEL_SIZE,
            device,
            compute_type,
        )
        self.model = WhisperModel(
            self.MODEL_SIZE,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded successfully.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> tuple[str, str]:
        """
        Return the (device, compute_type) pair that best suits the
        current hardware:
          - CUDA GPU available → ("cuda", "float16")
          - CPU only          → ("cpu",  "int8")
        """
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected — using float16 compute.")
            return "cuda", "float16"
        logger.info("No GPU detected — falling back to CPU int8 compute.")
        return "cpu", "int8"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe the audio file at *audio_path* and return the full
        transcript as a single, clean, UTF-8 string.

        Parameters
        ----------
        audio_path : str
            Absolute or relative path to an mp3, wav, or mp4 file.
        language : str | None
            ISO 639-1 language code (e.g. "en", "fr").  Pass ``None``
            to let Whisper auto-detect the language (default).

        Returns
        -------
        str
            The complete transcript with leading/trailing whitespace
            stripped and internal consecutive whitespace collapsed.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not point to an existing file.
        RuntimeError
            If faster-whisper raises an unrecoverable error during
            inference.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path!r}"
            )

        transcribe_kwargs: dict = {
            "beam_size": 5,
            "vad_filter": True,          # skip non-speech segments
            "vad_parameters": {"min_silence_duration_ms": 500},
        }
        if language:
            transcribe_kwargs["language"] = language

        try:
            segments, info = self.model.transcribe(audio_path, **transcribe_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Whisper transcription failed for {audio_path!r}: {exc}"
            ) from exc

        logger.info(
            "Detected language '%s' (probability %.2f).",
            info.language,
            info.language_probability,
        )

        # Concatenate segment text, normalising whitespace
        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        transcript = " ".join(parts)

        # Collapse multiple internal spaces/newlines into a single space
        import re
        transcript = re.sub(r"\s+", " ", transcript).strip()

        if not transcript:
            logger.warning("Transcription produced an empty result for %r.", audio_path)

        return transcript
