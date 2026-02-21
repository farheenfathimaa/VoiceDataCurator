"""
language_detector.py
--------------------
Detects the spoken language in an audio clip using OpenAI Whisper.

The WhisperLanguageDetector class loads the model once and reuses it
across all files in a batch to avoid repeated model initialization overhead.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy import to avoid hard crash if whisper not installed
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning(
        "openai-whisper is not installed. Language detection will be skipped. "
        "Run: pip install openai-whisper"
    )


class WhisperLanguageDetector:
    """
    Detects spoken language from audio files using OpenAI Whisper.

    Usage:
        detector = WhisperLanguageDetector(model_size="base")
        language, confidence = detector.detect_language("path/to/audio.wav")
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize and load the Whisper model.

        Args:
            model_size: One of "tiny", "base", "small", "medium", "large".
                        Larger models are more accurate but slower.
        """
        self.model_size = model_size
        self._model = None

        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available. Language detection disabled.")
            return

        logger.info(f"Loading Whisper model '{model_size}'... (first load may take a moment)")
        try:
            self._model = whisper.load_model(model_size)
            logger.info(f"Whisper model '{model_size}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_size}': {e}")
            self._model = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def detect_language(self, filepath: str) -> Tuple[str, float]:
        """
        Detect the spoken language of an audio file.

        Whisper processes a 30-second mel spectrogram of the audio and
        returns log-probabilities for each language. We pick the argmax.

        Returns:
            (language_code, confidence)
                language_code: ISO 639-1 code e.g. "en", "hi", "fr"
                               or "unknown" if detection fails.
                confidence:    Probability of detected language [0.0, 1.0].
        """
        if not self.is_ready:
            return "unknown", 0.0

        if not Path(filepath).exists():
            logger.warning(f"[detect_language] File not found: {filepath}")
            return "unknown", 0.0

        try:
            # Load and pad/trim audio to 30 seconds as Whisper expects
            audio = whisper.load_audio(filepath)
            audio = whisper.pad_or_trim(audio)

            # Compute mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self._model.device)

            # Detect language
            _, probs = self._model.detect_language(mel)

            # Best language by probability
            detected_lang = max(probs, key=probs.get)
            confidence = float(probs[detected_lang])

            logger.debug(
                f"[detect_language] {Path(filepath).name} → {detected_lang} "
                f"({confidence:.2%})"
            )
            return detected_lang, round(confidence, 4)

        except Exception as e:
            logger.warning(f"[detect_language] Error processing {filepath}: {e}")
            return "unknown", 0.0

    def transcribe_snippet(self, filepath: str, max_duration_sec: float = 10.0) -> str:
        """
        Transcribe a short snippet from the audio file (optional utility).

        Useful for spot-checking or logging what Whisper heard.

        Returns:
            Transcribed text string, or empty string on failure.
        """
        if not self.is_ready:
            return ""
        try:
            result = self._model.transcribe(filepath, fp16=False, verbose=False)
            text = result.get("text", "").strip()
            return text[:500]  # cap length
        except Exception as e:
            logger.warning(f"[transcribe_snippet] {filepath}: {e}")
            return ""
