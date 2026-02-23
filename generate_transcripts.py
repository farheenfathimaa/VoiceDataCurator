"""
generate_transcripts.py
-----------------------
Step 1 of the whisper_finetuner pipeline.

Auto-generates .txt transcript files for every audio file in data/raw/
using openai/whisper-tiny via librosa (no ffmpeg required).

These transcripts are used as reference text for WER evaluation.

NOTE: This is a self-referential / demo setup — whisper-tiny transcribes
      its own training data. For production use-cases, provide human-verified
      transcripts for more meaningful WER results. This script is fine for
      demonstrating that data quality (SNR, silence, clipping) affects
      fine-tuning outcomes.

Usage:
    python generate_transcripts.py [--input ./data/raw] [--overwrite]
"""

import argparse
import logging
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg"}


def load_whisper_model(model_size: str = "tiny"):
    """Load Whisper model once and reuse across all files."""
    try:
        import whisper
    except ImportError:
        raise SystemExit(
            "openai-whisper is not installed.\n"
            "Run: pip install openai-whisper"
        )
    logger.info(f"Loading Whisper '{model_size}' model for transcription...")
    model = whisper.load_model(model_size)
    logger.info("Model loaded.")
    return model


def transcribe_file(model, filepath: Path) -> str:
    """
    Transcribe a single audio file using Whisper via librosa loader.

    Returns the transcribed text string (stripped), or empty string on failure.
    """
    try:
        import whisper

        # Load audio via librosa — avoids ffmpeg dependency for MP3
        audio, _ = librosa.load(str(filepath), sr=whisper.audio.SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        # Transcribe — decode_options can be tuned; fp16=False for CPU safety
        result = model.transcribe(audio, fp16=False, language=None, verbose=False)
        return result.get("text", "").strip()

    except Exception as e:
        logger.warning(f"  Failed to transcribe {filepath.name}: {e}")
        return ""


def generate_transcripts(input_dir: Path, overwrite: bool = False) -> None:
    audio_files = [
        f for f in sorted(input_dir.iterdir())
        if f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not audio_files:
        logger.error(f"No audio files found in {input_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio file(s) in {input_dir}")

    # Count how many already have transcripts
    existing = sum(1 for f in audio_files if f.with_suffix(".txt").exists())
    if existing and not overwrite:
        logger.info(
            f"{existing} file(s) already have .txt transcripts — skipping them. "
            f"Use --overwrite to regenerate all."
        )

    model = load_whisper_model("tiny")

    skipped = 0
    written = 0
    failed = 0

    for audio_path in tqdm(audio_files, desc="Transcribing", unit="file", ncols=80):
        txt_path = audio_path.with_suffix(".txt")

        if txt_path.exists() and not overwrite:
            skipped += 1
            continue

        text = transcribe_file(model, audio_path)

        if text:
            txt_path.write_text(text, encoding="utf-8")
            logger.debug(f"  {audio_path.name} -> \"{text[:60]}...\"")
            written += 1
        else:
            # Write empty file so the pipeline can still detect the pair
            txt_path.write_text("", encoding="utf-8")
            logger.warning(f"  Empty transcript for {audio_path.name} — written as empty .txt")
            failed += 1

    print(
        f"\nDone. Written: {written} | Skipped (already existed): {skipped} | "
        f"Failed/empty: {failed}"
    )
    if failed:
        print(
            "  TIP: Files with empty transcripts will be excluded from training.\n"
            "  For better WER results, replace auto-transcripts with human-verified text."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-generate .txt transcripts for audio files using Whisper-tiny."
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/raw",
        help="Directory containing audio files (default: ./data/raw)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe even if a .txt file already exists",
    )
    args = parser.parse_args()

    generate_transcripts(Path(args.input), overwrite=args.overwrite)
