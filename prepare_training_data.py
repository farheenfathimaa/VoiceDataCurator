"""
prepare_training_data.py
------------------------
Step 2 of the whisper_finetuner pipeline.

Reads output/dataset_manifest.csv (produced by VoiceDataCurator pipeline)
and builds two HuggingFace Dataset objects saved to disk:

  whisper_finetuner/data/curated_train/   -- 80% of accepted-only files
  whisper_finetuner/data/curated_eval/    -- 20% of accepted-only files
  whisper_finetuner/data/raw_train/       -- 80% of ALL files (curated + rejected)
  whisper_finetuner/data/raw_eval/        -- 20% of ALL files (the uncurated baseline)

Each example in the dataset has:
  - audio_array:   float32 numpy array at 16kHz
  - sampling_rate: int (always 16000)
  - text:          reference transcript (from matching .txt file)
  - filename:      source filename (for debugging)
  - status:        "accepted" or "rejected" (from manifest)

REQUIRES: Each audio file in data/raw/ must have a matching .txt transcript.
          Run generate_transcripts.py first to auto-generate them.

Usage:
    python prepare_training_data.py [--manifest ./output/dataset_manifest.csv]
                                    [--audio-dir ./data/raw]
                                    [--output-dir ./whisper_finetuner/data]
                                    [--eval-split 0.2]
                                    [--seed 42]
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # Whisper expects 16 kHz


def load_example(audio_path: Path) -> Optional[Dict]:
    """
    Load one audio file + its matching .txt transcript.

    Returns a dict ready for HuggingFace Dataset, or None if the pair
    is incomplete (missing transcript or unreadable audio).
    """
    txt_path = audio_path.with_suffix(".txt")

    # --- Transcript ---
    if not txt_path.exists():
        logger.warning(f"  No .txt found for {audio_path.name} — skipping. "
                       "Run generate_transcripts.py first.")
        return None

    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.warning(f"  Empty transcript for {audio_path.name} — skipping.")
        return None

    # --- Audio ---
    try:
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"  Could not load audio {audio_path.name}: {e}")
        return None

    return {
        "audio_array": audio,
        "sampling_rate": SAMPLE_RATE,
        "text": text,
        "filename": audio_path.name,
    }


def build_dataset_from_rows(
    rows: pd.DataFrame,
    audio_dir: Path,
) -> Tuple[List[Dict], int]:
    """
    Build a list of examples (dicts) from a manifest DataFrame subset.

    Returns (examples, skipped_count).
    """
    examples = []
    skipped = 0

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Loading audio", ncols=80):
        audio_path = audio_dir / row["filename"]
        if not audio_path.exists():
            logger.warning(f"  Audio file not found: {audio_path}")
            skipped += 1
            continue

        example = load_example(audio_path)
        if example is None:
            skipped += 1
            continue

        example["status"] = row.get("status", "unknown")
        examples.append(example)

    return examples, skipped


def save_split(examples: List[Dict], output_path: Path, name: str) -> None:
    """Convert list of dicts to HuggingFace Dataset and save to disk."""
    try:
        from datasets import Dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' library is not installed.\n"
            "Run: pip install datasets"
        )

    if not examples:
        logger.error(f"  No examples to save for {name} — skipping.")
        return

    # HuggingFace Dataset requires lists, not numpy arrays for non-sequence columns
    # Audio arrays are kept as lists (Dataset handles the conversion)
    dataset = Dataset.from_list(examples)
    dataset.save_to_disk(str(output_path))
    logger.info(f"  Saved {name}: {len(dataset)} examples -> {output_path}")


def prepare(
    manifest_path: Path,
    audio_dir: Path,
    output_dir: Path,
    eval_split: float = 0.2,
    seed: int = 42,
) -> None:
    # ── 1. Load manifest ──────────────────────────────────────────────────────
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run the VoiceDataCurator pipeline first:\n"
            "  python main.py --input ./data/raw"
        )

    manifest = pd.read_csv(manifest_path)
    logger.info(f"Manifest loaded: {len(manifest)} total files")

    # ── 2. Define curated set (accepted only) and raw set (all files) ─────────
    curated = manifest[manifest["status"] == "accepted"].copy()
    raw = manifest.copy()  # includes both accepted and rejected

    logger.info(f"  Curated set (accepted only): {len(curated)} files")
    logger.info(f"  Raw set (all files):          {len(raw)} files")

    if len(curated) < 2:
        raise ValueError("Fewer than 2 accepted files — cannot create a train/eval split.")

    # ── 3. Shuffle and split ──────────────────────────────────────────────────
    curated = curated.sample(frac=1, random_state=seed).reset_index(drop=True)
    raw = raw.sample(frac=1, random_state=seed).reset_index(drop=True)

    curated_eval_n = max(1, int(len(curated) * eval_split))
    raw_eval_n = max(1, int(len(raw) * eval_split))

    curated_eval_rows = curated.iloc[:curated_eval_n]
    curated_train_rows = curated.iloc[curated_eval_n:]
    raw_eval_rows = raw.iloc[:raw_eval_n]
    raw_train_rows = raw.iloc[raw_eval_n:]

    logger.info(
        f"\n  Split sizes:"
        f"\n    curated_train : {len(curated_train_rows)}"
        f"\n    curated_eval  : {len(curated_eval_rows)}"
        f"\n    raw_train     : {len(raw_train_rows)}"
        f"\n    raw_eval      : {len(raw_eval_rows)}"
    )

    # ── 4. Load audio + transcripts ───────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading curated train set...")
    curated_train_ex, skipped = build_dataset_from_rows(curated_train_rows, audio_dir)
    if skipped:
        logger.warning(f"  Skipped {skipped} file(s) with missing audio/transcripts.")

    print("\n[2/4] Loading curated eval set...")
    curated_eval_ex, _ = build_dataset_from_rows(curated_eval_rows, audio_dir)

    print("\n[3/4] Loading raw train set...")
    raw_train_ex, _ = build_dataset_from_rows(raw_train_rows, audio_dir)

    print("\n[4/4] Loading raw eval set...")
    raw_eval_ex, _ = build_dataset_from_rows(raw_eval_rows, audio_dir)

    # ── 5. Save HuggingFace datasets to disk ─────────────────────────────────
    print("\nSaving datasets to disk...")
    save_split(curated_train_ex, output_dir / "curated_train", "curated_train")
    save_split(curated_eval_ex,  output_dir / "curated_eval",  "curated_eval")
    save_split(raw_train_ex,     output_dir / "raw_train",     "raw_train")
    save_split(raw_eval_ex,      output_dir / "raw_eval",      "raw_eval")

    print(
        f"\nAll datasets saved to: {output_dir}\n"
        f"  curated_train : {len(curated_train_ex)} examples\n"
        f"  curated_eval  : {len(curated_eval_ex)} examples\n"
        f"  raw_train     : {len(raw_train_ex)} examples\n"
        f"  raw_eval      : {len(raw_eval_ex)} examples\n"
        f"\nNext step: python finetune_whisper.py"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare HuggingFace datasets from VoiceDataCurator manifest."
    )
    parser.add_argument(
        "--manifest", "-m",
        default="./output/dataset_manifest.csv",
        help="Path to dataset_manifest.csv (default: ./output/dataset_manifest.csv)",
    )
    parser.add_argument(
        "--audio-dir", "-a",
        default="./data/raw",
        help="Directory containing audio + .txt files (default: ./data/raw)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./whisper_finetuner/data",
        help="Where to save the HuggingFace datasets (default: ./whisper_finetuner/data)",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for evaluation (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    args = parser.parse_args()

    prepare(
        manifest_path=Path(args.manifest),
        audio_dir=Path(args.audio_dir),
        output_dir=Path(args.output_dir),
        eval_split=args.eval_split,
        seed=args.seed,
    )
