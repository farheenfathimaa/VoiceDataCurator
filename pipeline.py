"""
pipeline.py
-----------
Orchestrates the full VoiceDataCurator processing pipeline:
  1. Discover all audio files in input_dir
  2. Run audio quality analysis on each file
  3. Detect spoken language via Whisper
  4. Apply configured thresholds to accept/reject each file
  5. Move rejected files to rejected_dir (unless --dry-run)
  6. Write dataset_manifest.csv to output_dir
  7. Write a run report to log_dir
"""

import logging
import shutil
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd  # noqa: F401  (kept for potential future use)
from tqdm import tqdm

from audio_analyzer import analyze_audio_file, AudioMetrics
from language_detector import WhisperLanguageDetector

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary statistics for one pipeline run."""
    total: int = 0
    accepted: int = 0
    rejected: int = 0
    skipped: int = 0
    duration_sec: float = 0.0
    manifest_path: str = ""
    report_path: str = ""


@dataclass
class FileRecord:
    """Full record for one processed audio file."""
    filename: str
    filepath: str
    language: str = "unknown"
    lang_confidence: float = 0.0
    duration: float = 0.0
    snr_db: float = 0.0
    silence_ratio: float = 0.0
    clipping_ratio: float = 0.0
    sample_rate: int = 0
    quality_score: float = 0.0
    status: str = "unknown"
    rejection_reason: str = ""


def _discover_files(input_dir: Path, supported_formats: List[str]) -> List[Path]:
    """Find all supported audio files recursively under input_dir."""
    files = []
    for fmt in supported_formats:
        files.extend(input_dir.rglob(f"*{fmt}"))
    return sorted(set(files))


def _is_language_accepted(language: str, accepted_languages: List[str]) -> bool:
    """Return True if accepted_languages is empty (accept all) or language is in the list."""
    if not accepted_languages:
        return True
    return language in accepted_languages


def run_pipeline(config: dict, dry_run: bool = False) -> PipelineResult:
    """
    Execute the full curation pipeline.

    Args:
        config:  Parsed config.yaml as a dict.
        dry_run: If True, don't move any files — only report what would happen.

    Returns:
        PipelineResult with run statistics.
    """
    pipeline_cfg = config.get("pipeline", {})
    audio_cfg = config.get("audio", {})
    lang_cfg = config.get("language", {})

    input_dir = Path(pipeline_cfg.get("input_dir", "./data/raw"))
    output_dir = Path(pipeline_cfg.get("output_dir", "./output"))
    rejected_dir = Path(pipeline_cfg.get("rejected_dir", "./rejected"))
    log_dir = Path(pipeline_cfg.get("log_dir", "./logs"))
    manifest_filename = pipeline_cfg.get("manifest_filename", "dataset_manifest.csv")
    supported_formats = pipeline_cfg.get("supported_formats", [".wav", ".mp3"])

    accepted_languages = lang_cfg.get("accepted_languages", [])
    whisper_model = lang_cfg.get("whisper_model", "base")
    skip_lang_detect = lang_cfg.get("skip_detection", False)

    min_snr = audio_cfg.get("min_snr_db", 10.0)
    max_silence = audio_cfg.get("max_silence_ratio", 0.4)
    max_clipping = audio_cfg.get("max_clipping_ratio", 0.01)

    # --- Setup directories ---
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover files ---
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return PipelineResult()

    audio_files = _discover_files(input_dir, supported_formats)
    if not audio_files:
        logger.warning(f"No audio files found in: {input_dir}")
        return PipelineResult()

    logger.info(f"Found {len(audio_files)} audio file(s) in {input_dir}")
    if dry_run:
        logger.info("[DRY RUN] No files will be moved.")

    # --- Load language detector ---
    if skip_lang_detect:
        logger.info("Language detection DISABLED (--no-language-detect). All files tagged as 'unknown'.")
        detector = None
    else:
        detector = WhisperLanguageDetector(model_size=whisper_model)

    # --- Process files ---
    records: List[FileRecord] = []
    result = PipelineResult(total=len(audio_files))
    start_time = time.time()

    for audio_path in tqdm(audio_files, desc="Processing audio", unit="file", ncols=80):
        filename = audio_path.name
        record = FileRecord(filename=filename, filepath=str(audio_path))

        # Quality analysis
        metrics: AudioMetrics = analyze_audio_file(str(audio_path), config)

        if not metrics.is_loadable:
            record.status = "rejected"
            record.rejection_reason = metrics.load_error or "load_error"
            result.rejected += 1
            records.append(record)
            _move_file(audio_path, rejected_dir / filename, dry_run)
            continue

        # Populate record from metrics
        record.duration = metrics.duration
        record.snr_db = metrics.snr_db
        record.silence_ratio = metrics.silence_ratio
        record.clipping_ratio = metrics.clipping_ratio
        record.sample_rate = metrics.sample_rate
        record.quality_score = metrics.quality_score

        # Language detection
        if detector is not None:
            lang, confidence = detector.detect_language(str(audio_path))
        else:
            lang, confidence = "unknown", 0.0
        record.language = lang
        record.lang_confidence = confidence

        # Language filter (skip if detection was disabled)
        if not skip_lang_detect and not _is_language_accepted(lang, accepted_languages):
            metrics.rejection_reasons.append(
                f"rejected_language ({lang} not in accepted list)"
            )

        # Final decision
        if metrics.rejection_reasons:
            record.status = "rejected"
            record.rejection_reason = " | ".join(metrics.rejection_reasons)
            result.rejected += 1
            _move_file(audio_path, rejected_dir / filename, dry_run)
            logger.info(f"  REJECTED: {filename} | {record.rejection_reason}")
        else:
            record.status = "accepted"
            result.accepted += 1
            logger.info(
                f"  ACCEPTED: {filename} [{lang}] score={metrics.quality_score:.3f}"
            )

        records.append(record)

    result.duration_sec = round(time.time() - start_time, 2)

    # --- Write manifest ---
    manifest_path = output_dir / manifest_filename if not dry_run else log_dir / f"dry_run_{manifest_filename}"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(records, manifest_path)
    result.manifest_path = str(manifest_path)

    # --- Write run report ---
    report_path = log_dir / f"run_report_{int(start_time)}.txt"
    if not dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        _write_report(result, records, report_path, dry_run)
        result.report_path = str(report_path)

    logger.info(
        f"\n{'[DRY RUN] ' if dry_run else ''}Pipeline complete in {result.duration_sec}s | "
        f"Total: {result.total} | Accepted: {result.accepted} | Rejected: {result.rejected}"
    )
    logger.info(f"Manifest: {result.manifest_path}")

    return result


def _move_file(src: Path, dst: Path, dry_run: bool) -> None:
    """Move src to dst unless dry_run."""
    if dry_run:
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except Exception as e:
        logger.warning(f"[_move_file] Could not move {src} → {dst}: {e}")


def _write_manifest(records: List[FileRecord], path: Path) -> None:
    """Write the dataset_manifest.csv file."""
    fieldnames = [
        "filename", "language", "lang_confidence", "duration",
        "snr_db", "silence_ratio", "clipping_ratio", "sample_rate",
        "quality_score", "status", "rejection_reason",
    ]
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "filename": r.filename,
                    "language": r.language,
                    "lang_confidence": r.lang_confidence,
                    "duration": r.duration,
                    "snr_db": r.snr_db,
                    "silence_ratio": r.silence_ratio,
                    "clipping_ratio": r.clipping_ratio,
                    "sample_rate": r.sample_rate,
                    "quality_score": r.quality_score,
                    "status": r.status,
                    "rejection_reason": r.rejection_reason,
                })
        logger.info(f"Manifest written to: {path}")
    except Exception as e:
        logger.error(f"Failed to write manifest: {e}")


def _write_report(
    result: PipelineResult,
    records: List[FileRecord],
    path: Path,
    dry_run: bool,
) -> None:
    """Write a human-readable run report."""
    try:
        accept_rate = (result.accepted / result.total * 100) if result.total > 0 else 0
        lines = [
            "=" * 60,
            f"VoiceDataCurator — Run Report",
            f"{'[DRY RUN]' if dry_run else ''}",
            "=" * 60,
            f"  Total files processed : {result.total}",
            f"  Accepted              : {result.accepted} ({accept_rate:.1f}%)",
            f"  Rejected              : {result.rejected}",
            f"  Pipeline duration     : {result.duration_sec}s",
            f"  Manifest              : {result.manifest_path}",
            "",
            "Rejection breakdown:",
        ]

        rejected_records = [r for r in records if r.status == "rejected"]
        for r in rejected_records:
            lines.append(f"  {r.filename}: {r.rejection_reason}")

        lines += ["", "=" * 60]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Run report saved to: {path}")
    except Exception as e:
        logger.warning(f"Could not write run report: {e}")
