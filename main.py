"""
main.py
-------
CLI entrypoint for VoiceDataCurator.

Usage examples:
    python main.py --input ./data/raw
    python main.py --input ./data/raw --dry-run
    python main.py --input ./data/raw --config my_config.yaml --verbose
    python main.py --input ./data/raw --output ./out --rejected ./bad

Run the dashboard separately:
    streamlit run dashboard.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml


def setup_logging(verbose: bool, log_dir: Path, write_to_file: bool) -> None:
    """Configure root logger to console (and optionally file)."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    if write_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"run_{int(time.time())}.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


def load_config(config_path: str) -> dict:
    """Load and parse YAML config file."""
    path = Path(config_path)
    if not path.exists():
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="VoiceDataCurator",
        description=(
            "VoiceDataCurator - Multilingual Speech Quality Analyzer & Dataset Curator\n\n"
            "Ingest raw audio, run quality checks, detect language, filter "
            "low-quality samples, and export a clean dataset manifest."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to folder containing raw audio files (WAV/MP3/FLAC/OGG). "
             "Overrides config.yaml input_dir.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for accepted files and manifest CSV. "
             "Overrides config.yaml output_dir.",
    )
    parser.add_argument(
        "--rejected",
        type=str,
        default=None,
        help="Directory to move rejected files into. "
             "Overrides config.yaml rejected_dir.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview filtering results without moving any files.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging output.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Override Whisper model size (tiny/base/small/medium/large).",
    )
    parser.add_argument(
        "--no-language-detect",
        action="store_true",
        help="Skip Whisper language detection (faster, tags all files as 'unknown').",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load config ---
    config = load_config(args.config)

    # --- CLI overrides ---
    pipeline_cfg = config.setdefault("pipeline", {})
    lang_cfg = config.setdefault("language", {})
    log_cfg = config.setdefault("logging", {})

    if args.input:
        pipeline_cfg["input_dir"] = args.input
    if args.output:
        pipeline_cfg["output_dir"] = args.output
    if args.rejected:
        pipeline_cfg["rejected_dir"] = args.rejected
    if args.whisper_model:
        lang_cfg["whisper_model"] = args.whisper_model
    if args.no_language_detect:
        lang_cfg["skip_detection"] = True

    # --- Setup logging ---
    log_dir = Path(pipeline_cfg.get("log_dir", "./logs"))
    write_log_file = log_cfg.get("write_to_file", True)
    setup_logging(verbose=args.verbose, log_dir=log_dir, write_to_file=write_log_file)

    logger = logging.getLogger(__name__)

    # --- Banner ---
    banner = """
+====================================================+
|        VoiceDataCurator  v1.0                      |
|  Multilingual Speech Quality Analyzer & Curator    |
+====================================================+
"""
    print(banner)

    if args.dry_run:
        logger.info("=== DRY RUN MODE: No files will be moved ===")

    logger.info(f"Config file  : {args.config}")
    logger.info(f"Input dir    : {pipeline_cfg.get('input_dir', 'NOT SET')}")
    logger.info(f"Output dir   : {pipeline_cfg.get('output_dir', './output')}")
    logger.info(f"Rejected dir : {pipeline_cfg.get('rejected_dir', './rejected')}")
    logger.info(f"Whisper model: {lang_cfg.get('whisper_model', 'base')}")

    # --- Validate input ---
    input_dir = pipeline_cfg.get("input_dir")
    if not input_dir:
        logger.error(
            "No input directory specified. Use --input <path> or set pipeline.input_dir in config.yaml"
        )
        sys.exit(1)

    if not Path(input_dir).exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # --- Import pipeline here (after logging is set up) ---
    from pipeline import run_pipeline

    try:
        result = run_pipeline(config=config, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Pipeline failed with unexpected error: {e}")
        sys.exit(1)

    # --- Final summary ---
    print("\n" + "=" * 56)
    print(f"  {'[DRY RUN] ' if args.dry_run else ''}Run Complete")
    print("=" * 56)
    print(f"  Total files  : {result.total}")
    print(f"  [ACCEPTED]   : {result.accepted}")
    print(f"  [REJECTED]   : {result.rejected}")
    print(f"  Duration     : {result.duration_sec}s")
    if result.manifest_path:
        print(f"  Manifest     : {result.manifest_path}")
    if result.report_path:
        print(f"  Report       : {result.report_path}")
    print("=" * 56)
    print("\nTo view the dashboard:")
    print("  streamlit run dashboard.py\n")


if __name__ == "__main__":
    main()
