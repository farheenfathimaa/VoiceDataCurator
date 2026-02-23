"""
evaluate_wer.py
---------------
Step 4 of the whisper_finetuner pipeline.

Evaluates three Whisper models on the SAME held-out eval set:
  1. openai/whisper-tiny        — zero-shot baseline (no fine-tuning)
  2. ./whisper-finetuned-raw/   — fine-tuned on ALL data (curated + rejected)
  3. ./whisper-finetuned-curated/— fine-tuned on clean, curated data only

Computes Word Error Rate (WER) for each model and prints a comparison table.
Lower WER = better transcription accuracy.

Output:
  - Comparison table printed to stdout
  - wer_results.json saved to disk

Requirements:
    pip install evaluate jiwer transformers datasets

Usage:
    python evaluate_wer.py [--eval-data ./whisper_finetuner/data/curated_eval]
                           [--output wer_results.json]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


# ── Inference ──────────────────────────────────────────────────────────────────

def transcribe_dataset(
    model_path: str,
    eval_examples: List[Dict],
    is_local: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Run a Whisper model over all eval examples.

    Args:
        model_path:    HuggingFace model ID or local path to fine-tuned model
        eval_examples: List of dicts with 'audio_array', 'sampling_rate', 'text'
        is_local:      True if model_path is a local directory (fine-tuned model)

    Returns:
        (predictions, references) — parallel lists of strings
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    label = Path(model_path).name if is_local else model_path
    logger.info(f"Loading model: {label}")

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    # Use greedy decoding for deterministic, fast CPU inference
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe"
    )

    predictions = []
    references = []

    for i, example in enumerate(eval_examples):
        audio = np.array(example["audio_array"], dtype=np.float32)
        reference_text = example["text"]

        # Extract log-mel features
        input_features = processor.feature_extractor(
            audio,
            sampling_rate=example["sampling_rate"],
            return_tensors="pt",
        ).input_features  # shape: [1, 80, 3000]

        # Generate transcription (greedy, no beam search — fastest for demo)
        import torch
        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        # Decode tokens to text
        predicted_text = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        predictions.append(predicted_text)
        references.append(reference_text)

        if (i + 1) % 5 == 0 or (i + 1) == len(eval_examples):
            logger.info(f"  Transcribed {i+1}/{len(eval_examples)}")

    return predictions, references


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate using HuggingFace evaluate (backed by jiwer).

    WER = (substitutions + deletions + insertions) / total reference words
    Lower is better. 0.0 = perfect, 1.0+ = worse than random.
    """
    try:
        import evaluate
        wer_metric = evaluate.load("wer")
    except ImportError:
        raise SystemExit(
            "The 'evaluate' library is not installed.\n"
            "Run: pip install evaluate jiwer"
        )

    # Filter out empty predictions or references (avoid division by zero)
    valid_pairs = [
        (p, r) for p, r in zip(predictions, references)
        if p.strip() and r.strip()
    ]
    if not valid_pairs:
        logger.error("No valid prediction/reference pairs to evaluate.")
        return float("inf")

    preds, refs = zip(*valid_pairs)
    wer = wer_metric.compute(predictions=list(preds), references=list(refs))
    return float(wer)


def print_table(results: List[Dict]) -> None:
    """Print a clean WER comparison table to stdout."""
    print("\n" + "=" * 65)
    print(f"  {'Model':<40}  {'WER':>8}  {'vs Baseline':>12}")
    print("=" * 65)

    baseline_wer = None
    for row in results:
        if baseline_wer is None:
            baseline_wer = row["wer"]

        wer_pct = f"{row['wer'] * 100:.1f}%"

        if row["wer"] == baseline_wer:
            delta = "  (baseline)"
        else:
            improvement = (baseline_wer - row["wer"]) / baseline_wer * 100
            sign = "-" if improvement > 0 else "+"
            delta = f" {sign}{abs(improvement):.1f}% {'better' if improvement > 0 else 'worse'}"

        best_marker = "  <-- best" if row == min(results, key=lambda x: x["wer"]) else ""
        print(f"  {row['label']:<40}  {wer_pct:>8}  {delta:>12}{best_marker}")

    print("=" * 65)

    # Summary
    best = min(results, key=lambda x: x["wer"])
    worst = max(results, key=lambda x: x["wer"])
    if best["wer"] < worst["wer"]:
        improvement = (worst["wer"] - best["wer"]) / worst["wer"] * 100
        print(
            f"\n  Best model '{best['label']}' achieves {improvement:.1f}% lower WER "
            f"than worst '{worst['label']}'."
        )


# ── Entry Point ────────────────────────────────────────────────────────────────

def evaluate(
    eval_data_path: Path,
    curated_model_path: str,
    raw_model_path: str,
    base_model: str,
    output_json: Path,
) -> None:
    from datasets import load_from_disk

    # --- Load eval set ---
    if not eval_data_path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found: {eval_data_path}\n"
            "Run prepare_training_data.py first."
        )
    eval_dataset = load_from_disk(str(eval_data_path))
    eval_examples = list(eval_dataset)
    logger.info(f"Eval set: {len(eval_examples)} examples from {eval_data_path}")

    results = []

    # --- Model 1: Zero-shot baseline (no fine-tuning) ---
    print(f"\n[1/3] Evaluating: {base_model} (zero-shot baseline)")
    preds, refs = transcribe_dataset(base_model, eval_examples, is_local=False)
    wer = compute_wer(preds, refs)
    results.append({"label": f"{base_model} (zero-shot)", "wer": wer, "model_path": base_model})
    logger.info(f"  WER: {wer * 100:.1f}%")

    # --- Model 2: Fine-tuned on raw data ---
    print(f"\n[2/3] Evaluating: {raw_model_path} (fine-tuned on raw data)")
    if not Path(raw_model_path).exists():
        logger.warning(f"  Model not found: {raw_model_path} — skipping.")
        results.append({"label": "fine-tuned on raw data", "wer": float("inf"), "model_path": raw_model_path})
    else:
        preds, refs = transcribe_dataset(raw_model_path, eval_examples, is_local=True)
        wer = compute_wer(preds, refs)
        results.append({"label": "fine-tuned on raw data", "wer": wer, "model_path": raw_model_path})
        logger.info(f"  WER: {wer * 100:.1f}%")

    # --- Model 3: Fine-tuned on curated data ---
    print(f"\n[3/3] Evaluating: {curated_model_path} (fine-tuned on curated data)")
    if not Path(curated_model_path).exists():
        logger.warning(f"  Model not found: {curated_model_path} — skipping.")
        results.append({"label": "fine-tuned on curated data", "wer": float("inf"), "model_path": curated_model_path})
    else:
        preds, refs = transcribe_dataset(curated_model_path, eval_examples, is_local=True)
        wer = compute_wer(preds, refs)
        results.append({"label": "fine-tuned on curated data", "wer": wer, "model_path": curated_model_path})
        logger.info(f"  WER: {wer * 100:.1f}%")

    # --- Print table ---
    print_table(results)

    # --- Save JSON ---
    output_data = {
        "eval_set": str(eval_data_path),
        "num_eval_examples": len(eval_examples),
        "results": [
            {**r, "wer_pct": f"{r['wer'] * 100:.2f}%"}
            for r in results
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate WER for baseline, raw fine-tuned, and curated fine-tuned Whisper models."
    )
    parser.add_argument(
        "--eval-data",
        default="./whisper_finetuner/data/curated_eval",
        help="Path to the HuggingFace eval dataset (default: ./whisper_finetuner/data/curated_eval)",
    )
    parser.add_argument(
        "--curated-model",
        default="./whisper-finetuned-curated",
        help="Path to fine-tuned curated model directory",
    )
    parser.add_argument(
        "--raw-model",
        default="./whisper-finetuned-raw",
        help="Path to fine-tuned raw model directory",
    )
    parser.add_argument(
        "--base-model",
        default="openai/whisper-tiny",
        help="HuggingFace model ID for the zero-shot baseline",
    )
    parser.add_argument(
        "--output",
        default="./wer_results.json",
        help="Where to save the WER results JSON (default: ./wer_results.json)",
    )
    args = parser.parse_args()

    evaluate(
        eval_data_path=Path(args.eval_data),
        curated_model_path=args.curated_model,
        raw_model_path=args.raw_model,
        base_model=args.base_model,
        output_json=Path(args.output),
    )
