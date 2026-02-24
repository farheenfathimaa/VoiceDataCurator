"""
finetune_whisper.py
-------------------
Step 3 of the whisper_finetuner pipeline.

Fine-tunes openai/whisper-tiny on TWO training sets:
  1. Curated set (accepted-only files from VoiceDataCurator)
       -> saved to ./whisper-finetuned-curated/
  2. Raw baseline set (all files, including rejected low-quality ones)
       -> saved to ./whisper-finetuned-raw/

Both runs use identical hyperparameters so the only variable is
data quality — demonstrating that VoiceDataCurator's curation improves
downstream model performance (measured by WER in evaluate_wer.py).

Requirements:
    pip install transformers datasets accelerate

Usage:
    python finetune_whisper.py [--data-dir ./whisper_finetuner/data]
                               [--epochs 3]
                               [--batch-size 4]
                               [--model openai/whisper-tiny]
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence noisy HuggingFace/tokenizers warnings
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


# ── Data Collator ──────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Collates a batch of speech examples for Seq2Seq training.

    - Pads input_features (mel spectrograms) to the same length
    - Pads label IDs, replacing padding token with -100 so loss ignores them
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # --- Input features (mel spectrograms) ---
        # Each has shape [n_mels, time_steps]; pad along time dimension
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # --- Labels (token IDs) ---
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id (-1 from tokenizer) with -100
        # so the cross-entropy loss correctly ignores padding positions
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If the beginning-of-sequence token was added, strip it from labels
        # (Whisper's decoder doesn't need BOS in the label sequence)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── Feature Extraction ─────────────────────────────────────────────────────────

def prepare_dataset(batch: Dict, processor) -> Dict:
    """
    Map function applied to each dataset example.

    Converts raw audio arrays -> log-mel spectrograms (input_features)
    Converts reference text -> token IDs (labels)
    """
    audio = np.array(batch["audio_array"], dtype=np.float32)

    # Extract log-mel spectrogram features; truncate/pad to 30 seconds
    batch["input_features"] = processor.feature_extractor(
        audio,
        sampling_rate=batch["sampling_rate"],
        return_tensors="np",
    ).input_features[0]  # shape: [80, 3000]

    # Tokenize the reference transcript
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch


# ── Fine-tuning ────────────────────────────────────────────────────────────────

def run_finetuning(
    train_data_path: Path,
    output_dir: Path,
    model_name: str,
    num_train_epochs: int,
    per_device_batch_size: int,
    run_label: str,  # e.g. "curated" or "raw" — for logging
) -> None:
    """
    Fine-tune whisper-tiny on the given training split and save the model.

    Args:
        train_data_path:       Path to the HF dataset saved by prepare_training_data.py
        output_dir:            Where to save checkpoints and final model
        model_name:            HuggingFace model ID (default: openai/whisper-tiny)
        num_train_epochs:      How many full passes through the training data
        per_device_batch_size: Batch size per GPU/CPU device
        run_label:             "curated" or "raw" — used in log messages only
    """
    from datasets import load_from_disk
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f" Fine-tuning on {run_label.upper()} data -> {output_dir}")
    logger.info(f"{'='*60}")

    # --- Load dataset ---
    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_data_path}\n"
            "Run prepare_training_data.py first."
        )
    train_dataset = load_from_disk(str(train_data_path))
    logger.info(f"Training examples: {len(train_dataset)}")

    # --- Load processor (feature extractor + tokenizer) ---
    logger.info(f"Loading processor from '{model_name}'...")
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language="english",   # set to None to let model infer language
        task="transcribe",
    )

    # --- Load model ---
    logger.info(f"Loading model '{model_name}'...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Disable caching during training (incompatible with gradient checkpointing)
    model.config.use_cache = False

    # --- Generation Config (transformers v5.x handles these here) ---
    # Tell model to predict transcriptions (not translations)
    if model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
    else:
        # Fallback for older transformers if generation_config is missing
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

    # --- Preprocess dataset: audio -> features, text -> tokens ---
    logger.info("Preprocessing dataset (extracting mel features + tokenizing)...")
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=["audio_array", "sampling_rate", "text", "filename", "status"],
        desc="Feature extraction",
    )

    # --- Data collator ---
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # --- Training arguments ---
    # fp16=False — safe for CPU training; set True only if NVIDIA GPU available
    # gradient_checkpointing=False — simpler for small datasets; saves GPU memory when True
    # predict_with_generate=False during training (we just minimize cross-entropy loss)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),

        # --- Core training config ---
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,

        # --- Optimizer ---
        learning_rate=1e-5,             # small LR for fine-tuning (avoid catastrophic forgetting)
        warmup_steps=10,                # warm up over first 10 steps
        weight_decay=0.01,

        # --- Hardware ---
        fp16=False,                     # set True if CUDA GPU with float16 support is available
        gradient_checkpointing=False,   # set True to reduce GPU memory (slightly slower)

        # --- Logging ---
        logging_steps=5,
        logging_dir=str(output_dir / "logs"),
        report_to="none",               # disable W&B / TensorBoard by default

        # --- Checkpointing ---
        save_strategy="epoch",          # save one checkpoint per epoch
        save_total_limit=1,             # keep only the best checkpoint (saves disk space)
        load_best_model_at_end=False,   # no eval during training loop (eval in evaluate_wer.py)

        # --- Generation (not used during training, only relevant for predict) ---
        predict_with_generate=False,
    )

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
    )

    # --- Train ---
    logger.info(f"Starting training ({num_train_epochs} epoch(s))...")
    trainer.train()

    # --- Save final model + processor ---
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info(f"Model saved to: {output_dir}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune whisper-tiny on curated and raw training sets."
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="./whisper_finetuner/data",
        help="Directory with HF datasets (default: ./whisper_finetuner/data)",
    )
    parser.add_argument(
        "--curated-output",
        default="./whisper-finetuned-curated",
        help="Where to save the curated fine-tuned model",
    )
    parser.add_argument(
        "--raw-output",
        default="./whisper-finetuned-raw",
        help="Where to save the raw fine-tuned model",
    )
    parser.add_argument(
        "--model",
        default="openai/whisper-tiny",
        help="Base Whisper model to fine-tune (default: openai/whisper-tiny)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4)",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip raw baseline fine-tuning (run curated only)",
    )
    parser.add_argument(
        "--skip-curated",
        action="store_true",
        help="Skip curated fine-tuning (run raw baseline only)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # --- Run 1: Fine-tune on curated (high-quality) data ---
    if not args.skip_curated:
        run_finetuning(
            train_data_path=data_dir / "curated_train",
            output_dir=Path(args.curated_output),
            model_name=args.model,
            num_train_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            run_label="curated",
        )

    # --- Run 2: Fine-tune on raw (curated + rejected) data ---
    if not args.skip_raw:
        run_finetuning(
            train_data_path=data_dir / "raw_train",
            output_dir=Path(args.raw_output),
            model_name=args.model,
            num_train_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            run_label="raw",
        )

    print(
        "\nFine-tuning complete!"
        "\nNext step: python evaluate_wer.py"
    )
