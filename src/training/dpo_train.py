"""DPO (Direct Preference Optimization) training script.

Fine-tunes the SFT model to prefer safe refusals over hallucinated answers.
Uses the TRL DPOTrainer with the preference pairs built by dpo_dataset_builder.py.

The core alignment goal:
  - chosen > rejected when:
    * chosen admits knowledge gaps and redirects to human/KB
    * rejected confidently fabricates prices, protocols, or medical advice

Requirements:
    pip install trl datasets peft accelerate

Usage:
    python src/training/dpo_train.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

# ─── Note: The imports below require the trl ecosystem.
# They are kept as inline documentation so the module can be inspected
# without triggering ImportError.
#
# from trl import DPOTrainer, DPOConfig
# from unsloth import FastLanguageModel
# from datasets import Dataset
# import torch


# ─── Configuration ──────────────────────────────────────────────────────────

DEFAULT_SFT_ADAPTER: str = "checkpoints/lora_adapters/stage3"
DEFAULT_DPO_DATA: str = "data/training/dpo_preferences.jsonl"
DEFAULT_OUTPUT_DIR: str = "checkpoints/dpo_adapters"

def load_dpo_dataset(filepath: str) -> Any:
    """Load DPO preference pairs from JSONL.

    Args:
        filepath: Path to the DPO preference JSONL file.

    Returns:
        A datasets.Dataset with 'prompt', 'chosen', 'rejected' columns.
    """
    print(f"[Mock] Loading DPO dataset from: {filepath}")
    samples: list[dict[str, str]] = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))
    print(f"[Mock] Loaded {len(samples)} preference pairs")

    # In production:
    #   dataset = Dataset.from_list(samples)
    #   return dataset
    return samples  # type: ignore[return-value]


def train_dpo(
    sft_adapter_path: str,
    dpo_data_path: str,
    output_dir: str,
    beta: float = 0.1,
    learning_rate: float = 5.0e-5,
    num_epochs: int = 1,
    max_length: int = 2048,
    max_prompt_length: int = 1024,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
) -> None:
    """Run DPO preference alignment training.

    Args:
        sft_adapter_path: Path to the SFT LoRA adapter.
        dpo_data_path: Path to DPO preference pairs JSONL.
        output_dir: Directory to save DPO adapters.
        beta: DPO temperature (lower = stronger preference signal).
        learning_rate: Peak learning rate (lower than SFT).
        num_epochs: Training epochs (DPO converges fast — 1 is usually enough).
        max_length: Maximum total sequence length (prompt + completion).
        max_prompt_length: Maximum prompt length.
        per_device_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Gradient accumulation steps.
    """
    effective_batch: int = per_device_batch_size * gradient_accumulation_steps
    print(f"\n{'='*60}")
    print("DPO Preference Alignment Training")
    print(f"{'='*60}")
    print(f"  SFT adapter:   {sft_adapter_path}")
    print(f"  DPO data:      {dpo_data_path}")
    print(f"  Output:        {output_dir}")
    print(f"  Beta:          {beta}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size:    {per_device_batch_size} × {gradient_accumulation_steps} = {effective_batch}")
    print(f"  Max length:    {max_length} (prompt ≤ {max_prompt_length})")

    # In production:
    #   model, tokenizer = FastLanguageModel.from_pretrained(
    #       model_name=sft_adapter_path,
    #       max_seq_length=max_length,
    #       dtype=None,
    #       load_in_4bit=True,
    #   )
    #
    #   dataset = load_dpo_dataset(dpo_data_path)
    #
    #   dpo_config = DPOConfig(
    #       beta=beta,
    #       per_device_train_batch_size=per_device_batch_size,
    #       gradient_accumulation_steps=gradient_accumulation_steps,
    #       num_train_epochs=num_epochs,
    #       learning_rate=learning_rate,
    #       bf16=True,
    #       logging_steps=5,
    #       optim="adamw_8bit",
    #       lr_scheduler_type="cosine",
    #       warmup_ratio=0.05,
    #       max_length=max_length,
    #       max_prompt_length=max_prompt_length,
    #       output_dir=output_dir,
    #       save_steps=100,
    #       save_total_limit=2,
    #       report_to="none",
    #       gradient_checkpointing=True,
    #   )
    #
    #   trainer = DPOTrainer(
    #       model=model,
    #       tokenizer=tokenizer,
    #       train_dataset=dataset,
    #       args=dpo_config,
    #   )
    #   trainer.train()
    #   model.save_pretrained(output_dir)

    print(f"[Mock] DPO training complete")
    print(f"[Mock] DPO adapter saved to: {output_dir}")


def main() -> None:
    """Parse CLI arguments and run DPO training."""
    parser = argparse.ArgumentParser(
        description="DPO preference alignment for exosome CRO agent"
    )
    parser.add_argument(
        "--sft_adapter", type=str, default=DEFAULT_SFT_ADAPTER,
        help="Path to SFT LoRA adapter"
    )
    parser.add_argument(
        "--dpo_data", type=str, default=DEFAULT_DPO_DATA,
        help="Path to DPO preference pairs JSONL"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for DPO adapter"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1,
        help="DPO temperature parameter"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5.0e-5,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs"
    )

    args = parser.parse_args()

    train_dpo(
        sft_adapter_path=args.sft_adapter,
        dpo_data_path=args.dpo_data,
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
    )

    print("\n✓ DPO alignment complete")
    print(f"  Next: merge adapter with src/training/merge_and_export.py")


if __name__ == "__main__":
    main()
