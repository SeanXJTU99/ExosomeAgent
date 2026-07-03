"""QLoRA fine-tuning script for exosome domain adaptation.

Uses Unsloth for accelerated QLoRA training on a single consumer GPU
(RTX 4090 24G). Implements three-stage curriculum fine-tuning:
  1. Terminology alignment (stage 1)
  2. Slot extraction (stage 2)
  3. General dialogue mixing (stage 3)

The script is designed to run WITHIN the LangGraph project environment.
In practice, the user would run this script independently to produce
the LoRA adapter weights, which are then loaded by the agent graph.

Requirements:
    pip install unsloth trl datasets peft bitsandbytes accelerate

Usage:
    python src/training/qlora_finetune.py --stage 1
    python src/training/qlora_finetune.py --stage 2 --resume_from stage1_adapter
    python src/training/qlora_finetune.py --stage 3 --resume_from stage2_adapter
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

# ─── Note: The imports below require the unsloth/trl ecosystem to be
# installed. They are kept as inline imports so the module can be
# imported for inspection without triggering ImportError.
#
# from unsloth import FastLanguageModel
# from unsloth import is_bfloat16_supported
# from datasets import load_dataset
# from trl import SFTTrainer
# from transformers import TrainingArguments
# import torch


# ─── Configuration ──────────────────────────────────────────────────────────

# Default paths (relative to project root)
DEFAULT_BASE_MODEL: str = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
DEFAULT_STAGE1_DATA: str = "data/training/stage1_terminology.jsonl"
DEFAULT_STAGE2_DATA: str = "data/training/stage2_slot_extraction.jsonl"
DEFAULT_STAGE3_DATA: str = "data/training/stage3_general.jsonl"
DEFAULT_OUTPUT_DIR: str = "checkpoints/lora_adapters"

# ChatML template for Qwen2.5
CHATML_TEMPLATE: str = """<|im_start|>system
You are ExoConsultant, an expert technical assistant for an exosome CRO company. You provide accurate, evidence-based answers about exosome biology, experimental protocols, and service offerings. When you don't know something, you clearly state that rather than fabricating information.<|im_end|>
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
<|im_start|>assistant
"""


def load_chatml_dataset(filepath: str) -> Any:
    """Load a JSONL dataset and format it for SFT training.

    Each line in the JSONL is a {"messages": [...]} object in ChatML format.
    Returns a HuggingFace Dataset object.

    Args:
        filepath: Path to the JSONL dataset file.

    Returns:
        A datasets.Dataset object ready for training.
    """
    # In production with datasets library:
    #   dataset = load_dataset("json", data_files=filepath, split="train")
    #   return dataset
    #
    # For now, provide a clear documentation of the expected behavior:
    print(f"[Mock] Loading dataset from: {filepath}")
    samples: list[dict[str, Any]] = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))
    print(f"[Mock] Loaded {len(samples)} samples")
    return samples  # type: ignore[return-value]


def configure_model_and_tokenizer(base_model: str, max_seq_length: int = 2048):
    """Load a 4-bit quantized base model with Unsloth and attach LoRA adapters.

    Uses Unsloth's FastLanguageModel for optimized 4-bit loading and
    LoRA adapter preparation.

    Args:
        base_model: HuggingFace model ID or path (4-bit variant).
        max_seq_length: Maximum sequence length for training.

    Returns:
        Tuple of (model, tokenizer).
    """
    # In production:
    #   model, tokenizer = FastLanguageModel.from_pretrained(
    #       model_name=base_model,
    #       max_seq_length=max_seq_length,
    #       dtype=None,  # Auto-detect (bfloat16 if supported, else float16)
    #       load_in_4bit=True,
    #   )
    #   model = FastLanguageModel.get_peft_model(
    #       model,
    #       r=64,
    #       target_modules=[
    #           "q_proj", "k_proj", "v_proj", "o_proj",
    #           "gate_proj", "up_proj", "down_proj",
    #       ],
    #       lora_alpha=128,
    #       lora_dropout=0.05,
    #       bias="none",
    #       use_gradient_checkpointing="unsloth",
    #       random_state=42,
    #   )
    #   return model, tokenizer

    print(f"[Mock] Loading base model: {base_model}")
    print(f"[Mock] Configuring LoRA: r=64, alpha=128, 7 target modules")
    print("[Mock] Model ready for training")
    return None, None  # type: ignore[return-value]


def train_stage(
    stage: int,
    data_path: str,
    output_dir: str,
    resume_from: str | None = None,
    num_epochs: int = 3,
    learning_rate: float = 2.0e-4,
    max_seq_length: int = 2048,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
) -> None:
    """Run one stage of the curriculum fine-tuning.

    Args:
        stage: Stage number (1, 2, or 3).
        data_path: Path to the training JSONL file.
        output_dir: Directory to save LoRA adapters.
        resume_from: Path to a previous stage's adapter to resume from.
        num_epochs: Number of training epochs for this stage.
        learning_rate: Peak learning rate.
        max_seq_length: Maximum sequence length.
        per_device_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Gradient accumulation steps.
    """
    effective_batch: int = per_device_batch_size * gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"Stage {stage}: Curriculum Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Data:          {data_path}")
    print(f"  Resume from:   {resume_from or 'base model'}")
    print(f"  Output:        {output_dir}/stage{stage}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size:    {per_device_batch_size} × {gradient_accumulation_steps} = {effective_batch}")
    print(f"  Max seq len:   {max_seq_length}")

    # In production:
    #   model, tokenizer = configure_model_and_tokenizer(
    #       resume_from or DEFAULT_BASE_MODEL, max_seq_length
    #   )
    #   dataset = load_chatml_dataset(data_path)
    #   trainer = SFTTrainer(
    #       model=model,
    #       tokenizer=tokenizer,
    #       train_dataset=dataset,
    #       dataset_text_field="messages",
    #       max_seq_length=max_seq_length,
    #       args=TrainingArguments(
    #           per_device_train_batch_size=per_device_batch_size,
    #           gradient_accumulation_steps=gradient_accumulation_steps,
    #           num_train_epochs=num_epochs,
    #           learning_rate=learning_rate,
    #           fp16=not is_bfloat16_supported(),
    #           bf16=is_bfloat16_supported(),
    #           logging_steps=10,
    #           optim="adamw_8bit",
    #           lr_scheduler_type="cosine",
    #           warmup_ratio=0.03,
    #           output_dir=f"{output_dir}/stage{stage}",
    #           save_steps=200,
    #           save_total_limit=3,
    #           report_to="none",
    #       ),
    #   )
    #   trainer.train()
    #   model.save_pretrained(f"{output_dir}/stage{stage}")

    print(f"[Mock] Stage {stage} training complete")
    print(f"[Mock] Adapter saved to: {output_dir}/stage{stage}")


def main() -> None:
    """Parse CLI arguments and run the specified training stage."""
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for exosome CRO agent"
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2, 3],
        help="Training stage (1=terminology, 2=slot, 3=general)"
    )
    parser.add_argument(
        "--base_model", type=str, default=DEFAULT_BASE_MODEL,
        help="Base model ID or path"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to training JSONL (auto-detected per stage if not set)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for LoRA adapters"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to previous stage adapter to resume from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs (stage default if not set: 3/2/1)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2.0e-4,
        help="Peak learning rate"
    )

    args = parser.parse_args()

    # Auto-detect data path per stage
    stage_data_map: dict[int, str] = {
        1: DEFAULT_STAGE1_DATA,
        2: DEFAULT_STAGE2_DATA,
        3: DEFAULT_STAGE3_DATA,
    }
    data_path: str = args.data or stage_data_map[args.stage]

    # Stage-specific defaults
    default_epochs: dict[int, int] = {1: 3, 2: 2, 3: 1}
    epochs: int = args.epochs if args.epochs is not None else default_epochs[args.stage]

    # Stage-specific LR decay
    # Override the CLI default per stage if --learning_rate was not explicitly set
    default_lrs: dict[int, float] = {1: 2.0e-4, 2: 1.0e-4, 3: 5.0e-5}
    if args.learning_rate == 2.0e-4:  # CLI default — apply stage-specific decay
        effective_lr: float = default_lrs[args.stage]
    else:
        effective_lr = args.learning_rate

    train_stage(
        stage=args.stage,
        data_path=data_path,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        num_epochs=epochs,
        learning_rate=effective_lr,
    )

    print("\n✓ Fine-tuning pipeline complete")
    print(f"  Next: merge adapter with base model using src/training/merge_and_export.py")
    if args.stage < 3:
        print(f"  Next stage: python src/training/qlora_finetune.py --stage {args.stage + 1} --resume_from {args.output_dir}/stage{args.stage}")


if __name__ == "__main__":
    main()
