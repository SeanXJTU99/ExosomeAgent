"""LoRA adapter merge and model export script.

Merges QLoRA/SFT/DPO adapters back into the base model weights and
exports a single FP16 checkpoint file. This merged model is then
ready for AWQ quantization or direct vLLM deployment.

Supports two merge paths:
  1. SFT-only: base_model + stage3_lora → merged FP16
  2. SFT+DPO: base_model + stage3_lora + dpo_lora → merged FP16

Usage:
    # Merge SFT adapter only
    python src/training/merge_and_export.py \
        --base_model Qwen/Qwen2.5-14B-Instruct \
        --adapter_path checkpoints/lora_adapters/stage3 \
        --output_path checkpoints/merged_sft_fp16

    # Merge SFT + DPO adapters (sequential merge)
    python src/training/merge_and_export.py \
        --base_model Qwen/Qwen2.5-14B-Instruct \
        --adapter_path checkpoints/lora_adapters/stage3 \
        --dpo_adapter_path checkpoints/dpo_adapters \
        --output_path checkpoints/merged_dpo_fp16
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Any


# ─── Configuration ──────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL: str = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_SFT_ADAPTER: str = "checkpoints/lora_adapters/stage3"
DEFAULT_OUTPUT: str = "checkpoints/merged_fp16"


def merge_lora(
    base_model: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """Merge a single LoRA adapter into the base model.

    Args:
        base_model: HuggingFace model ID or local path.
        adapter_path: Path to the LoRA adapter checkpoint.
        output_path: Directory to save the merged model.
    """
    print(f"\n{'='*60}")
    print("LoRA Adapter Merge")
    print(f"{'='*60}")
    print(f"  Base model:    {base_model}")
    print(f"  Adapter:       {adapter_path}")
    print(f"  Output:        {output_path}")

    # In production:
    #   from peft import PeftModel
    #   from transformers import AutoModelForCausalLM, AutoTokenizer
    #   import torch
    #
    #   # Load base model in FP16
    #   base = AutoModelForCausalLM.from_pretrained(
    #       base_model,
    #       torch_dtype=torch.float16,
    #       device_map="auto",
    #       trust_remote_code=True,
    #   )
    #   tokenizer = AutoTokenizer.from_pretrained(
    #       base_model, trust_remote_code=True
    #   )
    #
    #   # Load and merge adapter
    #   model = PeftModel.from_pretrained(base, adapter_path)
    #   model = model.merge_and_unload()
    #
    #   # Save merged model
    #   model.save_pretrained(output_path, safe_serialization=True)
    #   tokenizer.save_pretrained(output_path)

    print(f"[Mock] Merged model saved to: {output_path}")


def merge_sequential(
    base_model: str,
    sft_adapter: str,
    dpo_adapter: str,
    output_path: str,
) -> None:
    """Merge SFT adapter first, then DPO adapter on top.

    This is the recommended path when both SFT and DPO training
    have been performed. The SFT adapter is merged first because
    DPO was trained on top of the SFT-adapted model.

    Args:
        base_model: HuggingFace model ID or local path.
        sft_adapter: Path to the SFT LoRA adapter (stage 3).
        dpo_adapter: Path to the DPO LoRA adapter.
        output_path: Directory to save the final merged model.
    """
    intermediate_path: str = output_path + "_sft_intermediate"

    print("\n[Step 1/2] Merging SFT adapter...")
    merge_lora(base_model, sft_adapter, intermediate_path)

    print("\n[Step 2/2] Merging DPO adapter on top of SFT...")
    # In production, load the SFT-merged model as the new base
    merge_lora(intermediate_path, dpo_adapter, output_path)

    # Clean up intermediate
    if os.path.exists(intermediate_path):
        shutil.rmtree(intermediate_path)
        print(f"[Mock] Cleaned up intermediate: {intermediate_path}")

    print(f"\n✓ Sequential merge complete → {output_path}")


def compute_model_size(output_path: str) -> dict[str, float]:
    """Estimate the disk size and VRAM requirements of the merged model.

    Args:
        output_path: Path to the merged model directory.

    Returns:
        Dict with size estimates in GB.
    """
    # In production, walk the directory and sum file sizes.
    # The 14B FP16 model is approximately:
    #   14B params × 2 bytes/param = 28 GB (weights only)
    #   + tokenizer, config ~ 50 MB
    # For INT4: 14B × 0.5 bytes/param = 7 GB

    return {
        "fp16_disk_gb": 28.0,
        "fp16_vram_gb": 28.0,
        "int4_vram_gb": 7.0,
        "int4_disk_gb": 7.5,
    }


def main() -> None:
    """Parse CLI arguments and run merge."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters for exosome CRO agent deployment"
    )
    parser.add_argument(
        "--base_model", type=str, default=DEFAULT_BASE_MODEL,
        help="Base model ID or path"
    )
    parser.add_argument(
        "--adapter_path", type=str, default=DEFAULT_SFT_ADAPTER,
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--dpo_adapter_path", type=str, default=None,
        help="Optional path to DPO adapter (sequential merge)"
    )
    parser.add_argument(
        "--output_path", type=str, default=DEFAULT_OUTPUT,
        help="Output directory for merged model"
    )

    args = parser.parse_args()

    if args.dpo_adapter_path:
        merge_sequential(
            base_model=args.base_model,
            sft_adapter=args.adapter_path,
            dpo_adapter=args.dpo_adapter_path,
            output_path=args.output_path,
        )
    else:
        merge_lora(
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            output_path=args.output_path,
        )

    # Report size estimates
    sizes = compute_model_size(args.output_path)
    print(f"\nModel size estimates:")
    print(f"  FP16: {sizes['fp16_vram_gb']:.1f} GB VRAM (weights only)")
    print(f"  INT4: {sizes['int4_vram_gb']:.1f} GB VRAM (after AWQ quantization)")
    print(f"\n  Next: quantize with src/deploy/quantize_awq.py")


if __name__ == "__main__":
    main()
