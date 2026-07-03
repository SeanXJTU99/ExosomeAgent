"""VRAM budget calculator for RTX 4090 24G deployment.

Computes the memory breakdown for running Qwen2.5-14B-AWQ-INT4
with vLLM PagedAttention on a single consumer GPU. Used to verify
that the model fits within hardware constraints before deployment.
"""

from __future__ import annotations


def compute_vram_budget(
    model_params_b: float = 14.0,
    quant_bits: int = 4,
    max_context_len: int = 8192,
    num_kv_heads: int = 40,
    num_layers: int = 40,
    head_dim: int = 128,
    overhead_gb: float = 4.0,
) -> dict[str, float]:
    """Calculate VRAM requirements for the full deployment.

    Args:
        model_params_b: Model parameter count in billions (14.0 for 14B).
        quant_bits: Quantization bit width (4 for INT4).
        max_context_len: Maximum context length (tokens).
        num_kv_heads: Number of KV heads (40 for Qwen2.5-14B).
        num_layers: Number of transformer layers (40 for Qwen2.5-14B).
        head_dim: Dimension per attention head (128 for Qwen2.5-14B).
        overhead_gb: Runtime overhead (vLLM, LangGraph, Python, CUDA context).

    Returns:
        Dict with VRAM breakdown in GB.
    """
    # Model weights
    bytes_per_param: float = quant_bits / 8.0
    model_weights_gb: float = model_params_b * bytes_per_param
    # Add ~7% overhead for quantization scales and metadata
    model_weights_gb *= 1.07

    # KV Cache (PagedAttention)
    # Formula: 2 × num_layers × num_kv_heads × head_dim × max_context_len × 2 bytes
    kv_cache_bytes: float = (
        2 * num_layers * num_kv_heads * head_dim * max_context_len * 2
    )
    kv_cache_gb: float = kv_cache_bytes / (1024 ** 3)

    # Total
    total_gb: float = model_weights_gb + kv_cache_gb + overhead_gb

    return {
        "model_weights_gb": round(model_weights_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "runtime_overhead_gb": round(overhead_gb, 2),
        "total_gb": round(total_gb, 2),
        "headroom_gb": round(24.0 - total_gb, 2),
    }


def print_budget_table(budget: dict[str, float]) -> None:
    """Print a formatted VRAM budget table.

    Args:
        budget: VRAM budget dict from compute_vram_budget().
    """
    gpu_total: float = 24.0
    print(f"\n{'='*50}")
    print(f"VRAM Budget — RTX 4090 24G")
    print(f"{'='*50}")
    print(f"  {'Component':<28} {'VRAM':>10}")
    print(f"  {'-'*28} {'-'*10}")
    print(f"  {'Model weights (INT4)':<28} {budget['model_weights_gb']:>7.2f} GB")
    print(f"  {'KV Cache (8K ctx)':<28} {budget['kv_cache_gb']:>7.2f} GB")
    print(f"  {'Runtime overhead':<28} {budget['runtime_overhead_gb']:>7.2f} GB")
    print(f"  {'-'*28} {'-'*10}")
    print(f"  {'TOTAL':<28} {budget['total_gb']:>7.2f} GB")
    print(f"  {'GPU capacity':<28} {gpu_total:>7.2f} GB")
    print(f"  {'Headroom':<28} {budget['headroom_gb']:>7.2f} GB")

    # Verdict
    if budget["headroom_gb"] > 2.0:
        verdict: str = "[OK] FITS comfortably"
    elif budget["headroom_gb"] > 0:
        verdict = "[WARN] FITS but tight - reduce concurrency"
    else:
        verdict = "[FAIL] DOES NOT FIT - reduce context length or model size"

    print(f"\n  Verdict: {verdict}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Default: Qwen2.5-14B AWQ INT4 on RTX 4090
    budget = compute_vram_budget(
        model_params_b=14.0,
        quant_bits=4,
        max_context_len=8192,
        num_kv_heads=40,
        num_layers=40,
        head_dim=128,
        overhead_gb=4.0,
    )
    print_budget_table(budget)

    # Alternative: Qwen2.5-7B AWQ INT4
    print("Alternative configuration:")
    budget_7b = compute_vram_budget(
        model_params_b=7.0,
        quant_bits=4,
        max_context_len=8192,
        num_kv_heads=32,
        num_layers=32,
        head_dim=128,
        overhead_gb=3.0,
    )
    print_budget_table(budget_7b)
