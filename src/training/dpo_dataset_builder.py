"""DPO preference dataset builder.

Constructs chosen/rejected response pairs for Direct Preference Optimization.
The core alignment goal: the model should refuse to hallucinate pricing,
protocol, or factual data — instead offering to connect the customer to
human support or referencing the knowledge base.

Hand-crafted preference templates are stored in
data/training/dpo_preference_templates.json to keep this module under the
300-line limit. All data is FICTIONAL and for demonstration purposes only.

Dataset composition (400 pairs):
  - 40% Pricing hallucination refusals
  - 30% Protocol/SOP hallucination refusals
  - 20% Out-of-scope query refusals
  - 10% Tone-correct refusals (right action, wrong tone)
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

random.seed(42)

# Path to the hand-crafted DPO preference templates
_DPO_TEMPLATES_PATH: str = "data/training/dpo_preference_templates.json"


def _load_preference_templates() -> dict[str, list[dict[str, str]]]:
    """Load hand-crafted DPO preference pair templates from JSON.

    Returns:
        Dict with keys: pricing_hallucination_pairs, protocol_hallucination_pairs,
        out_of_scope_pairs, tone_correction_pairs. Each value is a list of
        {"prompt": ..., "chosen": ..., "rejected": ...} dicts.
    """
    if os.path.exists(_DPO_TEMPLATES_PATH):
        with open(_DPO_TEMPLATES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    # Fallback: minimal inline templates if JSON file is missing
    return {
        "pricing_hallucination_pairs": [
            {
                "prompt": "What's the exact price for exosome proteomics?",
                "chosen": (
                    "I don't have the exact pricing readily available. I can "
                    "generate a draft quotation or connect you with our commercial "
                    "team. Would either work?"
                ),
                "rejected": (
                    "The price is ¥1,800 per sample, totaling ¥27,000 for 15 "
                    "samples with TMT 10-plex labeling."
                ),
            },
        ],
        "protocol_hallucination_pairs": [
            {
                "prompt": "What's your exact exosome extraction protocol?",
                "chosen": (
                    "Our protocols follow published MISEV guidelines. Our specific "
                    "optimizations are proprietary, but I can point you to the "
                    "published methods we base our workflow on."
                ),
                "rejected": (
                    "Sure! Centrifuge plasma at 2,500 × g for 15 min, then "
                    "ultracentrifuge at 100,000 × g for exactly 72 minutes."
                ),
            },
        ],
        "out_of_scope_pairs": [
            {
                "prompt": "Can you diagnose my disease from my exosome data?",
                "chosen": (
                    "I'm a technical assistant, not a clinical diagnostic tool. "
                    "Our reports are for research use only. Please consult your "
                    "physician for clinical interpretation."
                ),
                "rejected": (
                    "Based on your miRNA profile, this is consistent with an "
                    "early-stage malignancy. Follow up with an oncologist."
                ),
            },
        ],
        "tone_correction_pairs": [
            {
                "prompt": "I urgently need my results — can you rush them?",
                "chosen": (
                    "I understand the urgency. We offer expedited processing. "
                    "Could you share your project ID and deadline?"
                ),
                "rejected": (
                    "Sorry, no can do. Our pipeline runs on a fixed schedule."
                ),
            },
        ],
    }


def build_dpo_dataset(num_samples: int = 400) -> list[dict[str, Any]]:
    """Build the complete DPO preference dataset.

    Samples from each category according to the target distribution:
    40% pricing, 30% protocol, 20% out-of-scope, 10% tone correction.

    Args:
        num_samples: Total number of DPO pairs to generate.

    Returns:
        List of DPO training samples in the format:
        {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    templates = _load_preference_templates()
    pricing_pairs: list[dict[str, str]] = templates["pricing_hallucination_pairs"]
    protocol_pairs: list[dict[str, str]] = templates["protocol_hallucination_pairs"]
    oos_pairs: list[dict[str, str]] = templates["out_of_scope_pairs"]
    tone_pairs: list[dict[str, str]] = templates["tone_correction_pairs"]

    pricing_n: int = int(num_samples * 0.40)
    protocol_n: int = int(num_samples * 0.30)
    oos_n: int = int(num_samples * 0.20)
    tone_n: int = num_samples - pricing_n - protocol_n - oos_n

    dataset: list[dict[str, str]] = []

    # Sample with replacement from each category to reach target counts
    for _ in range(pricing_n):
        pair = random.choice(pricing_pairs)
        dataset.append(dict(pair))

    for _ in range(protocol_n):
        pair = random.choice(protocol_pairs)
        dataset.append(dict(pair))

    for _ in range(oos_n):
        pair = random.choice(oos_pairs)
        dataset.append(dict(pair))

    for _ in range(tone_n):
        pair = random.choice(tone_pairs)
        dataset.append(dict(pair))

    random.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict[str, Any]], filepath: str) -> None:
    """Save DPO dataset to a JSONL file.

    Args:
        dataset: List of DPO training samples.
        filepath: Output JSONL path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset = build_dpo_dataset(400)
    save_dataset(dataset, "data/training/dpo_preferences.jsonl")

    # Print estimated distribution
    cat_count: dict[str, int] = {
        "pricing": int(len(dataset) * 0.40),
        "protocol": int(len(dataset) * 0.30),
        "out_of_scope": int(len(dataset) * 0.20),
        "tone": len(dataset) - int(len(dataset) * 0.40) - int(len(dataset) * 0.30) - int(len(dataset) * 0.20),
    }
    print(f"Generated {len(dataset)} DPO preference pairs")
    for cat, count in cat_count.items():
        print(f"  {cat}: ~{count}")
