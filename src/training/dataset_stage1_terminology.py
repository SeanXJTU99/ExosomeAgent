"""Stage 1 Dataset: Exosome terminology alignment.

Generates ~500 training samples that teach the base model exosome-specific
terminology, experimental protocols, and domain knowledge. The goal is to
align a general-purpose LLM with the vocabulary and concepts of extracellular
vesicle (EV) research.

Hand-crafted Q&A pairs are stored in data/training/terminology_samples.json
to keep this module under the 300-line limit. All data is FICTIONAL and for
demonstration purposes only.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

random.seed(42)

# Path to the hand-crafted terminology data
_TERMINOLOGY_DATA_PATH: str = "data/training/terminology_samples.json"


def _load_terminology_data() -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Load hand-crafted terminology samples and augmentation templates.

    Returns:
        Tuple of (terminology_samples, question_variants, topics).
    """
    if os.path.exists(_TERMINOLOGY_DATA_PATH):
        with open(_TERMINOLOGY_DATA_PATH, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return (
            data["terminology_samples"],
            data["question_variants"],
            data["topics"],
        )

    # Fallback: minimal inline data if JSON file is missing
    fallback_samples: list[dict[str, str]] = [
        {
            "user": "What is the best method to isolate exosomes from serum?",
            "assistant": (
                "For serum exosome isolation, the gold-standard method is "
                "ultracentrifugation (UC) combined with density gradient "
                "centrifugation at 100,000 × g for 70–90 minutes. SEC is "
                "recommended when vesicle integrity must be preserved for "
                "functional assays. Magnetic bead-based immunoaffinity capture "
                "offers higher purity but lower yield."
            ),
        },
    ]
    fallback_variants: list[str] = ["Can you tell me about {topic}?"]
    fallback_topics: list[str] = ["exosome isolation methods"]
    return fallback_samples, fallback_variants, fallback_topics


def generate_terminology_dataset(num_samples: int = 500) -> list[dict[str, Any]]:
    """Generate the stage 1 terminology alignment dataset.

    Combines hand-crafted high-quality Q&A pairs (loaded from external JSON)
    with template-augmented variants to reach the target sample count.
    Outputs in ChatML format.

    Args:
        num_samples: Target number of training samples.

    Returns:
        List of samples in ChatML conversation format.
    """
    samples, variants, topics = _load_terminology_data()

    dataset: list[dict[str, Any]] = []

    # Add hand-crafted samples first (highest quality)
    for sample in samples:
        dataset.append({
            "messages": [
                {"role": "user", "content": sample["user"]},
                {"role": "assistant", "content": sample["assistant"]},
            ]
        })

    # Augment with template-generated variants
    while len(dataset) < num_samples:
        topic = random.choice(topics)
        template = random.choice(variants)
        question = template.format(topic=topic)

        ref = random.choice(samples)
        answer = ref["assistant"]

        dataset.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        })

    random.shuffle(dataset)
    return dataset[:num_samples]


def save_dataset(dataset: list[dict[str, Any]], filepath: str) -> None:
    """Save dataset to a JSONL file in ChatML format.

    Args:
        dataset: List of training samples.
        filepath: Output JSONL path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset = generate_terminology_dataset(500)
    save_dataset(dataset, "data/training/stage1_terminology.jsonl")
    print(f"Generated {len(dataset)} terminology alignment samples")
