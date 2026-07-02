"""Stage 2 Dataset: SOP slot extraction.

Generates ~800 training samples that teach the model to extract structured
parameters (sample_type, sample_count, downstream_exp, qc_requirements)
from multi-turn exosome consultation dialogues.

Each sample is a multi-turn conversation ending with a JSON structured
output containing the extracted slots.

All data below is FICTIONAL and for demonstration purposes only.
"""

import json
import random
from typing import Any

random.seed(42)

# ─── Slot extraction scenario templates ────────────────────────────────────

# Each scenario defines a conversation pattern with expected slot outputs.
# The model is trained to produce the JSON at the end of the conversation.

SLOT_SCENARIOS: list[dict[str, Any]] = [
    {
        "conversation": [
            {"role": "user", "content": "Hi, I want to do exosome miRNA sequencing on rat serum samples. How much does it cost?"},
            {"role": "assistant", "content": "Hello! Rat serum exosome miRNA-seq is one of our standard services. To provide an accurate quote, may I ask: how many samples do you have, and what is the approximate volume per sample?"},
            {"role": "user", "content": "I have 12 samples, each about 1.5 mL of serum."},
            {"role": "assistant", "content": "Great, 1.5 mL per sample is within our recommended range. Do you need any specific QC assays, such as NTA, TEM, or Western Blot?"},
            {"role": "user", "content": "We'd like NTA and TEM, plus the CD63/CD81 markers by WB if possible."},
        ],
        "output": {
            "sample_type": "serum",
            "sample_count": 12,
            "downstream_exp": "mirna_seq",
            "qc_requirements": ["NTA", "TEM", "Western Blot"],
            "confidence_score": 0.95,
            "missing_slots": [],
        },
    },
    {
        "conversation": [
            {"role": "user", "content": "We have some cell culture supernatant that we want to analyze for exosomal proteins."},
            {"role": "assistant", "content": "Proteomics on cell supernatant exosomes — excellent. How many samples are you planning to submit, and what's the approximate culture volume per sample?"},
            {"role": "user", "content": "8 samples, 50 mL conditioned medium each. We're using HEK293T cells."},
            {"role": "assistant", "content": "50 mL is sufficient for UC-based isolation. Are you looking for label-free quantification or TMT-based multiplexing?"},
            {"role": "user", "content": "Label-free is fine. We just need NTA for QC — keeping costs down."},
        ],
        "output": {
            "sample_type": "cell_supernatant",
            "sample_count": 8,
            "downstream_exp": "proteomics",
            "qc_requirements": ["NTA"],
            "confidence_score": 0.90,
            "missing_slots": [],
        },
    },
    {
        "conversation": [
            {"role": "user", "content": "I need exosome isolation from urine samples for biomarker screening. Can you help?"},
            {"role": "assistant", "content": "Absolutely. Urine exosome biomarker discovery is a growing field. How many samples are in your cohort, and what downstream assay are you planning?"},
            {"role": "user", "content": "60 samples, we want to do miRNA-seq for a diagnostic biomarker panel."},
        ],
        "output": {
            "sample_type": "urine",
            "sample_count": 60,
            "downstream_exp": "mirna_seq",
            "qc_requirements": [],
            "confidence_score": 0.85,
            "missing_slots": ["qc_requirements"],
        },
    },
    {
        "conversation": [
            {"role": "user", "content": "Hello, can you extract exosomes from CSF?"},
            {"role": "assistant", "content": "Yes, we process CSF samples for exosome isolation. CSF typically has a very low exosome concentration, so we recommend at least 0.5 mL per sample. How many samples do you have, and what's your research goal?"},
            {"role": "user", "content": "We have 5 CSF samples, 1 mL each. We want to do total RNA-seq, and we need all three QC methods — NTA, TEM, and Western Blot."},
        ],
        "output": {
            "sample_type": "csf",
            "sample_count": 5,
            "downstream_exp": "rna_seq",
            "qc_requirements": ["NTA", "TEM", "Western Blot"],
            "confidence_score": 0.95,
            "missing_slots": [],
        },
    },
    {
        "conversation": [
            {"role": "user", "content": "Can I get a quote for exosome drug loading and targeting experiments?"},
            {"role": "assistant", "content": "Certainly! Functional exosome engineering is one of our specialized services. To prepare a quote, I need to know: what is your source material (serum, cell supernatant, etc.), how many samples, and what cargo are you loading?"},
            {"role": "user", "content": "We'll be using cell supernatant from dendritic cells, about 20 samples. We're loading doxorubicin for tumor targeting studies."},
            {"role": "assistant", "content": "Thank you. For drug-loaded exosomes, we also recommend comprehensive QC including NTA and TEM to confirm vesicle integrity after loading. Would you like those included?"},
            {"role": "user", "content": "Yes, include both NTA and TEM."},
        ],
        "output": {
            "sample_type": "cell_supernatant",
            "sample_count": 20,
            "downstream_exp": "drug_delivery",
            "qc_requirements": ["NTA", "TEM"],
            "confidence_score": 0.90,
            "missing_slots": [],
        },
    },
]

# Augmentation templates for generating variants
SAMPLE_TYPES: list[tuple[str, str]] = [
    ("serum", "serum"),
    ("plasma", "plasma"),
    ("cell_supernatant", "cell culture supernatant"),
    ("urine", "urine"),
    ("csf", "cerebrospinal fluid"),
    ("saliva", "saliva"),
]

DOWNSTREAM_LABELS: list[tuple[str, str]] = [
    ("mirna_seq", "miRNA sequencing"),
    ("rna_seq", "total RNA-seq"),
    ("proteomics", "proteomics / mass spectrometry"),
    ("drug_delivery", "drug delivery / targeting"),
    ("functional", "functional validation"),
    ("biomarker", "biomarker discovery"),
]

QC_OPTIONS: list[list[str]] = [
    ["NTA"],
    ["NTA", "TEM"],
    ["NTA", "TEM", "Western Blot"],
    [],
]


def _generate_variant_conversation() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Generate a randomized slot-filling conversation and its expected output.

    Returns:
        Tuple of (conversation_messages, expected_slot_output).
    """
    stype_key, stype_name = random.choice(SAMPLE_TYPES)
    dexp_key, dexp_name = random.choice(DOWNSTREAM_LABELS)
    qc_list: list[str] = random.choice(QC_OPTIONS)
    count: int = random.choice([3, 5, 8, 10, 12, 15, 20, 30, 50])

    # Build conversation
    conv: list[dict[str, str]] = []

    # Opening (user states need)
    openings: list[str] = [
        f"I'm interested in exosome {dexp_name} from {stype_name} samples.",
        f"Do you offer {dexp_name} services for exosomes from {stype_name}?",
        f"Can you help with exosome isolation from {stype_name} for {dexp_name}?",
        f"I need {dexp_name} analysis of {stype_name}-derived exosomes.",
    ]
    conv.append({"role": "user", "content": random.choice(openings)})

    # Assistant acknowledgement + sample count inquiry
    ack_responses: list[str] = [
        f"Certainly! {dexp_name} on {stype_name} exosomes is within our capabilities. How many samples are you planning to submit?",
        f"Sure. We process {stype_name} exosomes for {dexp_name} regularly. How many samples in your study?",
        f"Of course. To prepare your service quote, I'll need the sample count first. How many {stype_name} samples?",
    ]
    conv.append({"role": "assistant", "content": random.choice(ack_responses)})

    # User provides count
    count_responses: list[str] = [
        f"I have {count} samples.",
        f"We've collected {count} samples so far.",
        f"Approximately {count} samples in total.",
        f"{count} samples, each properly stored at -80°C.",
    ]
    conv.append({"role": "user", "content": random.choice(count_responses)})

    # Assistant asks about QC if not specified
    if not qc_list:
        conv.append({
            "role": "assistant",
            "content": (
                f"Noted — {count} samples. One more thing: do you have specific QC "
                "requirements (NTA, TEM, Western Blot), or would you like our "
                "standard package recommendation?"
            ),
        })
        conv.append({
            "role": "user",
            "content": "Standard package is fine — whatever you recommend.",
        })

    # Build expected output
    missing: list[str] = []
    if not qc_list:
        missing.append("qc_requirements")

    output: dict[str, Any] = {
        "sample_type": stype_key,
        "sample_count": count,
        "downstream_exp": dexp_key,
        "qc_requirements": qc_list,
        "confidence_score": round(random.uniform(0.82, 0.98), 2),
        "missing_slots": missing,
    }

    return conv, output


def generate_slot_extraction_dataset(num_samples: int = 800) -> list[dict[str, Any]]:
    """Generate the stage 2 slot extraction dataset.

    Combines hand-crafted high-quality dialogue scenarios with
    programmatically-generated variants to reach the target size.

    Args:
        num_samples: Target number of training samples.

    Returns:
        List of samples in ChatML format with JSON-structured assistant output.
    """
    dataset: list[dict[str, Any]] = []

    # Add hand-crafted scenarios first
    for scenario in SLOT_SCENARIOS:
        # The assistant's final turn should contain the JSON output
        conversation = list(scenario["conversation"])
        output_json: str = json.dumps(scenario["output"], ensure_ascii=False, indent=2)
        conversation.append({
            "role": "assistant",
            "content": (
                "Here is a summary of the extracted service parameters:\n\n"
                f"```json\n{output_json}\n```"
            ),
        })
        dataset.append({"messages": conversation})

    # Augment with generated variants
    while len(dataset) < num_samples:
        conv, output = _generate_variant_conversation()
        output_json = json.dumps(output, ensure_ascii=False, indent=2)
        conv.append({
            "role": "assistant",
            "content": (
                "Based on our conversation, here are the structured parameters:\n\n"
                f"```json\n{output_json}\n```"
            ),
        })
        dataset.append({"messages": conv})

    random.shuffle(dataset)
    return dataset[:num_samples]


def save_dataset(dataset: list[dict[str, Any]], filepath: str) -> None:
    """Save dataset to a JSONL file.

    Args:
        dataset: List of training samples.
        filepath: Output JSONL path.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset = generate_slot_extraction_dataset(800)
    save_dataset(dataset, "data/training/stage2_slot_extraction.jsonl")
    print(f"Generated {len(dataset)} slot extraction samples")
