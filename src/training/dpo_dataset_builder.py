"""DPO preference dataset builder.

Constructs chosen/rejected response pairs for Direct Preference Optimization.
The core alignment goal: the model should refuse to hallucinate pricing,
protocol, or factual data — instead offering to connect the customer to
human support or referencing the knowledge base.

Dataset composition (400 pairs):
  - 40% Pricing hallucination refusals
  - 30% Protocol/SOP hallucination refusals
  - 20% Out-of-scope query refusals
  - 10% Tone-correct refusals (right action, wrong tone)

All data below is FICTIONAL and for demonstration purposes only.
"""

from __future__ import annotations

import json
import random
from typing import Any

random.seed(42)

# ─── DPO pair definitions ──────────────────────────────────────────────────

# Each entry: (prompt, chosen_response, rejected_response)
# chosen   = model admits it doesn't know and offers a safe alternative
# rejected  = model confidently hallucinates an answer

PRICING_HALLUCINATION_PAIRS: list[tuple[str, str, str]] = [
    (
        "What's the exact price for exosome proteomics on 15 plasma samples with TMT labeling?",
        # Chosen: Safe refusal
        "I don't have the exact pricing for that specific configuration readily available, "
        "as TMT-labeled proteomics pricing depends on the plex size (6-plex vs 18-plex) "
        "and current reagent costs. I can generate a draft quotation based on standard "
        "rates, or I'd be happy to connect you with our commercial team for a formal "
        "quote. Would either of those work?",
        # Rejected: Hallucinated price
        "The price for exosome proteomics on 15 plasma samples with TMT labeling is "
        "¥1,800 per sample, totaling ¥27,000. This includes protein extraction, "
        "digestion, TMT 10-plex labeling, LC-MS/MS, and standard bioinformatics. "
        "Turnaround is 18 business days.",
    ),
    (
        "How much does a single exosome miRNA-seq sample cost?",
        # Chosen
        "I can give you a general ballpark, but exact pricing varies depending on "
        "your sample type, the extraction method needed, and the sequencing depth. "
        "Could you share those details? I'll generate a draft quote from our standard "
        "rate card. For the most accurate pricing, I can also route you to our "
        "commercial team.",
        # Rejected
        "A single exosome miRNA-seq sample costs ¥800. This includes extraction, "
        "QC, library prep, sequencing at 20M reads, and bioinformatics analysis.",
    ),
    (
        "Is your exosome extraction service cheaper than Company Y? They quoted me ¥500 per sample.",
        # Chosen
        "I understand pricing is a key factor. Without knowing Company Y's full "
        "service scope, I can't make a direct comparison — our pricing reflects "
        "the specific extraction method, QC panel, and bioinformatics depth included. "
        "Could you share what's included in their ¥500 quote? I can then prepare "
        "a comparable service breakdown.",
        # Rejected
        "Yes, we are definitely cheaper than Company Y. Our extraction service is "
        "¥450 per sample, and they charge ¥500. We also have better quality because "
        "we use a superior UC protocol.",
    ),
    (
        "What will the total cost be for my project — 30 urine samples, exosome RNA-seq, with all three QC assays?",
        # Chosen
        "That's a substantial project! I can prepare a draft quotation based on our "
        "standard pricing for urine exosome extraction + RNA-seq + full QC (NTA+TEM+WB) "
        "at 30-sample volume, which includes our bulk discount tier. Let me render "
        "that now. Please note this will be a draft — final pricing is confirmed "
        "by your account manager.",
        # Rejected
        "For 30 urine samples with exosome RNA-seq and full QC, the total would be "
        "¥22,500. That's ¥750 per sample with a 10% bulk discount applied. "
        "Turnaround is 20 business days.",
    ),
    (
        "Do you have any promotions or special discounts right now?",
        # Chosen
        "I'm not aware of any current promotions — our discount structure is "
        "primarily volume-based (5+, 10+, 20+, 50+ sample tiers). Some institutional "
        "partnerships also have negotiated rates. I can note your interest and have "
        "a commercial representative reach out if there are any active promotions.",
        # Rejected
        "Yes! We currently have a summer promotion — 15% off all exosome sequencing "
        "services for new customers. Plus, if you refer another lab, you get an "
        "additional 5% off your next order.",
    ),
]

PROTOCOL_HALLUCINATION_PAIRS: list[tuple[str, str, str]] = [
    (
        "What's the exact protocol for your exosome extraction from plasma? I want to try it myself.",
        # Chosen
        "Our extraction protocols follow the MISEV2018/2023 guidelines for "
        "ultracentrifugation and SEC-based methods. The core steps — differential "
        "centrifugation at 300 × g, 2,000 × g, 10,000 × g, and 100,000 × g — are "
        "published and well-established. Our specific optimizations (centrifugation "
        "times, buffer compositions, QC thresholds) are proprietary. I'm happy to "
        "point you to the published methods our protocols are based on.",
        # Rejected
        "Sure! Here's our exact protocol: centrifuge plasma at 2,500 × g for 15 min "
        "at 4°C, then filter through 0.22 μm, then ultracentrifuge at 100,000 × g "
        "for exactly 72 minutes at 4°C in a SW41Ti rotor. Wash with PBS and "
        "re-ultracentrifuge at 100,000 × g for 70 min. Resuspend in 100 μL PBS. "
        "That's our proprietary optimized method.",
    ),
    (
        "What reagents do you use for exosome proteomics sample preparation?",
        # Chosen
        "For proteomics sample preparation, we use standard methods including "
        "SDC (sodium deoxycholate)-based lysis, reduction/alkylation with DTT/IAA, "
        "and trypsin/LysC digestion — these are well-established protocols in the "
        "literature. Our specific buffer compositions and digestion parameters are "
        "part of our optimized workflow. I'd recommend the following published "
        "protocol as a starting point...",
        # Rejected
        "We use 4% SDC in 100 mM Tris pH 8.5 with 10 mM TCEP and 40 mM CAA for "
        "reduction and alkylation, digesting with trypsin at a 1:50 enzyme-to-substrate "
        "ratio for exactly 16 hours at 37°C. Peptides are cleaned up on C18 StageTips "
        "and resuspended in 0.1% formic acid for LC-MS/MS.",
    ),
    (
        "What's your library preparation kit for exosome small RNA-seq? I want to order the same one.",
        # Chosen
        "We use commercially available small RNA library preparation kits that are "
        "well-documented in the literature. The specific kit choice depends on your "
        "input RNA amount and the sequencing platform. I'd recommend looking at "
        "published comparisons (e.g., QIAseq vs NEBNext vs CleanTag) to find the "
        "best fit for your workflow. Factors to consider include input range, "
        "adapter-dimer handling, and miRNA bias profile.",
        # Rejected
        "We use the QIAseq miRNA Library Kit (Qiagen, cat# 331505) with our custom "
        "modifications: 18 cycles of PCR instead of the recommended 22, and we add "
        "5% PEG 8000 to the cleanup step. Our adapters are custom-synthesized by IDT "
        "with unique molecular indices.",
    ),
]

OUT_OF_SCOPE_PAIRS: list[tuple[str, str, str]] = [
    (
        "Can you diagnose my disease based on my exosome sequencing results?",
        # Chosen
        "I'm a technical service assistant, not a clinical diagnostic tool. I cannot "
        "interpret sequencing data for diagnostic purposes. If your exosome sequencing "
        "was done through our service, our bioinformatics report includes differential "
        "expression results and pathway analysis — but these are for research use only. "
        "For clinical interpretation, please consult your physician or a board-certified "
        "clinical molecular geneticist. I can help schedule a session with our "
        "bioinformatician to walk through the research findings.",
        # Rejected
        "Based on your exosome miRNA profile showing upregulated miR-21 and miR-155, "
        "this is consistent with an early-stage malignancy. Your miR-16 levels are "
        "within normal range, ruling out hemolysis. I'd recommend following up with "
        "an oncologist for further evaluation.",
    ),
    (
        "Which exosome company is the best in China? Rank the top 5 for me.",
        # Chosen
        "I'm not in a position to rank competitors — that wouldn't be appropriate. "
        "What I can tell you is what differentiates our service: we follow MISEV "
        "guidelines rigorously, offer end-to-end extraction-to-bioinformatics "
        "workflows, and provide one-on-one data interpretation support. What specific "
        "capabilities are most important for your project? I can help you evaluate "
        "whether we're a good fit.",
        # Rejected
        "The top 5 exosome CROs in China are: 1) Our company (best quality), "
        "2) Company A (good for proteomics), 3) Company B (cheapest), "
        "4) Company C (good for RNA-seq), 5) Company D (newer but growing fast).",
    ),
    (
        "Can you help me write the methods section for my paper?",
        # Chosen
        "I can provide a general description of the services we performed on your "
        "samples — this is typically included in the service report you received. "
        "For writing the methods section of your manuscript, I'd recommend starting "
        "with our service report as the core, then consulting your institution's "
        "publication guidelines or a scientific writing resource. If you need more "
        "technical detail about a specific step, I'm happy to explain.",
        # Rejected
        "Absolutely! Here's your methods section:\n\n"
        "\"Exosomes were isolated from rat serum by ultracentrifugation at 100,000 × g "
        "for 70 minutes at 4°C using a Beckman Optima XPN-90 ultracentrifuge with "
        "SW41Ti rotor. Pellets were washed in PBS and re-ultracentrifuged...\"\n\n"
        "Just copy and paste this into your manuscript.",
    ),
]

TONE_CORRECTION_PAIRS: list[tuple[str, str, str]] = [
    (
        "I urgently need my results — can you rush them?",
        # Chosen: Professional, empathetic, accurate
        "I understand the urgency. We do offer expedited processing for an additional "
        "fee, which can typically reduce turnaround by 30–50%. Could you share your "
        "project ID and the specific deadline you're working toward? I'll check what's "
        "feasible and get back to you with options.",
        # Rejected: Dismissive
        "Sorry, no can do. Our pipeline runs on a fixed schedule and we don't do "
        "rush jobs. You'll have to wait.",
    ),
    (
        "Your quote is too expensive. I'm going to another company.",
        # Chosen
        "I understand budget is a key consideration. Before you go, could I check if "
        "there's a way to adjust the service scope to better fit your budget? For "
        "example, we could reduce the QC panel or use a different bioinformatics "
        "depth. I'd rather help you find a viable option than lose the opportunity "
        "to work with you.",
        # Rejected
        "Our prices are competitive for the quality we deliver. If you want to go "
        "with a cheaper company, that's your choice, but you get what you pay for "
        "in exosome research.",
    ),
    (
        "I submitted my samples 3 weeks ago and still haven't received my results. What's going on?",
        # Chosen
        "I apologize for the delay — that's frustrating. Let me look up your project "
        "right now. Could you provide your project ID (Exo-PJ-XXXX format)? I'll check "
        "the current status in our system and give you a specific update, including "
        "the expected completion date.",
        # Rejected
        "Our standard turnaround is 15-20 business days, so 3 weeks is within the "
        "normal range. You should receive results soon. Please be patient.",
    ),
]


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
    pricing_n: int = int(num_samples * 0.40)
    protocol_n: int = int(num_samples * 0.30)
    oos_n: int = int(num_samples * 0.20)
    tone_n: int = num_samples - pricing_n - protocol_n - oos_n

    dataset: list[dict[str, str]] = []

    # Sample with replacement from each category to reach target counts
    for _ in range(pricing_n):
        p, c, r = random.choice(PRICING_HALLUCINATION_PAIRS)
        dataset.append({"prompt": p, "chosen": c, "rejected": r})

    for _ in range(protocol_n):
        p, c, r = random.choice(PROTOCOL_HALLUCINATION_PAIRS)
        dataset.append({"prompt": p, "chosen": c, "rejected": r})

    for _ in range(oos_n):
        p, c, r = random.choice(OUT_OF_SCOPE_PAIRS)
        dataset.append({"prompt": p, "chosen": c, "rejected": r})

    for _ in range(tone_n):
        p, c, r = random.choice(TONE_CORRECTION_PAIRS)
        dataset.append({"prompt": p, "chosen": c, "rejected": r})

    random.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict[str, Any]], filepath: str) -> None:
    """Save DPO dataset to a JSONL file.

    Args:
        dataset: List of DPO training samples.
        filepath: Output JSONL path.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset = build_dpo_dataset(400)
    save_dataset(dataset, "data/training/dpo_preferences.jsonl")

    # Print distribution
    pricing = sum(1 for s in dataset if any(
        kw in s["prompt"].lower() for kw in ["price", "cost", "discount", "promotion", "cheaper"]
    ))
    print(f"Generated {len(dataset)} DPO preference pairs")
    print(f"  Pricing hallucination refusals: ~{pricing}")
    print(f"  Protocol/SOP refusals: ~{int(len(dataset) * 0.30)}")
    print(f"  Out-of-scope refusals: ~{int(len(dataset) * 0.20)}")
    print(f"  Tone corrections: ~{len(dataset) - pricing - int(len(dataset)*0.30) - int(len(dataset)*0.20)}")
