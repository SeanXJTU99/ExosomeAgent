"""Commercial quote node.

Generates standardized quotations using hardcoded template rendering.
This node deliberately does NOT call the LLM — pricing data must be
deterministic and never hallucinated. The fake prices declared here
are EXAMPLE values for demonstration only.
"""

from langchain_core.messages import AIMessage

from src.agent.state import ExosomeAgentState

# ---------- Fake pricing table (EXAMPLE VALUES — not real commercial data) ----------
# Data anonymization notice: all prices, service codes, and turnaround times
# below are FICTIONAL placeholders for system demonstration purposes.

PRICING: dict[str, dict[str, object]] = {
    ("serum", "mirna_seq"): {
        "service_code": "Exo-A01",
        "service_name": "Serum Exosome Extraction + miRNA-seq",
        "unit_price": 800,
        "currency": "CNY",
        "turnaround": "15–20 business days",
        "includes": [
            "Exosome extraction (ultracentrifugation + density gradient)",
            "QC: NTA + TEM + Western Blot (CD63, CD9, CD81, TSG101, Calnexin)",
            "Small RNA library preparation",
            "Sequencing (Illumina NovaSeq, 20M reads/sample)",
            "Standard bioinformatics analysis (differential expression, target prediction)",
        ],
    },
    ("serum", "rna_seq"): {
        "service_code": "Exo-A02",
        "service_name": "Serum Exosome Extraction + Total RNA-seq",
        "unit_price": 950,
        "currency": "CNY",
        "turnaround": "18–22 business days",
        "includes": [
            "Exosome extraction (ultracentrifugation + density gradient)",
            "QC: NTA + TEM + Western Blot",
            "Ribosomal RNA depletion + strand-specific library prep",
            "Sequencing (Illumina NovaSeq, 40M reads/sample)",
            "Standard bioinformatics analysis",
        ],
    },
    ("serum", "proteomics"): {
        "service_code": "Exo-B01",
        "service_name": "Serum Exosome Extraction + Label-free Proteomics",
        "unit_price": 1200,
        "currency": "CNY",
        "turnaround": "20–25 business days",
        "includes": [
            "Exosome extraction (ultracentrifugation + density gradient)",
            "QC: NTA + TEM + Western Blot",
            "Protein digestion + LC-MS/MS (timsTOF or Orbitrap)",
            "Label-free quantification + bioinformatics (GO, KEGG, PPI)",
        ],
    },
    ("cell_supernatant", "mirna_seq"): {
        "service_code": "Exo-C01",
        "service_name": "Cell Supernatant Exosome Extraction + miRNA-seq",
        "unit_price": 700,
        "currency": "CNY",
        "turnaround": "15–20 business days",
        "includes": [
            "Exosome extraction (UC or SEC, method-matched)",
            "QC: NTA + TEM + Western Blot",
            "Small RNA library preparation + sequencing",
            "Standard bioinformatics analysis",
        ],
    },
    ("cell_supernatant", "rna_seq"): {
        "service_code": "Exo-C02",
        "service_name": "Cell Supernatant Exosome Extraction + Total RNA-seq",
        "unit_price": 850,
        "currency": "CNY",
        "turnaround": "18–22 business days",
        "includes": [
            "Exosome extraction (UC or SEC)",
            "QC: NTA + TEM + Western Blot",
            "Total RNA-seq library prep + sequencing",
            "Standard bioinformatics analysis",
        ],
    },
    ("urine", "mirna_seq"): {
        "service_code": "Exo-U01",
        "service_name": "Urine Exosome Extraction + miRNA-seq",
        "unit_price": 750,
        "currency": "CNY",
        "turnaround": "15–20 business days",
        "includes": [
            "Exosome extraction (UC + density gradient)",
            "QC: NTA + TEM + Western Blot",
            "Small RNA library preparation + sequencing",
            "Standard bioinformatics analysis",
        ],
    },
}

# Default service when the exact combination is not in the table
DEFAULT_SERVICE: dict[str, object] = {
    "service_code": "Exo-CUSTOM",
    "service_name": "Custom Exosome Service (to be confirmed)",
    "unit_price": 0,
    "currency": "CNY",
    "turnaround": "To be determined after requirement review",
    "includes": ["Custom extraction protocol", "QC as specified", "Downstream assay TBD"],
}

# Bulk discount thresholds (EXAMPLE values)
BULK_DISCOUNTS: list[tuple[int, float]] = [
    (5, 0.05),   # 5+ samples: 5% off
    (10, 0.10),  # 10+ samples: 10% off
    (20, 0.15),  # 20+ samples: 15% off
    (50, 0.20),  # 50+ samples: 20% off
]


def _lookup_service(sample_type: str, downstream_exp: str) -> dict[str, object]:
    """Look up pricing for a given sample type and experiment combination.

    Args:
        sample_type: Extracted sample type (e.g. "serum").
        downstream_exp: Extracted experiment goal (e.g. "mirna_seq").

    Returns:
        Service pricing dict. Returns DEFAULT_SERVICE if no exact match.
    """
    key = (sample_type, downstream_exp)
    return PRICING.get(key, DEFAULT_SERVICE)


def _compute_discount(sample_count: int) -> float:
    """Determine bulk discount rate based on sample count.

    Args:
        sample_count: Number of samples.

    Returns:
        Discount rate as a decimal (0.0 = no discount, 0.20 = 20% off).
    """
    rate: float = 0.0
    for threshold, discount in BULK_DISCOUNTS:
        if sample_count >= threshold:
            rate = discount
    return rate


def _render_quote(
    sample_type: str,
    sample_count: int,
    downstream_exp: str,
    qc_requirements: list[str],
) -> str:
    """Render a formatted quotation string from structured parameters.

    Args:
        sample_type: Sample type label.
        sample_count: Number of samples.
        downstream_exp: Downstream experiment label.
        qc_requirements: List of QC methods required.

    Returns:
        Formatted Markdown quotation string.
    """
    service: dict[str, object] = _lookup_service(sample_type, downstream_exp)
    unit_price: float = float(service["unit_price"])
    discount: float = _compute_discount(sample_count)
    discounted_unit: float = unit_price * (1.0 - discount)
    subtotal: float = discounted_unit * sample_count

    qc_str: str = ", ".join(qc_requirements) if qc_requirements else "Standard (NTA + TEM + WB)"
    discount_str: str = f"{discount*100:.0f}%" if discount > 0 else "N/A"

    quote: str = (
        f"## Quotation (Draft)\n\n"
        f"**Service Code**: {service['service_code']}\n"
        f"**Service Name**: {service['service_name']}\n\n"
        f"### Order Summary\n"
        f"| Item | Detail |\n"
        f"|------|--------|\n"
        f"| Sample Type | {sample_type} |\n"
        f"| Sample Count | {sample_count} |\n"
        f"| Downstream Assay | {downstream_exp} |\n"
        f"| QC Package | {qc_str} |\n"
        f"| Unit Price | ¥{unit_price:.0f} / sample |\n"
        f"| Bulk Discount | {discount_str} |\n"
        f"| Discounted Unit Price | ¥{discounted_unit:.0f} / sample |\n"
        f"| **Estimated Total** | **¥{subtotal:.0f}** |\n\n"
        f"### Included Services\n"
    )
    includes: list[str] = list(service.get("includes", []))
    for item in includes:
        quote += f"- {item}\n"

    quote += (
        f"\n### Timeline\n"
        f"**Estimated turnaround**: {service['turnaround']}\n\n"
        f"---\n"
        f"*This is a system-generated draft quotation. Prices are example values "
        f"for demonstration purposes only. A formal quotation will be issued by "
        f"your account manager after requirement confirmation.*\n"
    )
    return quote


def commercial_quote_node(state: ExosomeAgentState) -> dict:
    """LangGraph node: generate a deterministic quotation from templates.

    Reads structured slots from the state and renders a Markdown quotation.
    This node is pure template logic — no LLM, no network, no RAG.

    If required slots are missing, prompts the user for missing information.

    Args:
        state: Current agent state with populated slots.

    Returns:
        Partial state dict with the quotation message appended.
    """
    sample_type: str = state.get("sample_type", "")
    sample_count: int = state.get("sample_count", 0)
    downstream_exp: str = state.get("downstream_exp", "")
    qc_requirements: list[str] = list(state.get("qc_requirements", []))

    # Check for missing required fields
    missing: list[str] = []
    if not sample_type:
        missing.append("sample type (e.g., serum, plasma, cell supernatant)")
    if not downstream_exp:
        missing.append("downstream experiment (e.g., miRNA-seq, proteomics)")
    if sample_count <= 0:
        missing.append("sample count")

    if missing:
        prompt: str = (
            "To generate an accurate quotation, I need the following information:\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nCould you please provide these details?"
        )
        return {"messages": [AIMessage(content=prompt)]}

    quote_text: str = _render_quote(
        sample_type, sample_count, downstream_exp, qc_requirements
    )
    return {"messages": [AIMessage(content=quote_text)]}
