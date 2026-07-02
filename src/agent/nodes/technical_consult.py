"""Technical consultation node.

Calls the fine-tuned local LLM to answer complex exosome biology questions.
During early development (before actual fine-tuning), this node returns
mock responses that demonstrate the expected behavior.

Includes context-trim protection: if the message history exceeds the safe
window, older messages are summarized before inference.
"""

from langchain_core.messages import AIMessage

from src.agent.state import ExosomeAgentState
from src.agent.context.trim_manager import trim_context

# ---------- Mock fallback responses (used before fine-tuned model is available) ----------

MOCK_RESPONSES: dict[str, str] = {
    "serum_mirna": (
        "For rat serum exosome miRNA sequencing, we recommend ultracentrifugation "
        "combined with density gradient centrifugation as the gold-standard extraction "
        "method. Key considerations:\n\n"
        "1. **Hemolysis risk**: Rat serum is prone to hemolysis, which releases "
        "endogenous miRNAs like miR-16 and miR-451 that severely contaminate exosomal "
        "small RNA-seq data. We strongly recommend a two-step pre-centrifugation "
        "(3,000 × g, 15 min) to remove cells and platelets before exosome isolation.\n"
        "2. **Minimum input volume**: ≥ 1 mL serum per sample (1.5 mL recommended "
        "for technical replicates).\n"
        "3. **Anticoagulant**: EDTA plasma is preferred over heparin, as heparin "
        "interferes with downstream enzymatic reactions.\n\n"
        "Our service includes: exosome extraction + QC (NTA + TEM + WB, pick two) + "
        "small RNA library prep + sequencing + standard bioinformatics analysis."
    ),
    "cell_supernatant": (
        "For cell culture supernatant (conditioned medium), the recommended workflow "
        "depends on your starting volume and downstream goals:\n\n"
        "- **Large volume (> 100 mL) + proteomics/RNA-seq**: Ultracentrifugation "
        "(100,000 × g, 70 min) is the most cost-effective. Pre-clear by "
        "centrifugation at 2,000 × g for 10 min to remove cell debris, then "
        "0.22 μm filtration before UC.\n"
        "- **Small volume + high-throughput screening**: Magnetic bead-based kits "
        "(e.g., CD63/CD81/CD9 immunoaffinity) offer higher purity but lower yield.\n"
        "- **Functional studies (e.g., uptake assays)**: SEC (size exclusion "
        "chromatography) preserves vesicle integrity best.\n\n"
        "Please confirm your approximate culture volume and whether you need the "
        "exosomes to remain functional after isolation."
    ),
    "qc_standard": (
        "Our standard QC package for exosome characterization includes the "
        "'Big Three' assays:\n\n"
        "1. **NTA (Nanoparticle Tracking Analysis)**: Measures particle size "
        "distribution (typically 30–150 nm for pure exosomes) and concentration "
        "(particles/mL). We use a Malvern NanoSight NS300.\n"
        "2. **TEM (Transmission Electron Microscopy)**: Visual confirmation of "
        "the classic cup-shaped / saucer-like morphology. Negative staining with "
        "2% uranyl acetate.\n"
        "3. **Western Blot**: Detection of positive markers (CD63, CD9, CD81, "
        "TSG101) and negative marker (Calnexin, to rule out cellular contamination).\n\n"
        "You may choose any two of the three for the standard package, or all three "
        "for the comprehensive package (recommended for publication-quality data)."
    ),
    "default": (
        "Thank you for your question about exosome-related services. To provide you "
        "with the most accurate guidance, I'd like to clarify a few details:\n\n"
        "1. **Sample type**: What is your sample source (serum, plasma, cell "
        "supernatant, urine, CSF, etc.)?\n"
        "2. **Sample volume and count**: How many samples do you have, and what is "
        "the approximate volume per sample?\n"
        "3. **Downstream experiment**: What is your goal — RNA-seq, proteomics, "
        "functional validation, drug delivery?\n\n"
        "With this information, I can recommend the optimal extraction method, "
        "QC strategy, and provide an accurate quote."
    ),
}


def _select_mock_response(user_text: str) -> str:
    """Pick an appropriate mock response based on keyword matching.

    Args:
        user_text: The user's latest message, lowercased.

    Returns:
        A mock response string from the predefined set.
    """
    if any(kw in user_text for kw in ["serum", "plasma", "blood", "mirna", "rna-seq", "rnaseq"]):
        return MOCK_RESPONSES["serum_mirna"]
    if any(kw in user_text for kw in ["supernatant", "culture", "conditioned", "medium"]):
        return MOCK_RESPONSES["cell_supernatant"]
    if any(kw in user_text for kw in ["qc", "quality control", "nta", "tem", "western blot", "characterization", "marker"]):
        return MOCK_RESPONSES["qc_standard"]
    return MOCK_RESPONSES["default"]


def technical_consult_node(state: ExosomeAgentState) -> dict:
    """LangGraph node: answer technical exosome biology questions.

    Trims context if needed, then calls the fine-tuned LLM (or mock fallback).
    Does NOT extract slots here — slot extraction is a separate downstream node.

    Args:
        state: Current agent state.

    Returns:
        Partial state dict with the AI response appended to messages.
    """
    messages: list = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content=MOCK_RESPONSES["default"])]}

    user_text: str = (
        messages[-1].content
        if hasattr(messages[-1], "content")
        else str(messages[-1])
    )

    # Trim context to prevent OOM on long conversations
    trimmed_messages: list = trim_context(messages, max_tokens=4000)

    # In production, this would call the fine-tuned LLM:
    #   response = local_finetuned_llm.invoke(trimmed_messages)
    # For now, use mock responses that demonstrate expected behavior.
    response_text: str = _select_mock_response(user_text.lower())

    response = AIMessage(content=response_text)
    return {"messages": [response]}
