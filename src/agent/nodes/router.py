"""Intent classification router node.

Uses keyword + regex rule-based matching for zero-latency, deterministic
intent classification. Does NOT call the LLM — this ensures routing is
always fast and predictable.
"""

import re

from src.agent.state import ExosomeAgentState

# ---------- Keyword sets for intent classification ----------

TECHNICAL_KEYWORDS: list[str] = [
    "extraction", "extract", "purification", "purify", "isolation", "isolate",
    "centrifugation", "ultracentrifugation", "UC", "density gradient",
    "magnetic bead", "kit", "reagent", "protocol", "method", "technique",
    "NTA", "nanoparticle tracking", "TEM", "electron microscopy",
    "Western Blot", "WB", "CD63", "CD9", "CD81", "TSG101", "Calnexin",
    "marker", "characterization", "purity", "quality control", "QC",
    "miRNA", "mRNA", "RNA-seq", "RNA sequencing", "proteomic",
    "mass spectrometry", "LC-MS", "metabolomic", "lipidomic",
    "exosome", "exosomal", "extracellular vesicle", "EV", "microvesicle",
    "aptamer", "antibody", "ELISA", "flow cytometry", "FACS",
    "differential expression", "bioinformatics", "pipeline", "analysis",
    "drug delivery", "loading", "targeting", "uptake", "functional",
    "serum", "plasma", "urine", "CSF", "cerebrospinal", "saliva",
    "cell supernatant", "culture medium", "conditioned medium",
    "hemolysis", "hemolytic", "contamination", "platelet",
    "freeze-thaw", "anticoagulant", "EDTA", "heparin", "citrate",
    "size distribution", "polydispersity", "zeta potential",
    "pre-processing", "preprocessing", "filtration", "filter",
    "concentration", "BCA", "protein quantification",
]

COMMERCIAL_KEYWORDS: list[str] = [
    "price", "pricing", "cost", "quote", "quotation", "invoice",
    "discount", "bulk", "batch", "package", "promotion",
    "payment", "billing", "reimbursement", "receipt",
    "contract", "agreement", "NDA", "confidential",
    "turnaround", "delivery time", "timeline", "deadline",
    "order", "purchase", "subscription",
]

LOGISTICS_KEYWORDS: list[str] = [
    "shipping", "ship", "delivery", "courier", "SF Express",
    "sample collection", "sample preparation", "sample handling",
    "dry ice", "cold chain", "temperature", "storage",
    "label", "packaging", "package", "box",
    "address", "drop-off", "pick-up",
    "customs", "import", "export", "permit",
    "SOP", "standard operating", "guideline",
]

# Composite patterns for stronger matching
PRICE_PATTERN = re.compile(
    r"(how\s+much|what.*(?:price|cost|fee|charge|rate)|"
    r"(?:price|cost|fee|rate)\s+(?:of|for|range|estimate|list))",
    re.IGNORECASE,
)
SHIPPING_PATTERN = re.compile(
    r"(how\s+(?:to|do|should|can)\s+(?:I|we)\s+(?:send|ship|mail|deliver|package|prepare)|"
    r"(?:send|ship|mail|deliver)\s+(?:my|the|our)\s+sample)",
    re.IGNORECASE,
)


def _count_keyword_matches(text: str, keywords: list[str]) -> int:
    """Count how many keywords from a list appear in the text.

    Args:
        text: Lowercased input text to search within.
        keywords: List of lowercased keywords to match.

    Returns:
        Number of distinct keywords found in the text.
    """
    return sum(1 for kw in keywords if kw.lower() in text)


def classify_intent(user_message: str) -> str:
    """Classify user intent into technical, commercial, or logistics.

    Uses a weighted keyword-counting approach. Multiple strong signals
    in one category increase confidence. If no category dominates,
    defaults to "technical" (the core consultation domain).

    Args:
        user_message: The latest user message text.

    Returns:
        One of "technical", "commercial", "logistics", or "fallback".
    """
    text: str = user_message.lower()

    # Strong pattern matches take priority
    if PRICE_PATTERN.search(text):
        return "commercial"
    if SHIPPING_PATTERN.search(text):
        return "logistics"

    # Keyword counting
    tech_score: int = _count_keyword_matches(text, TECHNICAL_KEYWORDS)
    commercial_score: int = _count_keyword_matches(text, COMMERCIAL_KEYWORDS)
    logistics_score: int = _count_keyword_matches(text, LOGISTICS_KEYWORDS)

    # If no keywords matched at all — fallback for empty/minimal input
    if tech_score == 0 and commercial_score == 0 and logistics_score == 0:
        if len(user_message.strip()) < 3:
            return "fallback"
        # Short ambiguous message defaults to technical to engage in consultation
        return "technical"

    # Pick the category with the highest score (ties go to technical)
    if commercial_score > tech_score and commercial_score > logistics_score:
        return "commercial"
    if logistics_score > tech_score and logistics_score > commercial_score:
        return "logistics"
    return "technical"


def router_node(state: ExosomeAgentState) -> dict:
    """LangGraph node: classify user intent and set the route field.

    Examines the last user message and writes the classification result
    into state["route"]. Subsequent conditional edges read this field.

    Args:
        state: Current agent state.

    Returns:
        Partial state dict with updated "route" field.
    """
    messages: list = state.get("messages", [])
    if not messages:
        return {"route": "fallback"}

    # Get the last user message (skip AI messages)
    last_message = messages[-1]
    user_text: str = (
        last_message.content
        if hasattr(last_message, "content")
        else str(last_message)
    )

    intent: str = classify_intent(user_text)
    return {"route": intent}
