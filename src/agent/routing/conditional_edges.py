"""Conditional edge functions for the exosome CRO agent graph.

These functions are called by LangGraph after each node to determine the
next node. They implement the hard routing boundaries that prevent the
LLM from handling price/logistics content directly.
"""

from src.agent.state import ExosomeAgentState


def route_after_router(state: ExosomeAgentState) -> str:
    """Determine the next node after intent classification.

    Routes to the appropriate domain node based on the router's
    classification result. Fallback and unrecognized intents go
    directly to output with a prompt for clarification.

    Args:
        state: Current agent state with "route" field populated.

    Returns:
        Name of the next node to execute.
    """
    route: str = state.get("route", "fallback")

    route_map: dict[str, str] = {
        "technical": "technical_consult",
        "commercial": "commercial_quote",
        "logistics": "logistics_guide",
        "fallback": "output",
    }

    return route_map.get(route, "output")


def route_after_slot_extraction(state: ExosomeAgentState) -> str:
    """Determine the next node after slot extraction.

    Two hard boundaries here:
    1. If hardware/confidence boundary is triggered → output (fallback).
    2. If all required commercial slots are filled → redirect to commercial_quote.
       This is the deterministic cut: once we have sample_type + sample_count
       (+ downstream_exp if applicable), route to template-based quoting and
       prevent the LLM from generating pricing content.
    Otherwise, route to output to return the technical answer.

    Args:
        state: Current agent state after slot extraction.

    Returns:
        Name of the next node to execute.
    """
    # Hardware/confidence safety boundary
    if state.get("hardware_boundary_triggered"):
        return "output"

    confidence: float = state.get("confidence_score", 0.0)

    # If confidence is very low, trigger fallback
    if confidence < 0.3:
        return "output"

    # If core commercial slots are fully populated, redirect to quote
    # This hard-cuts the LLM from generating price/logistics content
    sample_type: str = state.get("sample_type", "")
    sample_count: int = state.get("sample_count", 0)
    downstream_exp: str = state.get("downstream_exp", "")

    if sample_type and sample_count > 0 and downstream_exp:
        return "commercial_quote"

    return "output"


def route_after_commercial(state: ExosomeAgentState) -> str:
    """Determine next node after commercial quote generation.

    If the user might also need logistics info (common pattern after
    receiving a quote), we could route to logistics. For now, go to output.

    Args:
        state: Current agent state.

    Returns:
        "output" — the quote is delivered directly.
    """
    return "output"


def route_after_logistics(state: ExosomeAgentState) -> str:
    """Determine next node after logistics guide.

    Args:
        state: Current agent state.

    Returns:
        "output" — the SOP is delivered directly.
    """
    return "output"
