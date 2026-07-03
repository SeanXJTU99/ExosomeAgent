"""Confidence-based routing fallback.

When the LLM's self-reported confidence falls below a threshold, or when
Pydantic validation fails entirely, this module triggers a hard fallback
that routes the conversation to template-based output (or human handoff)
instead of letting a low-confidence LLM response reach the user.

This is the second line of defense — after Pydantic validation.
"""

from __future__ import annotations

from src.agent.state import ExosomeAgentState

# ─── Thresholds ─────────────────────────────────────────────────────────────

# Confidence below this -> trigger fallback
CONFIDENCE_THRESHOLD: float = 0.6


def check_confidence(state: ExosomeAgentState) -> str:
    """Evaluate model confidence and determine routing.

    Called after slot extraction and Pydantic validation. Returns the
    name of the next node based on confidence assessment.

    Decision rules:
      1. hardware_boundary_triggered → "output" (emergency fallback)
      2. confidence_score < 0.6 → "output" (low confidence → refuse)
      3. sample_type == "UNKNOWN" → "output" (validation failed)
      4. confidence_score 0.6–0.8 → flag but continue (warn zone)
      5. confidence_score > 0.8 → normal routing

    Args:
        state: Current agent state with confidence_score set.

    Returns:
        Next node name: "output" for fallback, "commercial_quote" if
        all business slots are filled, otherwise "output".
    """
    # Emergency boundary
    if state.get("hardware_boundary_triggered"):
        return "output"

    confidence: float = state.get("confidence_score", 0.0)
    sample_type: str = state.get("sample_type", "")

    # Hard refusal: confidence too low or validation produced UNKNOWN
    if confidence < CONFIDENCE_THRESHOLD:
        return "output"

    if sample_type == "UNKNOWN":
        return "output"

    # Check if all business slots are filled → redirect to quote
    if (
        state.get("sample_type")
        and state.get("sample_count", 0) > 0
        and state.get("downstream_exp")
    ):
        return "commercial_quote"

    return "output"
