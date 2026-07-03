"""Pydantic output validator for LLM responses.

Enforces structured output from the fine-tuned model. After the LLM
generates a response with embedded JSON, this validator parses and
validates it against a strict schema. Malformed or incomplete outputs
are caught here before they can break downstream routing logic.

This is the first line of defense against quantized model format errors
(common with INT4 models: trailing commas, unclosed brackets, etc.).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError


class ExosomeConsultResult(BaseModel):
    """Structured output schema for exosome consultation responses.

    The model is fine-tuned (stage 2) to produce JSON matching this schema
    after each technical consultation turn. The validator extracts this JSON
    from the response text and validates it.
    """

    is_academic_question: bool = Field(
        description="Whether the user asked an academic/technical question"
    )
    extracted_sample_type: Optional[str] = Field(
        default=None,
        description="Extracted sample type: serum, plasma, cell_supernatant, "
                    "urine, csf, saliva, tissue",
    )
    extracted_sample_count: Optional[int] = Field(
        default=None, ge=0, le=10000,
        description="Number of samples the user intends to submit"
    )
    extracted_downstream_exp: Optional[str] = Field(
        default=None,
        description="Extracted downstream experiment: mirna_seq, rna_seq, "
                    "proteomics, drug_delivery, functional, biomarker",
    )
    extracted_qc_requirements: list[str] = Field(
        default_factory=list,
        description="QC methods mentioned: NTA, TEM, Western Blot"
    )
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Model self-reported confidence in this extraction"
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether the model needs more info from the user"
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Slots that still need to be filled"
    )


def _extract_json_block(text: str) -> str | None:
    """Extract a JSON block from model response text.

    Handles common formatting issues from quantized models:
    - JSON in ```json ... ``` code fences
    - Naked JSON objects
    - Trailing commas (attempts to fix)

    Args:
        text: Raw LLM output text.

    Returns:
        Extracted JSON string, or None if no JSON-like structure found.
    """
    # Try fenced JSON block first
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find a bare JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    return None


def _repair_json(json_str: str) -> str:
    """Attempt to repair common JSON formatting errors.

    INT4-quantized models occasionally produce:
    - Trailing commas before closing braces
    - Single quotes instead of double quotes
    - Unquoted keys

    Args:
        json_str: Potentially malformed JSON string.

    Returns:
        Repaired JSON string (best-effort).
    """
    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", json_str)

    # Replace single quotes with double quotes (naive approach)
    # Only do this if the string doesn't already have double-quoted keys
    if '"' not in repaired[:20]:
        repaired = repaired.replace("'", '"')

    return repaired


def validate_llm_output(raw_text: str) -> ExosomeConsultResult | None:
    """Parse and validate the LLM's structured output.

    Extracts the JSON block from the response text, attempts repair if
    needed, and validates against the ExosomeConsultResult schema.

    Args:
        raw_text: Raw text output from the LLM.

    Returns:
        Validated ExosomeConsultResult, or None if parsing/validation failed.
    """
    json_str: str | None = _extract_json_block(raw_text)
    if json_str is None:
        return None

    # Try direct parse first
    try:
        data = json.loads(json_str)
        return ExosomeConsultResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        pass

    # Attempt repair and retry
    repaired: str = _repair_json(json_str)
    try:
        data = json.loads(repaired)
        return ExosomeConsultResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None


def build_validation_error_response(
    raw_text: str,
    error_detail: str = "",
) -> dict:
    """Build a graceful fallback response when validation fails.

    Instead of crashing, returns a canned response that routes the user
    to human support. This prevents quantized model format errors from
    breaking the agent flow.

    Args:
        raw_text: The raw text that failed validation (for logging).
        error_detail: Optional error description.

    Returns:
        Dict with fallback response and error metadata.
    """
    return {
        "is_academic_question": True,
        "extracted_sample_type": "UNKNOWN",
        "extracted_sample_count": 0,
        "extracted_downstream_exp": None,
        "extracted_qc_requirements": [],
        "confidence_score": 0.0,
        "needs_clarification": False,
        "missing_fields": ["sample_type", "sample_count", "downstream_exp"],
    }
