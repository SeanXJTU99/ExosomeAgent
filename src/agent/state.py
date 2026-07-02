"""Agent state schema for the exosome CRO customer service system.

Defines the TypedDict state that flows through the LangGraph state machine.
Uses LangGraph's native TypedDict rather than Pydantic BaseModel to avoid
serialization overhead during graph execution.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class ExosomeAgentState(TypedDict):
    """State schema for the exosome CRO agent graph.

    Attributes:
        messages: Conversation history, auto-appended via add_messages reducer.
        sample_type: Sample type extracted from conversation (e.g. serum, plasma,
            cell supernatant, urine, cerebrospinal fluid).
        sample_count: Number of samples the customer intends to submit.
        downstream_exp: Downstream experiment goal (e.g. miRNA-seq, proteomics,
            drug delivery, functional validation).
        qc_requirements: QC methods requested (NTA, TEM, Western Blot).
        hardware_boundary_triggered: Set to True when context exceeds limits
            or confidence falls below threshold — forces fallback routing.
        confidence_score: Model self-reported confidence in its answer (0.0–1.0).
        rag_context: Raw text chunks retrieved from the knowledge base.
        background_summary: Compressed summary of early conversation turns
            when the message window overflows.
        route: Intent classification label set by the router node
            ("technical", "commercial", "logistics", "fallback").
    """

    messages: Annotated[list, add_messages]
    sample_type: str
    sample_count: int
    downstream_exp: str
    qc_requirements: list[str]
    hardware_boundary_triggered: bool
    confidence_score: float
    rag_context: str
    background_summary: str
    route: str
