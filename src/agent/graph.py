"""Graph builder for the exosome CRO agent.

Constructs the LangGraph StateGraph with all nodes, conditional edges,
and the entry/exit flow. The compiled graph is the main runtime artifact.
"""

from langgraph.graph import END, StateGraph

from src.agent.state import ExosomeAgentState
from src.agent.nodes.router import router_node
from src.agent.nodes.technical_consult import technical_consult_node
from src.agent.nodes.slot_extractor import slot_extractor_node
from src.agent.nodes.commercial_quote import commercial_quote_node
from src.agent.nodes.logistics_guide import logistics_guide_node
from src.agent.routing.conditional_edges import (
    route_after_router,
    route_after_slot_extraction,
)
from src.agent.context.trim_manager import trim_context_node


def output_node(state: ExosomeAgentState) -> dict:
    """Terminal node that returns the final message to the user.

    Args:
        state: Current agent state with populated messages.

    Returns:
        Empty dict — the final message is already in state["messages"].
    """
    return {}


def build_graph() -> StateGraph:
    """Build and compile the exosome CRO agent StateGraph.

    Topology:
        [User Input] → router → technical/commercial/logistics
            → slot_extractor (technical only) → conditional check → output

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    workflow = StateGraph(ExosomeAgentState)

    # Register nodes
    workflow.add_node("router", router_node)
    workflow.add_node("technical_consult", technical_consult_node)
    workflow.add_node("slot_extractor", slot_extractor_node)
    workflow.add_node("commercial_quote", commercial_quote_node)
    workflow.add_node("logistics_guide", logistics_guide_node)
    workflow.add_node("trim_context", trim_context_node)
    workflow.add_node("output", output_node)

    # Entry point: trim context first, then route
    workflow.set_entry_point("trim_context")

    # trim_context → router (always)
    workflow.add_edge("trim_context", "router")

    # Router → conditional split to technical / commercial / logistics / fallback
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "technical_consult": "technical_consult",
            "commercial_quote": "commercial_quote",
            "logistics_guide": "logistics_guide",
            "output": "output",
        },
    )

    # Technical → slot extraction → conditional (confidence + completeness check)
    workflow.add_edge("technical_consult", "slot_extractor")
    workflow.add_conditional_edges(
        "slot_extractor",
        route_after_slot_extraction,
        {
            "commercial_quote": "commercial_quote",
            "output": "output",
        },
    )

    # Commercial / logistics → output (template-based, no further processing)
    workflow.add_edge("commercial_quote", "output")
    workflow.add_edge("logistics_guide", "output")

    # Output → END
    workflow.add_edge("output", END)

    return workflow.compile()


# Module-level compiled graph instance
app = build_graph()
