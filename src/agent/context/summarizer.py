"""Conversation history summarizer.

When the message window overflows, older conversation turns are
compressed into a background_summary field rather than being
discarded. This preserves key context (sample type, requirements)
while freeing up tokens for recent turns.

In production, this would call a lightweight LLM for summarization.
The mock implementation extracts structured state as a summary.
"""

from langchain_core.messages import BaseMessage

from src.agent.state import ExosomeAgentState


def summarize_history(messages: list[BaseMessage]) -> str:
    """Generate a concise summary of conversation history.

    Extracts and formats the structured state information that would
    be needed for future turns. In production, this would call a small
    LLM (e.g., Qwen2.5-1.5B) for abstractive summarization.

    Args:
        messages: The full message history to summarize.

    Returns:
        A concise text summary of the conversation so far.
    """
    if not messages:
        return ""

    # Extract key facts from conversation content
    user_texts: list[str] = []
    ai_texts: list[str] = []
    for msg in messages:
        content: str = msg.content if hasattr(msg, "content") else str(msg)
        # Simple role detection by message type name
        msg_type: str = type(msg).__name__
        if "Human" in msg_type or "User" in msg_type:
            user_texts.append(content)
        else:
            ai_texts.append(content)

    full_text: str = " ".join(user_texts)
    summary_parts: list[str] = []

    # Topic extraction via keyword scanning (minimal heuristic)
    topics: list[str] = []
    if any(kw in full_text.lower() for kw in ["serum", "plasma", "blood"]):
        topics.append("serum/plasma exosome services")
    if any(kw in full_text.lower() for kw in ["supernatant", "culture", "conditioned"]):
        topics.append("cell supernatant exosome services")
    if any(kw in full_text.lower() for kw in ["mirna", "rna-seq", "sequencing"]):
        topics.append("RNA sequencing of exosomes")
    if any(kw in full_text.lower() for kw in ["proteom", "mass spec", "lc-ms"]):
        topics.append("proteomics analysis")
    if any(kw in full_text.lower() for kw in ["price", "quote", "cost", "discount"]):
        topics.append("pricing inquiry")
    if any(kw in full_text.lower() for kw in ["ship", "send", "deliver", "courier"]):
        topics.append("sample shipping logistics")

    if topics:
        summary_parts.append(f"Topics discussed: {', '.join(topics)}")
    else:
        summary_parts.append("General exosome CRO service inquiry")

    # Count turns
    summary_parts.append(f"Total conversation turns: {len(messages)}")

    # Count user vs AI messages for rough turn tracking
    user_count: int = len(user_texts)
    ai_count: int = len(ai_texts)
    summary_parts.append(f"User messages: {user_count}, AI responses: {ai_count}")

    return " | ".join(summary_parts)


def summarize_node(state: ExosomeAgentState) -> dict:
    """LangGraph node: compress conversation history into background_summary.

    Called when the trim manager detects message overflow. Stores the
    summary in state["background_summary"] for future turns to reference.

    Args:
        state: Current agent state.

    Returns:
        Partial state dict with updated background_summary.
    """
    messages: list[BaseMessage] = list(state.get("messages", []))
    if not messages:
        return {}

    summary: str = summarize_history(messages)
    return {"background_summary": summary}
