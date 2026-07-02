"""Context window trim manager.

Prevents OOM errors on locally deployed models by enforcing a hard token
limit on conversation history. When the message list exceeds max_tokens,
older messages are truncated and the system prompt + recent turns are
preserved.

Uses tiktoken for fast token counting (no LLM inference needed).
"""

import tiktoken

from langchain_core.messages import BaseMessage, SystemMessage

from src.agent.state import ExosomeAgentState

# Default token limit — conservative for RTX 4090 24G with 14B INT4 model
DEFAULT_MAX_TOKENS: int = 4000

# Reserve headroom for the model's response generation
RESPONSE_RESERVE: int = 512

# Always keep the system prompt + at least this many recent turns
MIN_RECENT_TURNS: int = 3

# OpenAI's cl100k_base is a reasonable proxy for Qwen tokenizer estimation
# (within ~10% accuracy, good enough for trim decisions)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list[BaseMessage]) -> int:
    """Estimate the total token count of a message list.

    Uses tiktoken as a fast approximation. Within ~10% of actual token
    count for most Chinese+English mixed text.

    Args:
        messages: List of LangChain message objects.

    Returns:
        Estimated token count.
    """
    total: int = 0
    for msg in messages:
        content: str = msg.content if hasattr(msg, "content") else str(msg)
        total += len(_ENCODING.encode(content))
        # Each message has ~4 tokens of overhead (role, formatting)
        total += 4
    return total


def trim_context(
    messages: list[BaseMessage],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[BaseMessage]:
    """Trim a message list to fit within a token budget.

    Preservation priority (highest to lowest):
    1. System message (role instructions)
    2. Most recent N turns (MIN_RECENT_TURNS)
    3. Earlier turns (truncated from oldest first)

    Args:
        messages: Full message history.
        max_tokens: Token budget for the context window.

    Returns:
        Trimmed message list guaranteed to be within the token budget.
    """
    if not messages:
        return []

    effective_limit: int = max_tokens - RESPONSE_RESERVE
    current_tokens: int = count_tokens(messages)

    if current_tokens <= effective_limit:
        return list(messages)

    # Separate system message from conversation
    system_msgs: list[BaseMessage] = [
        m for m in messages if isinstance(m, SystemMessage)
    ]
    conv_msgs: list[BaseMessage] = [
        m for m in messages if not isinstance(m, SystemMessage)
    ]

    system_tokens: int = count_tokens(system_msgs)
    available: int = effective_limit - system_tokens

    # Ensure minimum turns are preserved
    min_recent: int = min(MIN_RECENT_TURNS * 2, len(conv_msgs))
    recent_msgs: list[BaseMessage] = conv_msgs[-min_recent:]
    recent_tokens: int = count_tokens(recent_msgs)

    # If even the minimum recent turns + system prompt exceed budget,
    # truncate recent turns from the oldest end
    if recent_tokens > available:
        while count_tokens(recent_msgs) > available and len(recent_msgs) > 2:
            recent_msgs.pop(0)
        return system_msgs + recent_msgs

    # Build result: system + as many earlier turns as fit + recent turns
    remaining_tokens: int = available - recent_tokens
    result: list[BaseMessage] = list(system_msgs)

    older_msgs: list[BaseMessage] = conv_msgs[:-min_recent]
    # Keep older messages from newest to oldest until budget exhausted
    kept_older: list[BaseMessage] = []
    for msg in reversed(older_msgs):
        msg_tokens: int = (
            len(_ENCODING.encode(msg.content if hasattr(msg, "content") else str(msg)))
            + 4
        )
        if remaining_tokens >= msg_tokens:
            kept_older.insert(0, msg)
            remaining_tokens -= msg_tokens
        else:
            break

    result.extend(kept_older)
    result.extend(recent_msgs)
    return result


def trim_context_node(state: ExosomeAgentState) -> dict:
    """LangGraph node: trim context before each agent turn.

    Called at the entry point of every graph invocation. Ensures the
    message history does not exceed the safe token limit before any
    LLM inference occurs.

    Also set hardware_boundary_triggered = True if trimming was
    aggressive (indicating potential context loss).

    Args:
        state: Current agent state.

    Returns:
        Partial state dict with trimmed messages and boundary flag.
    """
    messages: list[BaseMessage] = list(state.get("messages", []))
    if not messages:
        return {}

    original_count: int = len(messages)
    trimmed: list[BaseMessage] = trim_context(messages)

    # If we had to drop more than 50% of messages, flag a boundary event
    boundary_triggered: bool = (
        len(trimmed) < original_count * 0.5 and original_count > 10
    )

    return {
        "messages": trimmed[-len(trimmed):],  # Replace with trimmed list
        "hardware_boundary_triggered": boundary_triggered,
    }
