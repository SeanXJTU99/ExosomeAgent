"""RAG system prompt template.

Defines the strict prompt template used for knowledge-grounded responses.
The key rule: if the retrieved context does not contain the answer, the
model MUST refuse to answer rather than fabricate information.

This is the "unknown → refuse" hard boundary for RAG responses.
"""

# ─── RAG System Prompt ───────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT: str = (
    "You are a technical support specialist for an exosome CRO company. "
    "Answer the user's question strictly based on the known knowledge base "
    "content provided below. Follow these rules exactly:\n\n"
    "1. If the answer IS in the knowledge base content, provide it accurately "
    "and cite the relevant service code or SOP section.\n"
    "2. If the answer is NOT in the knowledge base content, you MUST respond: "
    "\"Sorry, this question involves proprietary processes or has not been "
    "documented yet. I have recorded your inquiry and a technical account "
    "manager will follow up with you shortly.\"\n"
    "3. Do NOT fabricate, extrapolate, or guess any experimental data, "
    "pricing, or protocols beyond the provided content.\n"
    "4. Do NOT perform academic reasoning or clinical interpretation beyond "
    "what is explicitly stated in the knowledge base."
)

# Data anonymization notice: the prompt above references a fictional
# knowledge base. All knowledge base content is fabricated for demonstration.


def format_rag_prompt(context: str, question: str) -> str:
    """Format a complete RAG prompt with retrieved context.

    Injects the retrieved knowledge base chunks and the user's question
    into the system prompt template. The model is instructed to respond
    strictly from the provided context.

    Args:
        context: Concatenated knowledge base chunks retrieved by FAISS.
        question: The user's original question.

    Returns:
        A complete prompt string ready for LLM inference.
    """
    return (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"[Known Knowledge Base Content]:\n"
        f"{context}\n\n"
        f"[Current User Question]: {question}\n\n"
        f"Answer:"
    )
