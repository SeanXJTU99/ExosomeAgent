"""RAG retriever — orchestrates embedding, search, and prompt formatting.

Ties together the ExosomeEmbedder, ExosomeVectorStore, and prompt template
into a single retrieval pipeline. The retriever is called by the LangGraph
agent node when a knowledge-base query is detected.
"""

from __future__ import annotations

from typing import Any

from src.rag.embedder import ExosomeEmbedder
from src.rag.vector_store import ExosomeVectorStore
from src.rag.prompt_template import format_rag_prompt


class RAGRetriever:
    """End-to-end retrieval pipeline for the exosome knowledge base.

    Embeds the user query, searches the FAISS index, and formats
    the retrieved context into a prompt for the LLM.

    Attributes:
        embedder: ExosomeEmbedder instance.
        vector_store: ExosomeVectorStore instance.
        top_k: Number of chunks to retrieve.
        threshold: Minimum similarity score.
    """

    def __init__(
        self,
        embedder: ExosomeEmbedder | None = None,
        vector_store: ExosomeVectorStore | None = None,
        top_k: int = 2,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the retriever.

        Args:
            embedder: Pre-initialized embedder. Creates default if None.
            vector_store: Pre-loaded vector store. Creates default if None.
            top_k: Number of chunks to retrieve per query.
            threshold: Minimum similarity score (0.0–1.0).
        """
        self.embedder: ExosomeEmbedder = embedder or ExosomeEmbedder()
        self.vector_store: ExosomeVectorStore = vector_store or ExosomeVectorStore()
        self.top_k: int = top_k
        self.threshold: float = threshold

    def retrieve(self, query: str) -> dict[str, Any]:
        """Retrieve relevant knowledge base chunks for a query.

        Args:
            query: User's question or search query.

        Returns:
            Dict with:
              - "context": Concatenated chunk text (formatted for prompt).
              - "sources": List of source identifiers for attribution.
              - "scores": List of similarity scores.
              - "found": Whether any chunks passed the threshold.
        """
        # Embed the query
        query_embedding: list[float] = self.embedder.embed(query)

        # Search the vector store
        results: list[dict[str, Any]] = self.vector_store.search(
            query_embedding, k=self.top_k, threshold=self.threshold
        )

        if not results:
            return {
                "context": "",
                "sources": [],
                "scores": [],
                "found": False,
            }

        # Concatenate chunks with source attribution
        context_parts: list[str] = []
        sources: list[str] = []
        scores: list[float] = []

        for r in results:
            context_parts.append(r["text"])
            sources.append(r.get("source", "unknown"))
            scores.append(r.get("score", 0.0))

        context: str = "\n\n---\n\n".join(context_parts)

        # Enforce max context length
        max_chars: int = 1500
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[...]"

        return {
            "context": context,
            "sources": sources,
            "scores": scores,
            "found": True,
        }

    def query_with_prompt(
        self,
        user_question: str,
    ) -> dict[str, Any]:
        """Retrieve and format a complete RAG prompt for the LLM.

        This is the main entry point used by the LangGraph RAG node.
        If no relevant chunks are found, returns an empty prompt so
        the downstream node can trigger the "unknown → refuse" rule.

        Args:
            user_question: The user's latest question.

        Returns:
            Dict with:
              - "prompt": Formatted RAG prompt (empty string if nothing found).
              - "found": Whether any chunks were retrieved.
              - "sources": List of source identifiers.
        """
        result: dict[str, Any] = self.retrieve(user_question)

        if not result["found"]:
            return {"prompt": "", "found": False, "sources": []}

        prompt: str = format_rag_prompt(
            context=result["context"],
            question=user_question,
        )

        return {
            "prompt": prompt,
            "found": True,
            "sources": result["sources"],
        }
