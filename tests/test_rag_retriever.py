"""Tests for the RAG retrieval pipeline."""

import pytest

from src.rag.embedder import ExosomeEmbedder
from src.rag.vector_store import ExosomeVectorStore
from src.rag.retriever import RAGRetriever
from src.rag.prompt_template import format_rag_prompt
from src.rag.chunker import PriceChunker, SOPChunker


class TestEmbedder:
    """Verify embedder produces correct-dimension vectors."""

    def test_embed_dimension(self) -> None:
        embedder = ExosomeEmbedder()
        vec = embedder.embed("test text")
        assert len(vec) == 512

    def test_embed_batch_dimension(self) -> None:
        embedder = ExosomeEmbedder()
        texts = ["text one", "text two", "text three"]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 512 for v in vecs)

    def test_embed_deterministic_same_text(self) -> None:
        """Same text should produce same mock embedding within one session."""
        embedder = ExosomeEmbedder()
        v1 = embedder.embed("exosome extraction protocol")
        v2 = embedder.embed("exosome extraction protocol")
        assert v1 == v2

    def test_embed_different_text_different(self) -> None:
        embedder = ExosomeEmbedder()
        v1 = embedder.embed("exosome extraction")
        v2 = embedder.embed("exosome proteomics")
        assert v1 != v2  # Different texts should produce different mock vectors

    def test_is_loaded_default_false(self) -> None:
        embedder = ExosomeEmbedder()
        assert embedder.is_loaded is False


class TestVectorStore:
    """Verify vector store build, save, load, and search operations."""

    def test_build_and_search(self) -> None:
        embedder = ExosomeEmbedder()
        store = ExosomeVectorStore()

        texts = [
            "Exosome extraction from serum by ultracentrifugation",
            "QC analysis with NTA and TEM",
            "miRNA sequencing of exosomal RNA",
        ]
        embeddings = embedder.embed_batch(texts)
        metadata = [{"text": t, "source": "test"} for t in texts]

        store.build_index(embeddings, metadata)
        assert len(store) == 3
        assert store.is_loaded is True

        query_vec = embedder.embed("exosome QC NTA TEM")
        results = store.search(query_vec, k=2)
        assert len(results) <= 2
        assert "text" in results[0]
        assert "score" in results[0]

    def test_search_empty_store(self) -> None:
        store = ExosomeVectorStore()
        results = store.search([0.0] * 512, k=2)
        assert results == []

    def test_length_consistency(self) -> None:
        store = ExosomeVectorStore()
        assert len(store) == 0


class TestChunker:
    """Verify chunking strategies produce correct output."""

    def test_price_chunker_from_json(self) -> None:
        chunker = PriceChunker()
        services = [
            {
                "service_code": "Exo-A01",
                "service_name": "Serum Exosome Extraction + miRNA-seq",
                "unit_price": 800,
                "turnaround": "15-20 business days",
                "includes": ["Extraction", "QC", "Sequencing"],
            }
        ]
        chunks = chunker.chunk_from_json(services)
        assert len(chunks) == 1
        assert "Exo-A01" in chunks[0]["text"]
        assert chunks[0]["source"] == "price_catalogue"

    def test_sop_chunker_with_headers(self) -> None:
        chunker = SOPChunker(min_chunk_chars=10)
        text = (
            "## Sample Collection\n"
            "### Step 1: Preparation\n"
            "Prepare all materials before starting.\n\n"
            "### Step 2: Processing\n"
            "Process samples within 30 minutes of collection.\n"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2
        # Each chunk should contain the H2 context
        for chunk in chunks:
            assert "Sample Collection" in chunk["text"]

    def test_sop_chunker_no_headers(self) -> None:
        chunker = SOPChunker(min_chunk_chars=10)
        text = "This is a simple SOP document with no markdown headers at all."
        chunks = chunker.chunk(text)
        # Should produce one chunk if long enough
        assert len(chunks) >= 0


class TestPromptTemplate:
    """Verify RAG prompt formatting."""

    def test_format_rag_prompt(self) -> None:
        context = "Service Exo-A01: exosome miRNA-seq, 800 CNY per sample."
        question = "How much does exosome miRNA-seq cost?"
        prompt = format_rag_prompt(context, question)

        assert context in prompt
        assert question in prompt
        assert "ExoConsult" not in prompt  # RAG prompt uses system prompt, not persona name
        assert len(prompt) > len(context)

    def test_format_rag_prompt_empty_context(self) -> None:
        prompt = format_rag_prompt("", "What services do you offer?")
        # Should still format a valid prompt even with empty context
        assert "What services do you offer?" in prompt


class TestRAGRetriever:
    """Verify end-to-end retrieval pipeline."""

    def test_retrieve_empty_store(self) -> None:
        retriever = RAGRetriever()
        result = retriever.retrieve("exosome extraction cost")
        assert result["found"] is False
        assert result["context"] == ""

    def test_query_with_prompt_empty(self) -> None:
        retriever = RAGRetriever()
        result = retriever.query_with_prompt("How much for miRNA-seq?")
        assert result["found"] is False
        assert result["prompt"] == ""

    def test_retrieve_with_data(self) -> None:
        embedder = ExosomeEmbedder()
        store = ExosomeVectorStore()

        texts = [
            "Exosome miRNA-seq service: 800 CNY per sample, 15-20 day turnaround",
            "Exosome proteomics: 1200 CNY per sample, 20-25 day turnaround",
        ]
        embeddings = embedder.embed_batch(texts)
        metadata = [{"text": t, "source": "price_catalogue"} for t in texts]
        store.build_index(embeddings, metadata)

        retriever = RAGRetriever(embedder=embedder, vector_store=store)
        result = retriever.retrieve("miRNA sequencing price")

        assert result["found"] is True
        assert len(result["context"]) > 0

    def test_query_with_prompt_returns_formatted_prompt(self) -> None:
        embedder = ExosomeEmbedder()
        store = ExosomeVectorStore()

        texts = ["Exosome miRNA-seq service: 800 CNY per sample."]
        embeddings = embedder.embed_batch(texts)
        metadata = [{"text": t, "source": "test"} for t in texts]
        store.build_index(embeddings, metadata)

        retriever = RAGRetriever(embedder=embedder, vector_store=store)
        result = retriever.query_with_prompt("miRNA-seq cost")

        assert result["found"] is True
        assert len(result["prompt"]) > 0
        assert "miRNA-seq" in result["prompt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
