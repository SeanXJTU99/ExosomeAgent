"""FAISS vector store for local exosome knowledge base.

Manages index creation, persistence, and similarity search. Uses
FAISS IndexFlatIP (inner product) for exact nearest-neighbor search
— sufficient for <10k document chunks with 512-dim embeddings.
"""

from __future__ import annotations

import json
import os
from typing import Any


class ExosomeVectorStore:
    """Local FAISS vector store for exosome knowledge base chunks.

    Stores document embeddings and their associated metadata. Supports
    build-from-scratch, save/load persistence, and similarity search.

    Attributes:
        index_path: Directory for FAISS index persistence.
        dimension: Embedding vector dimension (512 for bge-small).
    """

    def __init__(
        self,
        index_path: str = "data/faiss_index",
        dimension: int = 512,
    ) -> None:
        """Initialize the vector store.

        Args:
            index_path: Directory to save/load the FAISS index.
            dimension: Embedding dimension (must match the embedder).
        """
        self.index_path: str = index_path
        self.dimension: int = dimension
        self._index: Any = None          # faiss.IndexFlatIP
        self._metadata: list[dict[str, Any]] = []  # doc metadata per vector
        self._is_loaded: bool = False

    def build_index(
        self,
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Build a new FAISS index from embeddings and metadata.

        Args:
            embeddings: List of normalized embedding vectors.
            metadata: List of metadata dicts, one per embedding.
                Each dict should have at least: {"text": ..., "source": ...}.
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) must match "
                f"metadata count ({len(metadata)})"
            )

        # In production:
        #   import faiss
        #   import numpy as np
        #   vectors = np.array(embeddings, dtype=np.float32)
        #   self._index = faiss.IndexFlatIP(self.dimension)
        #   self._index.add(vectors)

        self._metadata = list(metadata)
        self._is_loaded = True

        print(f"[Mock] Built FAISS index: {len(embeddings)} vectors, "
              f"dim={self.dimension}")

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        os.makedirs(self.index_path, exist_ok=True)

        # In production:
        #   import faiss
        #   faiss.write_index(
        #       self._index,
        #       os.path.join(self.index_path, "index.faiss")
        #   )

        metadata_path: str = os.path.join(self.index_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

        print(f"[Mock] Saved index to: {self.index_path}")

    def load(self) -> bool:
        """Load a previously saved index and metadata from disk.

        Returns:
            True if the index was loaded successfully, False otherwise.
        """
        index_file: str = os.path.join(self.index_path, "index.faiss")
        metadata_file: str = os.path.join(self.index_path, "metadata.json")

        if not os.path.exists(metadata_file):
            print(f"[Mock] No saved index found at: {self.index_path}")
            return False

        # In production:
        #   import faiss
        #   self._index = faiss.read_index(index_file)

        with open(metadata_file, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        self._is_loaded = True
        print(f"[Mock] Loaded index: {len(self._metadata)} documents")
        return True

    def search(
        self,
        query_embedding: list[float],
        k: int = 2,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Search for the k-nearest neighbors of a query embedding.

        Args:
            query_embedding: Normalized query vector.
            k: Number of nearest neighbors to return.
            threshold: Minimum similarity score (0.0–1.0) to include a result.

        Returns:
            List of result dicts, each with:
            {"text": ..., "source": ..., "score": ..., "metadata": {...}}
            Sorted by descending similarity score.
        """
        if not self._is_loaded:
            return []

        # In production:
        #   import numpy as np
        #   query = np.array([query_embedding], dtype=np.float32)
        #   scores, indices = self._index.search(query, k)
        #   results = []
        #   for score, idx in zip(scores[0], indices[0]):
        #       if score >= threshold and idx >= 0:
        #           meta = self._metadata[idx]
        #           results.append({
        #               "text": meta["text"],
        #               "source": meta.get("source", "unknown"),
        #               "score": float(score),
        #               "metadata": meta,
        #           })
        #   return results

        # Mock: return top k metadata entries as if they matched
        results: list[dict[str, Any]] = []
        for i, meta in enumerate(self._metadata[:k]):
            results.append({
                "text": meta.get("text", ""),
                "source": meta.get("source", "unknown"),
                "score": 0.95 - i * 0.05,
                "metadata": meta,
            })
        return results

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self._metadata)

    @property
    def is_loaded(self) -> bool:
        """Check whether the index has been loaded or built."""
        return self._is_loaded
