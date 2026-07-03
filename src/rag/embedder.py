"""Local embedding model wrapper.

Encapsulates bge-small-zh-v1.5 for Chinese academic/technical text embedding.
Runs on CPU by default to leave GPU VRAM free for the LLM inference engine.
The model is loaded once at module init and reused across requests.
"""

from __future__ import annotations

from typing import Any


class ExosomeEmbedder:
    """Lightweight embedding model for exosome domain text.

    Wraps BAAI/bge-small-zh-v1.5 — a 24-layer bert-base Chinese embedding
    model that produces 512-dimensional vectors. Runs entirely on CPU with
    negligible memory footprint (~500 MB RAM).

    Attributes:
        model_name: HuggingFace model identifier.
        device: Torch device string ("cpu" by default).
        dimension: Output embedding dimension (512 for bge-small).
    """

    model_name: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"
    dimension: int = 512

    def __init__(self, model_name: str | None = None, device: str = "cpu") -> None:
        """Initialize the embedder.

        In production, loads the SentenceTransformer model. The mock
        implementation returns zero vectors of the correct dimension
        for interface testing.

        Args:
            model_name: Override the default embedding model.
            device: Torch device string ("cpu" or "cuda:0").
        """
        if model_name:
            self.model_name = model_name
        self.device = device

        # In production:
        #   from sentence_transformers import SentenceTransformer
        #   self._model = SentenceTransformer(
        #       self.model_name, device=self.device
        #   )
        self._model: Any = None  # Placeholder

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            512-dimensional embedding vector as a list of floats.
        """
        # In production:
        #   return self._model.encode(
        #       text, normalize_embeddings=True
        #   ).tolist()

        # Mock: return a deterministic pseudo-embedding for testing
        # Hash-based to produce consistent values for the same text
        seed: int = abs(hash(text)) % (10 ** 8)
        import random as _random
        _random.seed(seed)
        return [_random.uniform(-1.0, 1.0) for _ in range(self.dimension)]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts in batches.

        Args:
            texts: List of input texts.
            batch_size: Number of texts per batch.

        Returns:
            List of embedding vectors, one per input text.
        """
        # In production:
        #   return self._model.encode(
        #       texts,
        #       batch_size=batch_size,
        #       normalize_embeddings=True,
        #       show_progress_bar=False,
        #   ).tolist()

        return [self.embed(t) for t in texts]

    @property
    def is_loaded(self) -> bool:
        """Check whether the underlying model has been loaded.

        Returns:
            True if the model is ready for inference.
        """
        return self._model is not None
