"""Document chunker for exosome CRO knowledge base.

Two chunking strategies:
  1. PriceChunker: Splits price tables into standalone QA sentences.
  2. SOPChunker: Splits SOP documents by Markdown headers (H3),
     keeping parent header context for completeness.

All data processed is FICTIONAL and for demonstration purposes only.
"""

from __future__ import annotations

import re
from typing import Any


class PriceChunker:
    """Split price/service catalogue text into QA-style sentence chunks.

    Each chunk is a self-contained "service description" that can be
    independently retrieved and understood. Example input:
      "Serum exosome miRNA-seq: Exo-A01, ¥800/sample, 15-20 days"
    becomes:
      {"text": "Serum exosome miRNA-seq service (code Exo-A01): ¥800 per
       sample, turnaround 15-20 business days. Includes extraction, QC,
       library prep, sequencing, and bioinformatics."}
    """

    def __init__(
        self,
        min_chunk_chars: int = 80,
        max_chunk_chars: int = 500,
    ) -> None:
        """Initialize the price chunker.

        Args:
            min_chunk_chars: Minimum characters per chunk (shorter chunks are merged).
            max_chunk_chars: Maximum characters per chunk (longer chunks are split).
        """
        self.min_chunk_chars: int = min_chunk_chars
        self.max_chunk_chars: int = max_chunk_chars

    def chunk(self, text: str, source: str = "price_catalogue") -> list[dict[str, Any]]:
        """Split price catalogue text into QA-style chunks.

        Splits on newlines and sentence boundaries. Each chunk should
        describe one service with its code, price, and turnaround time.

        Args:
            text: Raw price catalogue text.
            source: Document source identifier for metadata.

        Returns:
            List of chunk dicts with "text" and "source" keys.
        """
        # Split on double newlines (paragraphs) first, then sentences
        paragraphs: list[str] = re.split(r"\n\s*\n", text.strip())
        chunks: list[dict[str, Any]] = []

        for para in paragraphs:
            para = para.strip()
            if len(para) < self.min_chunk_chars:
                continue

            # Split long paragraphs on numbered items or bullet points
            if len(para) > self.max_chunk_chars:
                sub_chunks: list[str] = re.split(
                    r"(?=(?:^|\n)(?:\d+\.|\-|\*)\s)", para
                )
                for sc in sub_chunks:
                    sc = sc.strip()
                    if len(sc) >= self.min_chunk_chars:
                        chunks.append(self._make_chunk(sc, source))
            else:
                chunks.append(self._make_chunk(para, source))

        return chunks

    def chunk_from_json(
        self,
        service_entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create chunks from structured service entries in JSON.

        Each service entry is a dict with fields like:
        {"service_code", "service_name", "unit_price", "turnaround", "includes"}

        Args:
            service_entries: List of service entry dicts.

        Returns:
            List of chunk dicts ready for embedding.
        """
        chunks: list[dict[str, Any]] = []
        for entry in service_entries:
            text_parts: list[str] = [
                f"Service: {entry.get('service_name', 'N/A')}",
                f"Code: {entry.get('service_code', 'N/A')}",
                f"Price: ¥{entry.get('unit_price', 'N/A')} per sample",
                f"Turnaround: {entry.get('turnaround', 'N/A')}",
            ]
            includes: list[str] = entry.get("includes", [])
            if includes:
                text_parts.append("Includes: " + "; ".join(includes))

            text: str = ". ".join(text_parts) + "."
            chunks.append(self._make_chunk(text, "price_catalogue"))
        return chunks

    @staticmethod
    def _make_chunk(text: str, source: str) -> dict[str, Any]:
        """Create a standardized chunk dict.

        Args:
            text: Chunk text content.
            source: Source identifier.

        Returns:
            Chunk dict.
        """
        return {"text": text, "source": source}


class SOPChunker:
    """Split SOP documents by Markdown header hierarchy.

    Uses H3 (###) as the primary split point, but prepends the H2 ancestor
    header to each chunk so the retrieved text retains its document context.

    Example:
      ## Plasma Exosome Pre-processing SOP
      ### Step 1: Blood Collection
      ...
      ### Step 2: Centrifugation
      ...

    Each "###" section becomes a chunk with the "##" header prepended.
    """

    def __init__(
        self,
        min_chunk_chars: int = 100,
        max_chunk_chars: int = 1500,
    ) -> None:
        """Initialize the SOP chunker.

        Args:
            min_chunk_chars: Minimum characters per chunk.
            max_chunk_chars: Maximum characters per chunk.
        """
        self.min_chunk_chars: int = min_chunk_chars
        self.max_chunk_chars: int = max_chunk_chars

    def chunk(self, text: str, source: str = "sop") -> list[dict[str, Any]]:
        """Split a Markdown SOP document into header-anchored chunks.

        Args:
            text: Raw Markdown SOP text.
            source: Document source identifier.

        Returns:
            List of chunk dicts with "text" and "source" keys.
        """
        chunks: list[dict[str, Any]] = []

        # Find H2 headers (## ) as section anchors
        h2_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        h3_pattern = re.compile(r"^###\s+(.+)$", re.MULTILINE)

        # Split into H2 sections
        h2_splits: list[tuple[str, str]] = []  # [(h2_title, h2_content)]
        h2_matches = list(h2_pattern.finditer(text))

        for i, match in enumerate(h2_matches):
            h2_title = match.group(1).strip()
            start = match.end()
            end = h2_matches[i + 1].start() if i + 1 < len(h2_matches) else len(text)
            h2_content = text[start:end].strip()
            h2_splits.append((h2_title, h2_content))

        if not h2_splits:
            # No H2 headers — treat the whole text as one section
            h2_splits = [("", text)]

        # Within each H2 section, split on H3 boundaries
        for h2_title, h2_content in h2_splits:
            h3_matches = list(h3_pattern.finditer(h2_content))

            if not h3_matches:
                # No H3 — the whole H2 section is one chunk
                chunk_text = f"## {h2_title}\n\n{h2_content}" if h2_title else h2_content
                if len(chunk_text) >= self.min_chunk_chars:
                    chunks.append(self._make_chunk(chunk_text, source))
                continue

            for j, h3_match in enumerate(h3_matches):
                h3_title = h3_match.group(1).strip()
                start = h3_match.end()
                end = (
                    h3_matches[j + 1].start()
                    if j + 1 < len(h3_matches)
                    else len(h2_content)
                )
                h3_body = h2_content[start:end].strip()

                # Prepend H2 context to each H3 chunk
                header_prefix = f"## {h2_title}\n\n" if h2_title else ""
                chunk_text = f"{header_prefix}### {h3_title}\n\n{h3_body}"

                if len(chunk_text) >= self.min_chunk_chars:
                    # If too long, truncate with a note
                    if len(chunk_text) > self.max_chunk_chars:
                        chunk_text = chunk_text[:self.max_chunk_chars] + "\n\n[...]"
                    chunks.append(self._make_chunk(chunk_text, source))

        return chunks

    @staticmethod
    def _make_chunk(text: str, source: str) -> dict[str, Any]:
        """Create a standardized chunk dict.

        Args:
            text: Chunk text content.
            source: Source identifier.

        Returns:
            Chunk dict.
        """
        return {"text": text, "source": source}
