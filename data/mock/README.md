# Mock Data

Fictional example data for the exosome CRO agent knowledge base.

## Files

- `knowledge_base.json` — Service catalogue (10 services), SOPs (5 sample types), FAQs (10 entries), bulk discount tiers

## Data Notice

**All data in this directory is FICTIONAL and for demonstration purposes only.** See `docs/data_notice.md` for details.

## Usage

The mock data is used to:
1. **Build the FAISS vector index** — `PriceChunker` and `SOPChunker` process the services and SOPs into embeddable chunks.
2. **Serve RAG queries** — the `RAGRetriever` searches the index at inference time.
3. **Template rendering** — `commercial_quote.py` uses a separate hardcoded pricing table (this JSON is for the RAG path only).

## Building the Index

```python
from src.rag.embedder import ExosomeEmbedder
from src.rag.vector_store import ExosomeVectorStore
from src.rag.chunker import PriceChunker, SOPChunker
import json

# Load mock data
with open("data/mock/knowledge_base.json") as f:
    data = json.load(f)

# Chunk services and SOPs
price_chunker = PriceChunker()
sop_chunker = SOPChunker()

chunks = price_chunker.chunk_from_json(data["services"])
for sop in data["sops"]:
    chunks.extend(sop_chunker.chunk(sop["content"], source=sop["id"]))

# Build and save index
embedder = ExosomeEmbedder()
embeddings = embedder.embed_batch([c["text"] for c in chunks])
store = ExosomeVectorStore()
store.build_index(embeddings, chunks)
store.save()
```
