# Data Anonymization Notice

## Fictional Data Declaration

**All prices, service codes, SOPs, FAQ entries, quotations, and knowledge base content in this repository are FICTIONAL example values created for system demonstration purposes only.**

They do not represent any real company's:
- Commercial pricing or discount structures
- Experimental protocols or standard operating procedures
- Customer data or project information
- Service codes or product catalogues
- Shipping addresses or logistics arrangements

## Scope

The following files contain exclusively fictional data:

| File | Content |
|------|---------|
| `data/mock/knowledge_base.json` | Service catalogue, SOPs, FAQs |
| `data/training/stage1_terminology.jsonl` | Training dialogue samples |
| `data/training/stage2_slot_extraction.jsonl` | Training dialogue samples |
| `data/training/stage3_general.jsonl` | Training dialogue samples |
| `data/training/dpo_preferences.jsonl` | DPO preference pairs |
| `data/training/terminology_samples.json` | Terminology Q&A templates |
| `data/training/dpo_preference_templates.json` | DPO preference templates |
| `src/agent/nodes/commercial_quote.py` | Embedded pricing table |
| `src/agent/nodes/logistics_guide.py` | Embedded SOP templates |

## Purpose

This project demonstrates the technical architecture of a LangGraph-based intelligent customer service agent with QLoRA fine-tuning, DPO alignment, local RAG, and INT4 quantization. The fictional data exists solely to exercise the system's routing, retrieval, and generation pipelines.

No real commercial, clinical, or personal data is present anywhere in this repository.
