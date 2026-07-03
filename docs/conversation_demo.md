# Conversation Demos

Fictional example conversations demonstrating the agent's behavior across
different intent categories. All prices, protocols, and project details are
fabricated for demonstration.

---

## 1. Technical Consultation — miRNA-seq Inquiry

**User**: Hi, I want to do exosome miRNA sequencing on rat serum samples. How much does it cost?

**Agent**: Hello! Rat serum exosome miRNA-seq is one of our standard services. To provide an accurate quote and experimental recommendation, may I ask:

1. **Sample volume**: Do you have at least 1 mL of serum per sample? (1.5 mL recommended for replicates.)
2. **Sample count**: How many samples in total?
3. **Pre-processing**: Was the serum processed with a two-step centrifugation (3,000 × g, 15 min) to remove cells and platelets? Any visible hemolysis?

Our service includes: exosome extraction (UC + density gradient) + QC (NTA + TEM + WB, pick two) + small RNA library prep + sequencing (20M reads/sample) + standard bioinformatics.

**User**: I have 12 samples, 1.5 mL each. Serum was double-spun and looks clean — no hemolysis. We want NTA and TEM for QC.

**Agent**: *(internally: slot_extractor fills sample_type=serum, sample_count=12, downstream_exp=mirna_seq, qc_requirements=[NTA, TEM])*

Here is your draft quotation:

| Item | Detail |
|------|--------|
| Service Code | Exo-A01 |
| Service Name | Serum Exosome Extraction + miRNA-seq |
| Sample Type | serum |
| Sample Count | 12 |
| QC Package | NTA, TEM |
| Unit Price | ¥800 / sample |
| Bulk Discount | 10% (10+ samples) |
| Discounted Unit Price | ¥720 / sample |
| **Estimated Total** | **¥8,640** |

Estimated turnaround: 15–20 business days.

---

## 2. Commercial — Direct Price Inquiry

**User**: How much for exosome proteomics on 20 cell supernatant samples?

**Agent**: *(internally: router detects "how much" → commercial intent, bypasses LLM)*

To generate an accurate quotation, I need the following information:

- **Downstream experiment**: What type of proteomics analysis? Options: label-free quantification (LFQ), TMT multiplexing, or DIA.
- **QC requirements**: Which QC assays do you need? (NTA, TEM, Western Blot)

Could you please provide these details?

**User**: Label-free is fine. Just NTA for QC.

**Agent**: *(internally: slot_extractor fills downstream_exp=proteomics, qc_requirements=[NTA])*

Here is your draft quotation:

| Item | Detail |
|------|--------|
| Service Code | Exo-CUSTOM |
| Service Name | Custom Exosome Service (to be confirmed) |
| Sample Type | cell_supernatant |
| Sample Count | 20 |
| Downstream Assay | proteomics |
| QC Package | NTA |

*An account manager will confirm the final service configuration and pricing.*

---

## 3. Logistics — Sample Shipping Guide

**User**: How do I ship my plasma samples to you?

**Agent**: *(internally: router detects "ship" → logistics intent, returns SOP template)*

## Sample Shipping Guidelines

### Packaging Requirements
1. **Primary container**: Sterile, RNase/DNase-free cryovials (1.5–2.0 mL). Label with sample ID, date, and initials.
2. **Secondary container**: Sealed zip-lock bag with absorbent material.
3. **Tertiary container**: Sturdy foam box or insulated shipping container.

### Temperature
- **Plasma**: Ship on **dry ice** (~5 kg per 24h transit). Samples must be frozen at -80°C before packing.

### Plasma-Specific Preparation
1. Collect blood in **EDTA tubes** (NOT heparin).
2. Centrifuge at 3,000 × g for 15 min at 4°C within 30 min of collection.
3. Second spin at 3,000 × g for 10 min to remove residual platelets.
4. Check for hemolysis (pink/red discoloration).
5. Aliquot and freeze at -80°C. Avoid freeze-thaw cycles.

### Courier
We recommend SF Express cold chain. Schedule pickup Monday–Wednesday.

---

## 4. Boundary — Knowledge Gap (RAG "Refuse" Rule)

**User**: What's your proprietary exosome extraction buffer composition?

**Agent**: *(internally: RAG retrieves 0 relevant chunks above threshold)*

Sorry, this question involves proprietary processes or has not been documented yet. I have recorded your inquiry and a technical account manager will follow up with you shortly.

---

## 5. Boundary — Medical Advice Refusal (DPO Alignment)

**User**: Based on my exosome miRNA results showing high miR-21, do I have cancer?

**Agent**: I'm a technical service assistant, not a clinical diagnostic tool. I cannot interpret sequencing data for diagnostic purposes. Our bioinformatics reports are for research use only. For clinical interpretation, please consult your physician or a board-certified clinical molecular geneticist.

I can help schedule a session with our bioinformatician to walk through the research findings from your project. Would that be helpful?
