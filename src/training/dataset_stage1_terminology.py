"""Stage 1 Dataset: Exosome terminology alignment.

Generates ~500 training samples that teach the base model exosome-specific
terminology, experimental protocols, and domain knowledge. The goal is to
align a general-purpose LLM with the vocabulary and concepts of extracellular
vesicle (EV) research.

All data below is FICTIONAL and for demonstration purposes only.
"""

import json
import random
from typing import Any

random.seed(42)

# ─── Terminology alignment templates ───────────────────────────────────────

# Each entry: (user_question, assistant_answer)
# Covers: extraction methods, QC, sample types, downstream applications,
# common pitfalls, and basic exosome biology.

TERMINOLOGY_SAMPLES: list[dict[str, str]] = [
    # ── Extraction methods ──
    {
        "user": "What is the best method to isolate exosomes from serum?",
        "assistant": (
            "For serum exosome isolation, the gold-standard method is ultracentrifugation "
            "(UC) combined with density gradient centrifugation. UC typically runs at "
            "100,000 × g for 70–90 minutes. Density gradient (sucrose or iodixanol) "
            "further separates exosomes (1.13–1.19 g/mL density) from contaminating "
            "particles like protein aggregates and lipoproteins. For high-throughput "
            "clinical screening, magnetic bead-based immunoaffinity capture (targeting "
            "CD63/CD81/CD9) offers higher purity but lower yield. SEC (size exclusion "
            "chromatography) is recommended when vesicle integrity must be preserved "
            "for functional assays."
        ),
    },
    {
        "user": "What's the difference between ultracentrifugation and kit-based exosome extraction?",
        "assistant": (
            "Ultracentrifugation (UC) and kit-based extraction differ in purity, yield, "
            "and scalability:\n\n"
            "- **UC (100,000 × g, 70 min)**: High yield, moderate purity. Co-pellets "
            "protein aggregates and lipoproteins. Best for large-volume samples (≥10 mL) "
            "when cost is a concern.\n"
            "- **Density gradient UC**: Highest purity. Separates by buoyant density "
            "(1.13–1.19 g/mL for exosomes). Lower yield. Best for publication-quality "
            "proteomics or RNA-seq.\n"
            "- **SEC (Size Exclusion Chromatography)**: Preserves vesicle integrity. "
            "Good purity. Lower throughput. Best for functional studies (uptake, "
            "proliferation assays).\n"
            "- **Magnetic bead kits (immunoaffinity)**: High purity for specific "
            "subpopulations (e.g., CD63+ exosomes). Lower yield. Best for biomarker "
            "discovery with limited sample volume.\n"
            "- **Precipitation kits (PEG-based)**: Fast and cheap but low purity — "
            "co-precipitates non-exosomal proteins and RNAs. Not recommended for "
            "omics applications."
        ),
    },
    {
        "user": "Can I use a precipitation kit for exosome RNA-seq?",
        "assistant": (
            "We strongly advise against using PEG-based precipitation kits for exosome "
            "RNA-seq. Precipitation methods co-isolate non-exosomal contaminants "
            "(protein aggregates, lipoprotein particles, free-circulating miRNAs) that "
            "confound downstream RNA-seq results. The contaminating RNA can account for "
            "30–50% of total reads, obscuring genuine exosomal RNA signals.\n\n"
            "For RNA-seq applications, we recommend UC + density gradient or SEC "
            "for isolation, followed by a quality check with NTA and Western Blot "
            "before library preparation."
        ),
    },
    # ── Sample types and pre-processing ──
    {
        "user": "How should I prepare plasma samples for exosome extraction?",
        "assistant": (
            "Proper plasma preparation is critical for exosome quality:\n\n"
            "1. **Anticoagulant choice**: Use EDTA tubes. Do NOT use heparin — heparin "
            "interferes with reverse transcription and PCR steps in RNA-seq.\n"
            "2. **Timing**: Process blood within 30 minutes of collection at room "
            "temperature. Delayed processing increases platelet activation and "
            "microparticle release.\n"
            "3. **Two-step centrifugation**:\n"
            "   - Step 1: 2,500 × g, 15 min, 4°C → removes cells and platelets.\n"
            "   - Step 2: 3,000 × g, 10 min, 4°C → removes residual platelet fragments.\n"
            "4. **Hemolysis check**: Measure free hemoglobin (absorbance at 414 nm). "
            "Hemolyzed samples (Hb > 0.3 mg/mL) should be flagged — erythrocyte-derived "
            "miRNAs (miR-16, miR-451a, miR-144) contaminate exosomal miRNA profiles.\n"
            "5. **Storage**: Aliquot plasma and freeze at -80°C. Avoid freeze-thaw "
            "cycles — each cycle reduces exosome yield by ~10%."
        ),
    },
    {
        "user": "What sample types can you process for exosome isolation?",
        "assistant": (
            "We routinely process the following sample types for exosome isolation:\n\n"
            "- **Serum / Plasma**: Most common. Require ≥1 mL per sample.\n"
            "- **Cell Culture Supernatant**: From adherent or suspension cells. "
            "Require exosome-depleted FBS for the conditioning period.\n"
            "- **Urine**: First morning void or 24h collection. Require ≥50 mL "
            "before concentration.\n"
            "- **Cerebrospinal Fluid (CSF)**: ≥0.5 mL. Handle with extra care "
            "due to low exosome concentration.\n"
            "- **Saliva**: ≥2 mL. Pre-filter to remove mucins and bacteria.\n"
            "- **Milk / Colostrum**: ≥10 mL. Require casein depletion before isolation.\n"
            "- **Bronchoalveolar Lavage Fluid (BALF)**: ≥5 mL.\n"
            "- **Ascites / Pleural Effusion**: ≥10 mL.\n"
            "- **Synovial Fluid**: ≥2 mL. Pre-treat with hyaluronidase to reduce "
            "viscosity.\n\n"
            "For each sample type, we provide a specific pre-processing SOP. Please "
            "contact us with your sample type for the detailed protocol."
        ),
    },
    # ── QC and characterization ──
    {
        "user": "What QC tests do you perform on isolated exosomes?",
        "assistant": (
            "Our standard exosome QC package includes the 'Big Three' assays "
            "recommended by MISEV2018/2023 guidelines:\n\n"
            "1. **NTA (Nanoparticle Tracking Analysis)**: Malvern NanoSight NS300. "
            "Measures particle size distribution (D10, D50, D90) and concentration "
            "(particles/mL). Pure exosomes typically show a monodisperse peak at "
            "30–150 nm.\n"
            "2. **TEM (Transmission Electron Microscopy)**: Negative staining with "
            "2% uranyl acetate. Visual confirmation of the classic cup-shaped / "
            "saucer-like morphology. We capture images at 80–120 kV.\n"
            "3. **Western Blot**: Detection of:\n"
            "   - Positive markers: CD63 (~53 kDa), CD9 (~24 kDa), CD81 (~26 kDa), "
            "TSG101 (~46 kDa), Syntenin-1 (~32 kDa).\n"
            "   - Negative marker: Calnexin (~90 kDa) — to rule out ER/ cellular "
            "contamination.\n\n"
            "Additional QC options available:\n"
            "- BCA protein quantification\n"
            "- Tunable Resistive Pulse Sensing (TRPS/qNano) for higher-resolution sizing\n"
            "- Cryo-EM for near-native morphology (publication-grade)\n"
            "- Flow cytometry (bead-coupled) for surface marker profiling"
        ),
    },
    {
        "user": "Why is my Western Blot for CD63 showing no band?",
        "assistant": (
            "Missing CD63 signal on Western Blot is a common troubleshooting point. "
            "Here are the most likely causes:\n\n"
            "1. **Insufficient protein loading**: Exosome lysate should be ≥5 µg per "
            "lane. Run BCA quantification first to confirm protein concentration.\n"
            "2. **Sample buffer and boiling**: CD63 is a tetraspanin with 4 "
            "transmembrane domains. Boil samples at 95°C for 5 min in reducing "
            "buffer (with DTT or β-mercaptoethanol). Under-boiling gives poor "
            "solubilization.\n"
            "3. **Antibody specificity**: Not all CD63 antibodies recognize exosomal "
            "CD63 equally. We use clone Ts63 (or H5C6) validated for exosome WB. "
            "Non-reducing conditions may improve detection for some epitopes.\n"
            "4. **Low exosome yield**: If your isolation method had low recovery "
            "(e.g., SEC with a small loading volume), the exosome amount may be "
            "below WB detection limit. Consider concentrating your sample.\n"
            "5. **CD63 is not universally expressed**: Some cell types (e.g., "
            "certain cancer lines) have low CD63. Consider probing for CD9 or "
            "CD81 simultaneously.\n"
            "6. **Glycosylation**: CD63 is heavily glycosylated — the band may "
            "appear as a smear at 40–65 kDa rather than a sharp band at 53 kDa."
        ),
    },
    # ── Downstream applications ──
    {
        "user": "What sequencing depth do you recommend for exosome miRNA-seq?",
        "assistant": (
            "For exosome miRNA-seq, we recommend:\n\n"
            "- **Sequencing platform**: Illumina NovaSeq 6000 or X Plus, SE50 or SE75.\n"
            "- **Depth**: 15–20 million reads per sample for standard differential "
            "expression analysis. For novel miRNA discovery or low-input samples "
            "(<10 ng RNA), increase to 25–30 million reads.\n"
            "- **Library prep**: Use a kit optimized for small RNA (e.g., QIAseq "
            "miRNA, NEBNext Small RNA, or CATS Small RNA-seq). Standard TruSeq "
            "Small RNA has known bias against certain miRNA sequences — discuss "
            "with our bioinformatics team if your miRNA of interest is affected.\n"
            "- **Spike-ins**: We include synthetic spike-in controls (e.g., "
            "UniSp2/UniSp4/UniSp5 from Exiqon) for normalization and QC.\n\n"
            "Note: Exosomal RNA is highly fragmented and dominated by small RNAs "
            "(<200 nt). The RIN (RNA Integrity Number) is NOT a relevant QC metric "
            "for exosome RNA — use Bioanalyzer Small RNA chip or Fragment Analyzer "
            "instead."
        ),
    },
    {
        "user": "Can you do proteomics on exosomes?",
        "assistant": (
            "Yes, exosome proteomics is one of our core services. We offer:\n\n"
            "- **Label-free quantification (LFQ)**: LC-MS/MS on timsTOF Pro or "
            "Orbitrap Eclipse. Typically identifies 800–2,000 protein groups from "
            "10 µg exosome protein. Economical for large cohort studies.\n"
            "- **TMT (Tandem Mass Tag)**: 6-plex to 18-plex multiplexing. Better "
            "quantification accuracy. Good for small-to-medium sample sets with "
            "replicates.\n"
            "- **DIA (Data-Independent Acquisition)**: Higher reproducibility, "
            "better for low-abundance proteins. Recommended for biomarker discovery.\n\n"
            "Key consideration: Exosome samples require special lysis and digestion "
            "protocols because of the high lipid content. We use an optimized "
            "SDC (sodium deoxycholate)-based protocol that achieves >90% digestion "
            "efficiency for exosomal membrane proteins.\n\n"
            "Bioinformatics: Standard analysis includes GO enrichment, KEGG pathway "
            "mapping, PPI network analysis (STRING/Cytoscape), and comparison to "
            "ExoCarta/Vesiclepedia databases."
        ),
    },
    # ── Exosome biology fundamentals ──
    {
        "user": "What is the typical size range of exosomes?",
        "assistant": (
            "Exosomes are typically 30–150 nm in diameter, as measured by NTA or "
            "TEM. This distinguishes them from:\n\n"
            "- **Microvesicles / ectosomes**: 100–1,000 nm, shed directly from the "
            "plasma membrane.\n"
            "- **Apoptotic bodies**: 1–5 µm, released during programmed cell death.\n"
            "- **Exomeres**: <50 nm, non-membranous nanoparticles recently identified "
            "as a distinct extracellular particle class.\n\n"
            "In practice, most isolation methods yield a mixture of small EVs "
            "(exosomes + exomeres + small microvesicles). The MISEV2018 guidelines "
            "recommend using the term 'small extracellular vesicles (sEVs)' rather "
            "than 'exosomes' unless you have demonstrated endosomal origin (e.g., "
            "via syntenin-1 or CD63-GFP tracing). Our service reports 'sEV-enriched "
            "fractions' with the size distribution profile for transparency."
        ),
    },
    # ── More domain-specific Q&A ──
    {
        "user": "Does hemolysis affect exosome miRNA sequencing results?",
        "assistant": (
            "Yes, hemolysis is one of the most significant pre-analytical variables "
            "that can compromise exosome miRNA-seq data:\n\n"
            "1. **Mechanism**: Hemolysis releases erythrocyte-derived miRNAs (miR-16-5p, "
            "miR-451a, miR-144-3p, miR-486-5p) into plasma/serum. These miRNAs are "
            "highly abundant — a single erythrocyte contains ~0.1 pg of miRNA.\n"
            "2. **Contamination effect**: In hemolyzed samples, erythrocyte miRNAs "
            "can constitute >50% of total miRNA reads, masking genuine exosomal "
            "miRNA signals.\n"
            "3. **Detection**: Measure ΔCt(miR-23a – miR-451a) by qPCR. ΔCt > 5 "
            "indicates significant hemolysis. Alternatively, measure free hemoglobin "
            "by spectrophotometry (A414).\n"
            "4. **Mitigation**: (a) Strict pre-analytical SOP (two-step centrifugation); "
            "(b) Exclude samples with visible hemolysis (pink/red discoloration); "
            "(c) Bioinformatically flag and adjust for hemolysis-associated miRNAs.\n\n"
            "Our QC pipeline includes a hemolysis assessment for every plasma/serum "
            "sample before proceeding to library preparation."
        ),
    },
    {
        "user": "How do you validate that the isolated particles are actually exosomes?",
        "assistant": (
            "We follow the MISEV2018/2023 minimal experimental requirements for "
            "single-vesicle characterization:\n\n"
            "1. **Size**: NTA shows particles in the expected 30–150 nm range with "
            "a monodisperse distribution (D10/D90 ratio < 5).\n"
            "2. **Morphology**: TEM shows cup-shaped vesicles with a lipid bilayer "
            "membrane (negative staining artifact — cryo-EM would show round shape).\n"
            "3. **Markers**: At least 3 positive protein markers (typically CD63, "
            "CD9, CD81, TSG101, Syntenin-1) and 1 negative marker (Calnexin, GM130, "
            "or Histone H3).\n"
            "4. **Topology**: Protease protection assay — CD63 signal is protected "
            "from trypsin digestion unless the membrane is permeabilized with "
            "detergent, confirming lumen-facing epitope orientation.\n\n"
            "For publication-grade validation, we also offer:\n"
            "- Immunogold TEM (CD63/CD81 labeling)\n"
            "- Single-particle interferometric reflectance imaging (SP-IRIS, ExoView)\n"
            "- Nanoflow cytometry (Apogee or CytoFLEX with violet SSC)"
        ),
    },
]

# Augment the base samples with variants to reach target count
QUESTION_VARIANTS: list[str] = [
    "Can you tell me about {topic}?",
    "I'd like to know more about {topic}.",
    "What should I know about {topic}?",
    "Could you explain {topic} for exosome research?",
    "I'm new to this — what's the deal with {topic}?",
    "Give me a quick overview of {topic}.",
    "How does {topic} work in the context of exosomes?",
    "What are the key points about {topic}?",
]

TOPICS: list[str] = [
    "exosome extraction from plasma",
    "exosome quality control standards",
    "NTA analysis for exosomes",
    "TEM imaging of exosomes",
    "exosome Western Blot markers",
    "exosome miRNA sequencing",
    "exosome proteomics analysis",
    "ultracentrifugation vs SEC for exosomes",
    "exosome storage conditions",
    "hemolysis effects on exosome samples",
    "exosome drug delivery applications",
    "CD63 and CD9 as exosome markers",
    "MISEV guidelines for exosome research",
    "exosome concentration measurement by BCA",
    "differential centrifugation for exosome isolation",
]


def generate_terminology_dataset(num_samples: int = 500) -> list[dict[str, Any]]:
    """Generate the stage 1 terminology alignment dataset.

    Combines hand-crafted high-quality Q&A pairs with template-augmented
    variants to reach the target sample count. Outputs in ChatML format.

    Args:
        num_samples: Target number of training samples.

    Returns:
        List of samples in ChatML conversation format.
    """
    dataset: list[dict[str, Any]] = []

    # Add hand-crafted samples first (highest quality)
    for sample in TERMINOLOGY_SAMPLES:
        dataset.append({
            "messages": [
                {"role": "user", "content": sample["user"]},
                {"role": "assistant", "content": sample["assistant"]},
            ]
        })

    # Augment with template-generated variants
    while len(dataset) < num_samples:
        topic = random.choice(TOPICS)
        template = random.choice(QUESTION_VARIANTS)
        question = template.format(topic=topic)

        # Pick a random hand-crafted answer as a rough proxy
        # In production, each variant would have a manually-reviewed answer
        ref = random.choice(TERMINOLOGY_SAMPLES)
        answer = ref["assistant"]

        dataset.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        })

    random.shuffle(dataset)
    return dataset[:num_samples]


def save_dataset(dataset: list[dict[str, Any]], filepath: str) -> None:
    """Save dataset to a JSONL file in ChatML format.

    Args:
        dataset: List of training samples.
        filepath: Output JSONL path.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset = generate_terminology_dataset(500)
    save_dataset(dataset, "data/training/stage1_terminology.jsonl")
    print(f"Generated {len(dataset)} terminology alignment samples")
