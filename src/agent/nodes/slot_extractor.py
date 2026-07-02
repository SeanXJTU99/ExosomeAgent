"""Multi-turn slot extraction node.

Extracts structured business parameters (sample type, count, downstream
experiment, QC requirements) from the conversation history using the
fine-tuned LLM's structured output capability.

Outputs a JSON object that updates the agent state slots. This is the
critical node that converts unstructured conversation into actionable
structured data for downstream routing.
"""

import json
import re

from src.agent.state import ExosomeAgentState


# ---------- Slot extraction patterns (regex fallback before LLM is available) ----------

SAMPLE_TYPE_PATTERNS: dict[str, re.Pattern] = {
    "serum": re.compile(r"(?:大鼠|小鼠|人|rat|mouse|human|mice|patient)?\s*(?:的)?\s*血清|serum", re.IGNORECASE),
    "plasma": re.compile(r"(?:大鼠|小鼠|人|rat|mouse|human|mice|patient)?\s*(?:的)?\s*血浆|plasma", re.IGNORECASE),
    "cell_supernatant": re.compile(
        r"细胞上清|细胞培养(?:上清|液|基)|conditioned\s*medium|"
        r"culture\s*(?:supernatant|medium)|supernatant",
        re.IGNORECASE,
    ),
    "urine": re.compile(r"尿液|urine", re.IGNORECASE),
    "csf": re.compile(r"脑脊液|cerebrospinal\s*fluid|CSF", re.IGNORECASE),
    "saliva": re.compile(r"唾液|saliva", re.IGNORECASE),
    "tissue": re.compile(r"组织|tissue", re.IGNORECASE),
}

DOWNSTREAM_PATTERNS: dict[str, re.Pattern] = {
    "mirna_seq": re.compile(
        r"miRNA[-\s]?seq|小RNA测序|microRNA|miR", re.IGNORECASE
    ),
    "rna_seq": re.compile(
        r"(?:total\s+)?RNA[-\s]?seq|转录组|transcriptom|mRNA[-\s]?seq", re.IGNORECASE
    ),
    "proteomics": re.compile(
        r"蛋白组|proteom|质谱|mass\s*spec|LC[-\s]MS|TMT|label[-\s]free|DIA|DDA",
        re.IGNORECASE,
    ),
    "drug_delivery": re.compile(
        r"药物(?:递送|载体|装载|加载)|drug\s*(?:delivery|loading|carrier)|靶向|targeting",
        re.IGNORECASE,
    ),
    "functional": re.compile(
        r"功能(?:实验|验证|研究)|functional\s*(?:assay|study|validation|experiment)|"
        r"uptake|摄取|proliferation|增殖|migration|迁移|invasion|侵袭",
        re.IGNORECASE,
    ),
    "biomarker": re.compile(
        r"标志物|biomarker|诊断|diagnos|prognos|预后|筛选|screen",
        re.IGNORECASE,
    ),
}

QC_PATTERNS: dict[str, re.Pattern] = {
    "NTA": re.compile(r"NTA|nanoparticle\s*tracking|粒径|particle\s*size|浓度测定", re.IGNORECASE),
    "TEM": re.compile(r"TEM|transmission\s*electron|透射电镜|电镜|morphology|形貌", re.IGNORECASE),
    "Western Blot": re.compile(
        r"WB|Western\s*Blot|蛋白印迹|CD63|CD9|CD81|TSG101|Calnexin|marker",
        re.IGNORECASE,
    ),
}

SAMPLE_COUNT_PATTERN = re.compile(
    r"(\d+)\s*(?:个|份|例|管|支|样|samples?|specimens?|cases?|tubes?)",
    re.IGNORECASE,
)

# Confidence threshold for extracted data
MIN_CONFIDENCE: float = 0.6


def _extract_sample_type(text: str) -> tuple[str, float]:
    """Extract sample type from text using regex patterns.

    Args:
        text: User message text to scan.

    Returns:
        Tuple of (sample_type, confidence).
    """
    for stype, pattern in SAMPLE_TYPE_PATTERNS.items():
        if pattern.search(text):
            return stype, 0.9
    return "", 0.0


def _extract_downstream_exp(text: str) -> tuple[str, float]:
    """Extract downstream experiment goal from text.

    Args:
        text: User message text to scan.

    Returns:
        Tuple of (experiment_type, confidence).
    """
    for dexp, pattern in DOWNSTREAM_PATTERNS.items():
        if pattern.search(text):
            return dexp, 0.85
    return "", 0.0


def _extract_qc_requirements(text: str) -> tuple[list[str], float]:
    """Extract QC requirements from text.

    Args:
        text: User message text to scan.

    Returns:
        Tuple of (qc_list, confidence).
    """
    found: list[str] = []
    for qc_name, pattern in QC_PATTERNS.items():
        if pattern.search(text):
            found.append(qc_name)
    confidence: float = min(0.9, 0.5 + 0.2 * len(found)) if found else 0.0
    return found, confidence


def _extract_sample_count(text: str) -> tuple[int, float]:
    """Extract sample count from text.

    Args:
        text: User message text to scan.

    Returns:
        Tuple of (count, confidence). count is 0 if not found.
    """
    match = SAMPLE_COUNT_PATTERN.search(text)
    if match:
        count = int(match.group(1))
        # Sanity check — unreasonable numbers
        if 1 <= count <= 10000:
            return count, 0.95
        return count, 0.3  # Suspiciously large or zero
    return 0, 0.0


def _build_slot_json(state: ExosomeAgentState, text: str) -> dict:
    """Extract all slots from text and merge with existing state values.

    Existing state values take precedence (already validated in prior turns).
    Only fills in slots that are currently empty.

    Args:
        state: Current agent state.
        text: Full conversation text to extract from.

    Returns:
        Dict with updated slot values and a confidence score.
    """
    updates: dict = {}
    total_confidence: float = 1.0
    filled_count: int = 0

    # Sample type
    if not state.get("sample_type"):
        stype, conf = _extract_sample_type(text)
        if stype:
            updates["sample_type"] = stype
            total_confidence = min(total_confidence, conf)
            filled_count += 1

    # Sample count
    if not state.get("sample_count"):
        count, conf = _extract_sample_count(text)
        if count > 0:
            updates["sample_count"] = count
            total_confidence = min(total_confidence, conf)
            filled_count += 1

    # Downstream experiment
    if not state.get("downstream_exp"):
        dexp, conf = _extract_downstream_exp(text)
        if dexp:
            updates["downstream_exp"] = dexp
            total_confidence = min(total_confidence, conf)
            filled_count += 1

    # QC requirements
    existing_qc: list[str] = list(state.get("qc_requirements", []))
    qc_list, conf = _extract_qc_requirements(text)
    new_qc = [q for q in qc_list if q not in existing_qc]
    if new_qc:
        updates["qc_requirements"] = existing_qc + new_qc
        total_confidence = min(total_confidence, conf)
        filled_count += 1

    if filled_count == 0:
        total_confidence = 0.0

    updates["confidence_score"] = total_confidence
    return updates


def slot_extractor_node(state: ExosomeAgentState) -> dict:
    """LangGraph node: extract structured slots from conversation.

    Scans the full message history for business parameters and updates
    the state with any new information found. Sets confidence_score
    based on extraction reliability.

    In production, this would call the fine-tuned LLM with function calling
    or structured output (JSON mode). The regex-based implementation here
    serves as a fast fallback and demonstrates the expected interface.

    Args:
        state: Current agent state (after technical consultation).

    Returns:
        Partial state dict with updated slot fields and confidence_score.
    """
    messages: list = state.get("messages", [])

    # Concatenate all message content for scanning
    full_text_parts: list[str] = []
    for msg in messages:
        content: str = msg.content if hasattr(msg, "content") else str(msg)
        full_text_parts.append(content)
    full_text: str = "\n".join(full_text_parts)

    # Extract slots
    updates: dict = _build_slot_json(state, full_text)

    # Build a structured JSON summary for downstream logging/debugging
    extracted_summary: dict[str, object] = {
        "sample_type": updates.get("sample_type", state.get("sample_type", "")),
        "sample_count": updates.get("sample_count", state.get("sample_count", 0)),
        "downstream_exp": updates.get("downstream_exp", state.get("downstream_exp", "")),
        "qc_requirements": updates.get("qc_requirements", state.get("qc_requirements", [])),
        "confidence_score": updates.get("confidence_score", 0.0),
    }

    # Store the JSON snapshot in rag_context temporarily for downstream inspection
    updates["rag_context"] = json.dumps(extracted_summary, ensure_ascii=False, indent=2)

    return updates
