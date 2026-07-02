"""Logistics guide node.

Returns standardized sample shipping SOPs using hardcoded templates.
This node deliberately does NOT call the LLM — shipping instructions,
addresses, and packaging requirements must be deterministic.
"""

from langchain_core.messages import AIMessage

from src.agent.state import ExosomeAgentState

# ---------- Shipping SOP templates (EXAMPLE data) ----------

GENERAL_SHIPPING_GUIDE: str = (
    "## Sample Shipping Guidelines\n\n"
    "### Packaging Requirements\n"
    "1. **Primary container**: Use sterile, RNase/DNase-free cryovials "
    "(1.5–2.0 mL recommended). Label each vial clearly with sample ID, "
    "date, and your initials.\n"
    "2. **Secondary container**: Place cryovials in a sealed zip-lock bag "
    "with absorbent material to contain potential leaks.\n"
    "3. **Tertiary container**: Use a sturdy foam box or insulated shipping "
    "container.\n\n"
    "### Temperature Requirements\n"
    "- **Serum/Plasma/Urine**: Ship on **dry ice** (~5 kg per 24h transit). "
    "Ensure samples are frozen at -80°C before packing.\n"
    "- **Cell supernatant**: Pre-centrifuge at 2,000 × g for 10 min to remove "
    "cell debris. Ship on dry ice.\n"
    "- **Tissue**: Snap-freeze in liquid nitrogen immediately after collection. "
    "Ship on dry ice. Include a small piece for RNA integrity check (optional).\n"
    "- **CSF**: Ship on dry ice. Avoid repeated freeze-thaw cycles.\n\n"
    "### Documentation\n"
    "- Include a printed **Sample Manifest** listing all sample IDs, types, "
    "volumes, and collection dates.\n"
    "- If applicable, include **IACUC approval** or ethics committee documentation.\n\n"
    "### Courier\n"
    "- We recommend SF Express (顺丰冷链) or equivalent cold-chain courier.\n"
    "- Schedule pickup for Monday–Wednesday to avoid weekend delays.\n"
    "- Tracking number must be shared with your account manager.\n\n"
    "### Pre-shipment Checklist\n"
    "- [ ] All vials labeled and sealed\n"
    "- [ ] Dry ice sufficient for transit time + 12h buffer\n"
    "- [ ] Sample manifest completed\n"
    "- [ ] Courier booked and tracking number received\n"
    "- [ ] Account manager notified\n"
)

SERUM_PLASMA_SPECIFIC: str = (
    "## Serum / Plasma Sample Preparation (Additional Notes)\n\n"
    "### Pre-processing (CRITICAL)\n"
    "1. **Serum**: Collect blood in serum separator tubes. Allow clotting at "
    "room temperature for 30 min. Centrifuge at 3,000 × g for 15 min at 4°C. "
    "Transfer supernatant (serum) to a new tube.\n"
    "2. **Plasma**: Collect blood in EDTA tubes (preferred) or citrate tubes. "
    "**Do NOT use heparin tubes** — heparin interferes with downstream enzymatic "
    "reactions. Centrifuge at 3,000 × g for 15 min at 4°C within 30 min of "
    "collection.\n"
    "3. **Second spin** (recommended): Centrifuge the collected serum/plasma "
    "again at 3,000 × g for 10 min to remove residual cells and platelets.\n"
    "4. **Hemolysis check**: Visually inspect for pink/red discoloration. "
    "Hemolyzed samples (free hemoglobin > 0.3 mg/mL) will be flagged and may "
    "affect exosome RNA-seq results.\n"
    "5. Aliquot and freeze at -80°C. Avoid freeze-thaw cycles.\n\n"
    "### Minimum Volume\n"
    "- Serum: ≥ 1.0 mL per sample (1.5 mL recommended)\n"
    "- Plasma: ≥ 1.0 mL per sample (1.5 mL recommended)\n"
)

CELL_SUPERNATANT_SPECIFIC: str = (
    "## Cell Culture Supernatant Preparation (Additional Notes)\n\n"
    "### Pre-processing\n"
    "1. Culture cells to desired confluency (typically 70–80%).\n"
    "2. Wash cells 2–3× with PBS, then replace with **exosome-depleted FBS "
    "medium** (or serum-free medium) for the conditioning period (24–48h typical).\n"
    "3. Collect conditioned medium. Centrifuge at 300 × g for 10 min to remove "
    "cells, then at 2,000 × g for 10–20 min to remove cell debris.\n"
    "4. Optional: 0.22 μm filtration to remove larger microvesicles and apoptotic "
    "bodies before exosome isolation.\n"
    "5. If not processing immediately, snap-freeze in liquid nitrogen and store "
    "at -80°C.\n\n"
    "### Minimum Volume\n"
    "- Cell supernatant: ≥ 10 mL per sample (50 mL recommended for UC-based "
    "isolation)\n"
    "- For magnetic bead-based isolation: ≥ 1 mL\n"
)

URINE_SPECIFIC: str = (
    "## Urine Sample Preparation (Additional Notes)\n\n"
    "### Pre-processing\n"
    "1. Collect first morning void (most concentrated) or 24h urine.\n"
    "2. Add protease inhibitor cocktail (e.g., Roche cOmplete) within 30 min "
    "of collection.\n"
    "3. Centrifuge at 2,000 × g for 20 min at 4°C to remove cells, bacteria, "
    "and Tamm-Horsfall protein aggregates.\n"
    "4. Concentrate if needed (e.g., Amicon Ultra-15, 100 kDa cutoff).\n"
    "5. Aliquot and freeze at -80°C.\n\n"
    "### Minimum Volume\n"
    "- Urine: ≥ 50 mL before concentration (10 mL after concentration)\n"
)


SAMPLE_TYPE_SOPS: dict[str, str] = {
    "serum": SERUM_PLASMA_SPECIFIC,
    "plasma": SERUM_PLASMA_SPECIFIC,
    "cell_supernatant": CELL_SUPERNATANT_SPECIFIC,
    "urine": URINE_SPECIFIC,
}


def logistics_guide_node(state: ExosomeAgentState) -> dict:
    """LangGraph node: return shipping and sample preparation SOPs.

    Appends the general shipping guide plus any sample-type-specific
    instructions to the conversation.

    Args:
        state: Current agent state.

    Returns:
        Partial state dict with the logistics guide appended to messages.
    """
    sample_type: str = state.get("sample_type", "")
    parts: list[str] = [GENERAL_SHIPPING_GUIDE]

    if sample_type and sample_type in SAMPLE_TYPE_SOPS:
        parts.append(SAMPLE_TYPE_SOPS[sample_type])

    parts.append(
        "\n---\n"
        "*These SOPs are example guidelines for demonstration purposes. "
        "Please confirm specific requirements with your account manager "
        "before shipping.*"
    )

    full_guide: str = "\n\n".join(parts)
    return {"messages": [AIMessage(content=full_guide)]}
