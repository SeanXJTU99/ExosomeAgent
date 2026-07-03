"""Tests for the intent classification router."""

import pytest

from src.agent.nodes.router import classify_intent


class TestIntentClassification:
    """Verify router classifies user messages into correct intent categories."""

    # ── Technical ──

    def test_mirna_technical(self) -> None:
        assert classify_intent("What method is best for exosome miRNA-seq?") == "technical"

    def test_qc_technical(self) -> None:
        assert classify_intent("How do you do NTA and TEM analysis?") == "technical"

    def test_marker_technical(self) -> None:
        assert classify_intent("What markers should I check for exosome WB?") == "technical"

    def test_extraction_technical(self) -> None:
        assert classify_intent("Ultracentrifugation vs SEC for exosome isolation") == "technical"

    def test_hemolysis_technical(self) -> None:
        assert classify_intent("Does hemolysis affect exosome RNA-seq results?") == "technical"

    def test_proteomics_technical(self) -> None:
        assert classify_intent("Can you do LC-MS proteomics on exosomes?") == "technical"

    # ── Commercial ──

    def test_price_commercial(self) -> None:
        assert classify_intent("What's the price for exosome sequencing?") == "commercial"

    def test_how_much_commercial(self) -> None:
        assert classify_intent("How much does exosome extraction cost?") == "commercial"

    def test_discount_commercial(self) -> None:
        assert classify_intent("Do you offer bulk discounts for 20 samples?") == "commercial"

    def test_quote_commercial(self) -> None:
        assert classify_intent("Can you give me a quote for my project?") == "commercial"

    # ── Logistics ──

    def test_ship_logistics(self) -> None:
        assert classify_intent("How do I ship my serum samples to you?") == "logistics"

    def test_send_logistics(self) -> None:
        assert classify_intent("How should I send my samples?") == "logistics"

    def test_dry_ice_logistics(self) -> None:
        assert classify_intent("What kind of dry ice packaging do I need?") == "logistics"

    def test_courier_logistics(self) -> None:
        assert classify_intent("Which courier should I use to deliver samples?") == "logistics"

    # ── Edge cases ──

    def test_empty_message(self) -> None:
        result = classify_intent("")
        assert result in ("technical", "commercial", "logistics", "fallback")

    def test_very_short_message(self) -> None:
        result = classify_intent("Hi")
        assert result in ("technical", "commercial", "logistics", "fallback")

    def test_ambiguous_short(self) -> None:
        """Ambiguous short messages default to technical for engagement."""
        assert classify_intent("help") == "technical"

    def test_ambiguous_price_pattern_wins(self) -> None:
        """Strong price patterns override keyword counting."""
        result = classify_intent("how much does exosome extraction cost per sample")
        assert result == "commercial"

    def test_ambiguous_shipping_pattern_wins(self) -> None:
        """Strong shipping patterns override keyword counting."""
        result = classify_intent("how should i send my samples to your lab")
        assert result == "logistics"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
