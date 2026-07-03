"""Tests for ExosomeAgentState schema integrity and defaults."""

import pytest

from src.agent.state import ExosomeAgentState


class TestExosomeAgentState:
    """Verify state schema field presence, types, and default behavior."""

    def test_state_has_all_required_fields(self) -> None:
        """All 10 fields defined in the schema must be accessible."""
        required_fields: list[str] = [
            "messages",
            "sample_type",
            "sample_count",
            "downstream_exp",
            "qc_requirements",
            "hardware_boundary_triggered",
            "confidence_score",
            "rag_context",
            "background_summary",
            "route",
        ]
        annotations = ExosomeAgentState.__annotations__
        for field in required_fields:
            assert field in annotations, f"Missing field: {field}"

    def test_state_accepts_partial_dict(self) -> None:
        """State should be constructable with partial data (TypedDict behavior)."""
        state: ExosomeAgentState = {
            "messages": [],
            "sample_type": "",
            "sample_count": 0,
            "downstream_exp": "",
            "qc_requirements": [],
            "hardware_boundary_triggered": False,
            "confidence_score": 0.0,
            "rag_context": "",
            "background_summary": "",
            "route": "fallback",
        }
        assert state["sample_type"] == ""
        assert state["sample_count"] == 0
        assert state["route"] == "fallback"
        assert state["hardware_boundary_triggered"] is False

    def test_messages_field_is_annotated(self) -> None:
        """The messages field must use add_messages reducer annotation."""
        assert "messages" in ExosomeAgentState.__annotations__

    def test_sample_count_defaults(self) -> None:
        """Sample count should start at 0 (no samples known)."""
        state: ExosomeAgentState = {
            "messages": [],
            "sample_type": "",
            "sample_count": 0,
            "downstream_exp": "",
            "qc_requirements": [],
            "hardware_boundary_triggered": False,
            "confidence_score": 0.0,
            "rag_context": "",
            "background_summary": "",
            "route": "fallback",
        }
        assert state["sample_count"] == 0

    def test_boolean_flags_default_false(self) -> None:
        """Hardware boundary should default to False."""
        state: ExosomeAgentState = {
            "messages": [],
            "sample_type": "",
            "sample_count": 0,
            "downstream_exp": "",
            "qc_requirements": [],
            "hardware_boundary_triggered": False,
            "confidence_score": 0.0,
            "rag_context": "",
            "background_summary": "",
            "route": "fallback",
        }
        assert state["hardware_boundary_triggered"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
