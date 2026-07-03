"""Tests for the Pydantic output validator."""

import json

import pytest

from src.safety.pydantic_validator import (
    ExosomeConsultResult,
    _extract_json_block,
    _repair_json,
    validate_llm_output,
    build_validation_error_response,
)


class TestJSONExtraction:
    """Verify JSON block extraction from LLM output text."""

    def test_extract_fenced_json(self) -> None:
        text = 'Here is the result:\n```json\n{"is_academic_question": true}\n```'
        result = _extract_json_block(text)
        assert result == '{"is_academic_question": true}'

    def test_extract_fenced_no_lang(self) -> None:
        text = '```\n{"is_academic_question": false}\n```'
        result = _extract_json_block(text)
        assert result == '{"is_academic_question": false}'

    def test_extract_bare_json(self) -> None:
        text = 'Some text {"is_academic_question": true, "confidence_score": 0.9} more text'
        result = _extract_json_block(text)
        assert result is not None
        assert "is_academic_question" in result

    def test_no_json_returns_none(self) -> None:
        text = "No JSON here, just plain text."
        assert _extract_json_block(text) is None


class TestJSONRepair:
    """Verify JSON repair for common quantized model errors."""

    def test_remove_trailing_comma_in_object(self) -> None:
        broken = '{"key": "value",}'
        repaired = _repair_json(broken)
        assert repaired == '{"key": "value"}'

    def test_remove_trailing_comma_in_array(self) -> None:
        broken = '["a", "b",]'
        repaired = _repair_json(broken)
        assert repaired == '["a", "b"]'

    def test_preserve_valid_json(self) -> None:
        valid = '{"key": "value", "num": 42}'
        repaired = _repair_json(valid)
        assert repaired == valid


class TestLLMOutputValidation:
    """Verify end-to-end validation of LLM structured output."""

    def test_valid_output_passes(self) -> None:
        result_dict = {
            "is_academic_question": True,
            "extracted_sample_type": "serum",
            "extracted_sample_count": 12,
            "extracted_downstream_exp": "mirna_seq",
            "extracted_qc_requirements": ["NTA", "TEM"],
            "confidence_score": 0.9,
            "needs_clarification": False,
            "missing_fields": [],
        }
        text = "```json\n" + json.dumps(result_dict) + "\n```"
        result = validate_llm_output(text)
        assert result is not None
        assert result.extracted_sample_type == "serum"
        assert result.extracted_sample_count == 12
        assert result.confidence_score == 0.9

    def test_malformed_json_returns_none(self) -> None:
        text = "```json\n{this is not json}\n```"
        result = validate_llm_output(text)
        assert result is None

    def test_missing_fields_default(self) -> None:
        """Minimal valid JSON with only required fields."""
        text = '{"is_academic_question": true, "confidence_score": 0.5}'
        result = validate_llm_output(text)
        assert result is not None
        assert result.is_academic_question is True
        assert result.extracted_sample_type is None  # Optional, not provided

    def test_no_json_block_returns_none(self) -> None:
        text = "I'm sorry, I cannot process this request."
        result = validate_llm_output(text)
        assert result is None


class TestExosomeConsultResult:
    """Verify the Pydantic model validation rules."""

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            ExosomeConsultResult(
                is_academic_question=True,
                confidence_score=1.5,  # > 1.0
            )

    def test_sample_count_bounds(self) -> None:
        with pytest.raises(Exception):
            ExosomeConsultResult(
                is_academic_question=True,
                confidence_score=0.5,
                extracted_sample_count=-1,  # negative
            )

    def test_default_values(self) -> None:
        result = ExosomeConsultResult(is_academic_question=True)
        assert result.confidence_score == 0.0
        assert result.extracted_sample_type is None
        assert result.extracted_qc_requirements == []
        assert result.needs_clarification is False


class TestBuildErrorResponse:
    """Verify fallback error response generation."""

    def test_returns_unknown_sample_type(self) -> None:
        resp = build_validation_error_response("bad json", "parse error")
        assert resp["extracted_sample_type"] == "UNKNOWN"
        assert resp["confidence_score"] == 0.0
        assert len(resp["missing_fields"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
