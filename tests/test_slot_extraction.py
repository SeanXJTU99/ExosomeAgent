"""Tests for slot extraction patterns in slot_extractor.py."""

import pytest

from src.agent.nodes.slot_extractor import (
    _extract_sample_type,
    _extract_sample_count,
    _extract_downstream_exp,
    _extract_qc_requirements,
)


class TestSampleTypeExtraction:
    """Verify sample type regex patterns match expected inputs."""

    def test_extract_serum(self) -> None:
        stype, conf = _extract_sample_type("I have rat serum samples for exosome isolation")
        assert stype == "serum"
        assert conf > 0.8

    def test_extract_plasma(self) -> None:
        stype, conf = _extract_sample_type("Human plasma samples, EDTA tubes")
        assert stype == "plasma"
        assert conf > 0.8

    def test_extract_cell_supernatant(self) -> None:
        stype, conf = _extract_sample_type("We have cell culture supernatant from HEK293T")
        assert stype == "cell_supernatant"
        assert conf > 0.8

    def test_extract_urine(self) -> None:
        stype, conf = _extract_sample_type("Urine samples from 50 patients")
        assert stype == "urine"
        assert conf > 0.8

    def test_extract_csf(self) -> None:
        stype, conf = _extract_sample_type("Cerebrospinal fluid from lumbar puncture")
        assert stype == "csf"
        assert conf > 0.8

    def test_no_match_returns_empty(self) -> None:
        stype, conf = _extract_sample_type("Hello, I have a question about your services")
        assert stype == ""
        assert conf == 0.0


class TestSampleCountExtraction:
    """Verify sample count regex patterns."""

    def test_extract_simple_count(self) -> None:
        count, conf = _extract_sample_count("I have 12 samples")
        assert count == 12
        assert conf > 0.9

    def test_extract_count_with_chinese(self) -> None:
        count, conf = _extract_sample_count("我们有20个样本")
        assert count == 20
        assert conf > 0.9

    def test_extract_large_count(self) -> None:
        count, conf = _extract_sample_count("We have 50 specimens ready")
        assert count == 50
        assert conf > 0.9

    def test_no_count_returns_zero(self) -> None:
        count, conf = _extract_sample_count("I want to know the price")
        assert count == 0
        assert conf == 0.0

    def test_unreasonably_large_count(self) -> None:
        count, conf = _extract_sample_count("I have 99999 samples")
        assert conf < 0.5  # Suspiciously large


class TestDownstreamExpExtraction:
    """Verify downstream experiment pattern matching."""

    def test_extract_mirna_seq(self) -> None:
        dexp, conf = _extract_downstream_exp("We want miRNA sequencing of exosomes")
        assert dexp == "mirna_seq"
        assert conf > 0.8

    def test_extract_rna_seq(self) -> None:
        dexp, conf = _extract_downstream_exp("Total RNA-seq of exosomal RNA")
        assert dexp == "rna_seq"

    def test_extract_proteomics(self) -> None:
        dexp, conf = _extract_downstream_exp("Mass spectrometry proteomics analysis")
        assert dexp == "proteomics"

    def test_extract_drug_delivery(self) -> None:
        dexp, conf = _extract_downstream_exp("Exosome drug delivery for tumor targeting")
        assert dexp == "drug_delivery"

    def test_no_match_returns_empty(self) -> None:
        dexp, conf = _extract_downstream_exp("What services do you offer?")
        assert dexp == ""
        assert conf == 0.0


class TestQCExtraction:
    """Verify QC requirements pattern matching."""

    def test_extract_nta(self) -> None:
        qc, conf = _extract_qc_requirements("We need NTA analysis")
        assert "NTA" in qc
        assert conf > 0.5

    def test_extract_tem(self) -> None:
        qc, conf = _extract_qc_requirements("TEM imaging required")
        assert "TEM" in qc

    def test_extract_western_blot(self) -> None:
        qc, conf = _extract_qc_requirements("Western Blot for CD63 and CD81")
        assert "Western Blot" in qc

    def test_extract_multiple_qc(self) -> None:
        qc, conf = _extract_qc_requirements("We want NTA, TEM, and Western Blot — all three")
        assert len(qc) == 3
        assert conf > 0.8

    def test_no_qc_returns_empty(self) -> None:
        qc, conf = _extract_qc_requirements("No specific QC needed")
        assert qc == []
        assert conf == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
