"""Unit tests for MechLens benchmark module.

Tests dataset loading and evaluation helpers without GPU.
"""

import json
import pytest
import tempfile
from pathlib import Path

from mechlens.benchmark.chinese_hallucination import (
    _check_correct,
    load_dataset,
)
from mechlens.types import (
    HallucinationDomain,
    HallucinationType,
    UnsupportedModelError,
)


class TestCheckCorrect:
    """Test the _check_correct helper function."""

    def test_exact_match(self):
        assert _check_correct("北京", "北京") is True

    def test_containment(self):
        assert _check_correct("中国的首都是北京市", "北京") is True

    def test_no_match(self):
        assert _check_correct("上海是一个大城市", "北京") is False

    def test_case_insensitive(self):
        assert _check_correct("The answer is Paris", "paris") is True

    def test_numeric_match(self):
        assert _check_correct("The distance is about 384400 km", "384400") is True

    def test_empty_output(self):
        assert _check_correct("", "北京") is False

    def test_parenthetical(self):
        assert _check_correct("北京（Beijing）是首都", "北京（Beijing）") is True


class TestLoadDataset:
    """Test dataset loading."""

    def test_load_valid_dataset(self, tmp_path):
        data = {
            "samples": [
                {
                    "id": "test_001",
                    "question": "中国的首都是哪里？",
                    "ground_truth": "北京",
                    "hallucination_type": "factual_fabrication",
                    "domain": "common_sense",
                    "should_refuse": False,
                    "reference_sources": ["wiki"],
                },
                {
                    "id": "test_002",
                    "question": "水的化学式是什么？",
                    "ground_truth": "H2O",
                    "hallucination_type": "factual_fabrication",
                    "domain": "science",
                    "should_refuse": False,
                },
            ]
        }
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        samples = load_dataset(path)
        assert len(samples) == 2
        assert samples[0].id == "test_001"
        assert samples[0].question == "中国的首都是哪里？"
        assert samples[0].hallucination_type == HallucinationType.FACTUAL_FABRICATION
        assert samples[0].domain == HallucinationDomain.COMMON_SENSE
        assert samples[1].domain == HallucinationDomain.SCIENCE

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/dataset.json")

    def test_load_empty_dataset(self, tmp_path):
        data = {"samples": []}
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        samples = load_dataset(path)
        assert len(samples) == 0


class TestModelValidation:
    """Test model support validation."""

    def test_non_qwen_model_rejected(self):
        from mechlens.benchmark.chinese_hallucination import _validate_model_support

        with pytest.raises(UnsupportedModelError, match="not supported"):
            _validate_model_support("EleutherAI/pythia-1.4b")

    def test_qwen_model_accepted(self):
        from mechlens.benchmark.chinese_hallucination import _validate_model_support

        # Should not raise
        _validate_model_support("Qwen/Qwen2.5-0.5B")

    def test_qwen_by_pattern(self):
        from mechlens.benchmark.chinese_hallucination import _validate_model_support

        # Unknown Qwen variant should pass pattern check
        _validate_model_support("qwen-custom-model")
