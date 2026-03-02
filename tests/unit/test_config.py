"""Unit tests for MechLens config module."""

import pytest

from mechlens.config import (
    DEFAULT_ANALYSIS_CONFIG,
    DEFAULT_INTERVENTION_CONFIG,
    DEFAULT_VISUALIZATION_CONFIG,
    MODEL_ALIASES,
    SUPPORTED_MODELS,
    AnalysisConfig,
    InterventionConfig,
    ModelMetadata,
    VisualizationConfig,
    get_model_config,
    get_model_metadata,
    list_supported_models,
)
from mechlens.types import MLPType


class TestSupportedModels:
    """Test model registry."""

    def test_all_models_registered(self):
        assert "Qwen/Qwen2.5-0.5B" in SUPPORTED_MODELS
        assert "Qwen/Qwen2.5-7B" in SUPPORTED_MODELS
        assert "meta-llama/Llama-3.1-8B" in SUPPORTED_MODELS
        assert "EleutherAI/pythia-1.4b" in SUPPORTED_MODELS

    def test_qwen_05b_metadata(self):
        meta = SUPPORTED_MODELS["Qwen/Qwen2.5-0.5B"]
        assert meta.n_layers == 24
        assert meta.n_heads == 14
        assert meta.n_kv_heads == 2
        assert meta.d_model == 896
        assert meta.mlp_type == MLPType.SWIGLU
        assert meta.supports_chinese_bench is True

    def test_qwen_7b_metadata(self):
        meta = SUPPORTED_MODELS["Qwen/Qwen2.5-7B"]
        assert meta.n_layers == 28
        assert meta.n_heads == 28
        assert meta.n_kv_heads == 4
        assert meta.d_model == 3584
        assert meta.vram_int8_gb == 13.0

    def test_pythia_metadata(self):
        meta = SUPPORTED_MODELS["EleutherAI/pythia-1.4b"]
        assert meta.n_layers == 24
        assert meta.n_heads == 16
        assert meta.n_kv_heads == 16  # MHA
        assert meta.d_model == 2048
        assert meta.mlp_type == MLPType.GELU
        assert meta.supports_chinese_bench is False

    def test_llama_metadata(self):
        meta = SUPPORTED_MODELS["meta-llama/Llama-3.1-8B"]
        assert meta.n_layers == 32
        assert meta.n_kv_heads == 8  # GQA
        assert meta.supports_rome_memit is False


class TestModelAliases:
    """Test model name aliases."""

    def test_aliases_exist(self):
        assert "qwen-0.5b" in MODEL_ALIASES
        assert "qwen-7b" in MODEL_ALIASES
        assert "llama-8b" in MODEL_ALIASES
        assert "pythia" in MODEL_ALIASES
        assert "pythia-1.4b" in MODEL_ALIASES

    def test_aliases_resolve(self):
        assert MODEL_ALIASES["qwen-0.5b"] == "Qwen/Qwen2.5-0.5B"
        assert MODEL_ALIASES["qwen-7b"] == "Qwen/Qwen2.5-7B"
        assert MODEL_ALIASES["pythia"] == "EleutherAI/pythia-1.4b"


class TestGetModelMetadata:
    """Test get_model_metadata function."""

    def test_direct_name(self):
        meta = get_model_metadata("Qwen/Qwen2.5-0.5B")
        assert meta.display_name == "Qwen2.5-0.5B"

    def test_alias(self):
        meta = get_model_metadata("pythia")
        assert meta.hf_name == "EleutherAI/pythia-1.4b"

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="not supported"):
            get_model_metadata("unknown/model")


class TestGetModelConfig:
    """Test get_model_config function."""

    def test_creates_config(self):
        cfg = get_model_config("qwen-0.5b")
        assert cfg.model_name == "Qwen/Qwen2.5-0.5B"
        assert cfg.n_layers == 24
        assert cfg.mlp_type == MLPType.SWIGLU

    def test_custom_dtype_device(self):
        cfg = get_model_config("pythia", dtype="bfloat16", device="cpu")
        assert cfg.dtype == "bfloat16"
        assert cfg.device == "cpu"


class TestDefaultConfigs:
    """Test default configuration values."""

    def test_intervention_defaults(self):
        assert DEFAULT_INTERVENTION_CONFIG.default_ablation_value == 0.0
        assert DEFAULT_INTERVENTION_CONFIG.default_scaling_factor == 1.0
        assert DEFAULT_INTERVENTION_CONFIG.max_batch_size == 32

    def test_visualization_defaults(self):
        assert DEFAULT_VISUALIZATION_CONFIG.default_colorscale == "RdBu"
        assert DEFAULT_VISUALIZATION_CONFIG.export_format == "pdf"
        assert DEFAULT_VISUALIZATION_CONFIG.export_dpi == 300

    def test_analysis_defaults(self):
        assert DEFAULT_ANALYSIS_CONFIG.include_logit_lens is False
        assert DEFAULT_ANALYSIS_CONFIG.causal_trace_noise_level == 0.1
        assert DEFAULT_ANALYSIS_CONFIG.sae_top_k_features == 20


class TestListSupportedModels:
    """Test list_supported_models function."""

    def test_returns_list(self):
        models = list_supported_models()
        assert isinstance(models, list)
        assert len(models) == 4

    def test_model_info_keys(self):
        models = list_supported_models()
        for m in models:
            assert "name" in m
            assert "display_name" in m
            assert "n_layers" in m
            assert "n_heads" in m
            assert "mlp_type" in m
            assert "vram_fp16_gb" in m
