"""Unit tests for MechLens type definitions."""

import pytest
import torch

from mechlens.types import (
    ActivationData,
    AnalysisType,
    AttentionData,
    CausalTraceResult,
    CircuitEdge,
    CircuitGraph,
    CircuitNode,
    ComparisonResult,
    ComponentType,
    EditMetrics,
    EditOperation,
    HallucinationDomain,
    HallucinationSample,
    HallucinationType,
    InterventionError,
    InterventionOperation,
    InterventionResult,
    InterventionStatus,
    InterventionTarget,
    InterventionType,
    MLPType,
    ModelConfig,
    ModelLoadError,
    SAEFeature,
    ShapeMismatchError,
    UnsupportedModelError,
)


class TestEnums:
    """Test enum types."""

    def test_analysis_type_values(self):
        assert AnalysisType.ATTENTION.value == "attention"
        assert AnalysisType.ACTIVATION.value == "activation"
        assert AnalysisType.CAUSAL_TRACE.value == "causal_trace"
        assert AnalysisType.CIRCUIT.value == "circuit"
        assert AnalysisType.LOGIT_LENS.value == "logit_lens"
        assert AnalysisType.FULL.value == "full"

    def test_component_type_values(self):
        assert ComponentType.ATTN_HEAD.value == "attn_head"
        assert ComponentType.MLP_NEURON.value == "mlp_neuron"
        assert ComponentType.RESID.value == "resid"

    def test_intervention_type_values(self):
        assert InterventionType.ABLATION.value == "ablation"
        assert InterventionType.SCALING.value == "scaling"
        assert InterventionType.INJECTION.value == "injection"

    def test_intervention_status_values(self):
        assert InterventionStatus.PENDING.value == "pending"
        assert InterventionStatus.COMPLETED.value == "completed"
        assert InterventionStatus.FAILED.value == "failed"

    def test_mlp_type_values(self):
        assert MLPType.GELU.value == "gelu"
        assert MLPType.SWIGLU.value == "swiglu"

    def test_hallucination_type_values(self):
        assert HallucinationType.FACTUAL_FABRICATION.value == "factual_fabrication"
        assert HallucinationType.CAUSAL_ERROR.value == "causal_error"
        assert HallucinationType.TEMPORAL_DISPLACEMENT.value == "temporal_displacement"
        assert HallucinationType.IDENTITY_CONFUSION.value == "identity_confusion"

    def test_hallucination_domain_values(self):
        assert HallucinationDomain.HISTORY.value == "history"
        assert HallucinationDomain.MEDICINE.value == "medicine"
        assert HallucinationDomain.SCIENCE.value == "science"
        assert HallucinationDomain.COMMON_SENSE.value == "common_sense"


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_defaults(self):
        cfg = ModelConfig(model_name="test")
        assert cfg.model_name == "test"
        assert cfg.dtype == "float16"
        assert cfg.device == "cuda"
        assert cfg.n_layers is None
        assert cfg.mlp_type == MLPType.GELU
        assert cfg.supports_rome_memit is False

    def test_custom_values(self):
        cfg = ModelConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            dtype="bfloat16",
            device="cpu",
            n_layers=24,
            n_heads=14,
            n_kv_heads=2,
            d_model=896,
            mlp_type=MLPType.SWIGLU,
            supports_chinese_bench=True,
        )
        assert cfg.n_layers == 24
        assert cfg.mlp_type == MLPType.SWIGLU
        assert cfg.supports_chinese_bench is True


class TestInterventionTarget:
    """Test InterventionTarget dataclass."""

    def test_minimal(self):
        t = InterventionTarget(layer=5, component_type=ComponentType.RESID)
        assert t.layer == 5
        assert t.component_type == ComponentType.RESID
        assert t.component_id is None
        assert t.factor is None

    def test_with_head(self):
        t = InterventionTarget(
            layer=3,
            component_type=ComponentType.ATTN_HEAD,
            component_id=7,
            factor=0.5,
        )
        assert t.component_id == 7
        assert t.factor == 0.5


class TestInterventionResult:
    """Test InterventionResult dataclass."""

    def test_minimal(self):
        r = InterventionResult(
            original_output="Paris",
            intervened_output="Berlin",
        )
        assert r.original_output == "Paris"
        assert r.intervened_output == "Berlin"
        assert r.activation_diff is None
        assert r.metrics == {}

    def test_with_metrics(self):
        r = InterventionResult(
            original_output="Paris",
            intervened_output="Berlin",
            metrics={"kl_divergence": 0.5, "logit_diff": 1.2},
        )
        assert r.metrics["kl_divergence"] == 0.5


class TestActivationData:
    """Test ActivationData dataclass."""

    def test_creation(self):
        n_layers, seq_len, d_model = 24, 10, 896
        data = ActivationData(
            residual_stream=torch.randn(n_layers, seq_len, d_model),
            mlp_output=torch.randn(n_layers, seq_len, d_model),
            attn_output=torch.randn(n_layers, seq_len, d_model),
        )
        assert data.residual_stream.shape == (24, 10, 896)
        assert data.logit_lens is None


class TestAttentionData:
    """Test AttentionData dataclass."""

    def test_creation(self):
        data = AttentionData(
            patterns=torch.randn(24, 14, 10, 10),
            head_labels=[f"L{l}H{h}" for l in range(24) for h in range(14)],
        )
        assert data.patterns.shape == (24, 14, 10, 10)
        assert data.qk_scores is None


class TestCausalTraceResult:
    """Test CausalTraceResult dataclass."""

    def test_creation(self):
        result = CausalTraceResult(
            base_output="Paris",
            corrupted_output="random",
            patch_results=torch.randn(24),
            component_type="mlp",
            target_token_idx=-1,
        )
        assert result.patch_results.shape == (24,)
        assert result.component_type == "mlp"


class TestCircuitGraph:
    """Test CircuitGraph dataclass."""

    def test_creation(self):
        nodes = [
            CircuitNode(id="L0H0", layer=0, component_type="attn_head", importance=0.9),
            CircuitNode(id="MLP3", layer=3, component_type="mlp", importance=0.7),
        ]
        edges = [
            CircuitEdge(source="L0H0", target="MLP3", weight=0.85),
        ]
        graph = CircuitGraph(
            nodes=nodes, edges=edges, faithfulness=0.95, completeness=0.8
        )
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.faithfulness == 0.95


class TestHallucinationSample:
    """Test HallucinationSample dataclass."""

    def test_creation(self):
        sample = HallucinationSample(
            id="test_001",
            question="中国的首都是哪里？",
            ground_truth="北京",
            hallucination_type=HallucinationType.FACTUAL_FABRICATION,
            domain=HallucinationDomain.COMMON_SENSE,
            should_refuse=False,
        )
        assert sample.id == "test_001"
        assert sample.reference_sources == []


class TestEditMetrics:
    """Test EditMetrics dataclass."""

    def test_creation(self):
        m = EditMetrics(es=0.95, ps=0.88, ns=0.92)
        assert m.es == 0.95
        assert m.ps == 0.88
        assert m.ns == 0.92


class TestSAEFeature:
    """Test SAEFeature dataclass."""

    def test_defaults(self):
        f = SAEFeature(feature_idx=42, layer=5, activation=3.7)
        assert f.description is None
        assert f.top_examples == []


class TestExceptions:
    """Test custom exceptions."""

    def test_model_load_error(self):
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("CUDA OOM")

    def test_unsupported_model_error(self):
        with pytest.raises(UnsupportedModelError):
            raise UnsupportedModelError("Qwen only")

    def test_shape_mismatch_error(self):
        with pytest.raises(ShapeMismatchError):
            raise ShapeMismatchError("Expected [10, 896]")

    def test_intervention_error(self):
        with pytest.raises(InterventionError):
            raise InterventionError("Factor must be >= 0")
