"""MechLens type definitions.

All dataclasses and enums used across the MechLens package.
Per data-model.md specification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import torch


class AnalysisType(Enum):
    """Type of analysis performed."""

    ATTENTION = "attention"
    ACTIVATION = "activation"
    CAUSAL_TRACE = "causal_trace"
    CIRCUIT = "circuit"
    FEATURE_DECOMPOSITION = "feature_decomposition"
    LOGIT_LENS = "logit_lens"
    FULL = "full"


class InterventionStatus(Enum):
    """Status of an intervention operation."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class InterventionType(Enum):
    """Type of activation intervention."""

    ABLATION = "ablation"
    SCALING = "scaling"
    INJECTION = "injection"


class ComponentType(Enum):
    """Type of model component for intervention."""

    ATTN_HEAD = "attn_head"
    MLP_NEURON = "mlp_neuron"
    RESID = "resid"


class MLPType(Enum):
    """MLP architecture type."""

    GELU = "gelu"
    SWIGLU = "swiglu"


class HallucinationType(Enum):
    """Type of hallucination in ChineseHallucinationBench."""

    FACTUAL_FABRICATION = "factual_fabrication"
    CAUSAL_ERROR = "causal_error"
    TEMPORAL_DISPLACEMENT = "temporal_displacement"
    IDENTITY_CONFUSION = "identity_confusion"


class HallucinationDomain(Enum):
    """Domain of hallucination sample."""

    HISTORY = "history"
    MEDICINE = "medicine"
    SCIENCE = "science"
    COMMON_SENSE = "common_sense"


@dataclass
class ModelConfig:
    """Model configuration for loading and analysis."""

    model_name: str
    dtype: str = "float16"
    device: str = "cuda"
    n_layers: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    d_model: int | None = None
    mlp_type: MLPType = MLPType.GELU
    supports_rome_memit: bool = False
    supports_chinese_bench: bool = False
    supports_icl_study: bool = False


@dataclass
class AnalysisResult:
    """Container for a single analysis run."""

    id: str
    model_config: ModelConfig
    input_text: str
    input_tokens: list[str]
    timestamp: datetime
    analysis_type: AnalysisType


@dataclass
class AttentionData:
    """Attention pattern data."""

    patterns: torch.Tensor  # [layers, heads, seq, seq]
    head_labels: list[str]
    qk_scores: torch.Tensor | None = None


@dataclass
class ActivationData:
    """Activation distribution data."""

    residual_stream: torch.Tensor  # [layers, seq, d_model]
    mlp_output: torch.Tensor  # [layers, seq, d_model]
    attn_output: torch.Tensor  # [layers, seq, d_model]
    logit_lens: torch.Tensor | None = None  # [layers, seq, vocab]


@dataclass
class CausalTraceResult:
    """Causal tracing result."""

    base_output: str
    corrupted_output: str
    patch_results: torch.Tensor  # [layers, components]
    component_type: str
    target_token_idx: int


@dataclass
class CircuitNode:
    """A node in a circuit graph."""

    id: str  # e.g., "L3H7" = Layer 3 Head 7
    layer: int
    component_type: str  # "attn_head", "mlp", "embed", "unembed"
    importance: float


@dataclass
class CircuitEdge:
    """An edge in a circuit graph."""

    source: str
    target: str
    weight: float


@dataclass
class CircuitGraph:
    """Circuit graph structure."""

    nodes: list[CircuitNode]
    edges: list[CircuitEdge]
    faithfulness: float
    completeness: float


@dataclass
class SAEFeature:
    """Sparse autoencoder feature."""

    feature_idx: int
    layer: int
    activation: float
    description: str | None = None
    top_examples: list[str] = field(default_factory=list)


@dataclass
class InterventionTarget:
    """Target specification for an intervention."""

    layer: int
    component_type: ComponentType
    component_id: int | None = None
    factor: float | None = None  # For scaling
    source_activation: torch.Tensor | None = None  # For injection


@dataclass
class InterventionResult:
    """Result of an intervention operation."""

    original_output: str
    intervened_output: str
    activation_diff: ActivationData | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionOperation:
    """Record of an intervention operation."""

    id: str
    intervention_type: InterventionType
    targets: list[InterventionTarget]
    status: InterventionStatus
    result: InterventionResult | None = None


@dataclass
class EditMetrics:
    """Metrics for a model edit operation."""

    es: float  # Edit Success rate
    ps: float  # Paraphrase Success rate
    ns: float  # Neighborhood Specificity


@dataclass
class EditOperation:
    """Record of a ROME/MEMIT edit operation."""

    id: str
    method: str  # "ROME" or "MEMIT"
    subject: str
    target_old: str
    target_new: str
    layers_edited: list[int]
    success_metrics: EditMetrics


@dataclass
class HallucinationSample:
    """A sample from ChineseHallucinationBench."""

    id: str
    question: str
    ground_truth: str
    hallucination_type: HallucinationType
    domain: HallucinationDomain
    should_refuse: bool
    reference_sources: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of comparing multiple intervention strategies."""

    strategies: list[dict[str, Any]]
    per_strategy_metrics: list[dict[str, Any]]
    ranking: list[str]  # Strategy IDs ranked by primary metric
    diff_table: dict[str, Any]


class ModelLoadError(Exception):
    """Error loading a model."""

    pass


class UnsupportedModelError(Exception):
    """Model not supported for this operation."""

    pass


class ShapeMismatchError(Exception):
    """Tensor shape mismatch error."""

    pass


class InterventionError(Exception):
    """Error during intervention operation."""

    pass
