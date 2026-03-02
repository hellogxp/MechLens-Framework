"""MechLens configuration module.

SUPPORTED_MODELS registry and default settings for all 4 model families.
Per plan.md and R9 model coverage matrix.
"""

from dataclasses import dataclass
from typing import Any

from mechlens.types import MLPType, ModelConfig


@dataclass
class ModelMetadata:
    """Metadata for a supported model."""

    hf_name: str
    display_name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    mlp_type: MLPType
    supports_rome_memit: bool
    supports_chinese_bench: bool
    supports_icl_study: bool
    vram_fp16_gb: float
    vram_int8_gb: float | None = None


# Model registry with metadata per R9 model coverage matrix
SUPPORTED_MODELS: dict[str, ModelMetadata] = {
    "Qwen/Qwen2.5-0.5B": ModelMetadata(
        hf_name="Qwen/Qwen2.5-0.5B",
        display_name="Qwen2.5-0.5B",
        n_layers=24,
        n_heads=14,
        n_kv_heads=2,
        d_model=896,
        mlp_type=MLPType.SWIGLU,
        supports_rome_memit=True,
        supports_chinese_bench=True,
        supports_icl_study=True,
        vram_fp16_gb=2.0,
    ),
    "Qwen/Qwen2.5-7B": ModelMetadata(
        hf_name="Qwen/Qwen2.5-7B",
        display_name="Qwen2.5-7B",
        n_layers=28,
        n_heads=28,
        n_kv_heads=4,
        d_model=3584,
        mlp_type=MLPType.SWIGLU,
        supports_rome_memit=True,
        supports_chinese_bench=True,
        supports_icl_study=True,
        vram_fp16_gb=20.0,
        vram_int8_gb=13.0,
    ),
    # Qwen2.5-14B for cross-scale Late Crystallization validation (Section 4.3)
    "Qwen/Qwen2.5-14B": ModelMetadata(
        hf_name="Qwen/Qwen2.5-14B",
        display_name="Qwen2.5-14B",
        n_layers=48,
        n_heads=40,
        n_kv_heads=8,  # GQA with 8 KV heads per paper Table 1
        d_model=5120,
        mlp_type=MLPType.SWIGLU,
        supports_rome_memit=True,
        supports_chinese_bench=True,
        supports_icl_study=False,  # Not used in ICL study per paper
        vram_fp16_gb=35.0,
        vram_int8_gb=20.0,
    ),
    "meta-llama/Llama-3.1-8B": ModelMetadata(
        hf_name="meta-llama/Llama-3.1-8B",
        display_name="Llama 3.1-8B",
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        d_model=4096,
        mlp_type=MLPType.SWIGLU,
        supports_rome_memit=False,  # Not covered per R4
        supports_chinese_bench=False,
        supports_icl_study=False,
        vram_fp16_gb=22.0,
        vram_int8_gb=14.0,
    ),
    # Llama-2-7B for cross-architecture Late Crystallization validation
    "meta-llama/Llama-2-7b-hf": ModelMetadata(
        hf_name="meta-llama/Llama-2-7b-hf",
        display_name="Llama 2-7B",
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,  # MHA: n_kv_heads == n_heads (no GQA)
        d_model=4096,
        mlp_type=MLPType.SWIGLU,
        supports_rome_memit=False,  # Not covered per R4
        supports_chinese_bench=False,
        supports_icl_study=False,
        vram_fp16_gb=18.0,
        vram_int8_gb=11.0,
    ),
    # Mistral-7B for sliding window attention architecture diversity
    "mistralai/Mistral-7B-v0.1": ModelMetadata(
        hf_name="mistralai/Mistral-7B-v0.1",
        display_name="Mistral 7B",
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,  # GQA with sliding window attention
        d_model=4096,
        mlp_type=MLPType.SWIGLU,
        supports_rome_memit=False,
        supports_chinese_bench=False,
        supports_icl_study=False,
        vram_fp16_gb=18.0,
        vram_int8_gb=11.0,
    ),
    "EleutherAI/pythia-1.4b": ModelMetadata(
        hf_name="EleutherAI/pythia-1.4b",
        display_name="Pythia-1.4B",
        n_layers=24,
        n_heads=16,
        n_kv_heads=16,  # MHA: n_kv_heads == n_heads
        d_model=2048,
        mlp_type=MLPType.GELU,
        supports_rome_memit=True,  # Native GELU, original ROME architecture
        supports_chinese_bench=False,
        supports_icl_study=False,
        vram_fp16_gb=5.0,
    ),
}

# Model name aliases for convenience
MODEL_ALIASES: dict[str, str] = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen-7b": "Qwen/Qwen2.5-7B",
    "qwen-14b": "Qwen/Qwen2.5-14B",
    "llama-8b": "meta-llama/Llama-3.1-8B",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "pythia": "EleutherAI/pythia-1.4b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}


@dataclass
class InterventionConfig:
    """Default configuration for intervention operations."""

    default_ablation_value: float = 0.0
    default_scaling_factor: float = 1.0
    max_batch_size: int = 32
    save_activation_diff: bool = True


@dataclass
class VisualizationConfig:
    """Default configuration for visualizations."""

    default_colorscale: str = "RdBu"
    figure_width: int = 1000
    figure_height: int = 600
    export_dpi: int = 300
    export_format: str = "pdf"


@dataclass
class AnalysisConfig:
    """Default configuration for analysis operations."""

    include_logit_lens: bool = False
    causal_trace_noise_level: float = 0.1
    circuit_discovery_threshold: float = 0.1
    sae_top_k_features: int = 20


# Global default configurations
DEFAULT_INTERVENTION_CONFIG = InterventionConfig()
DEFAULT_VISUALIZATION_CONFIG = VisualizationConfig()
DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()

# Default dtype and device
DEFAULT_DTYPE = "float16"

def _detect_device() -> str:
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEFAULT_DEVICE = _detect_device()


def get_model_metadata(model_name: str) -> ModelMetadata:
    """Get metadata for a model by name or alias.

    Args:
        model_name: Model name or alias

    Returns:
        ModelMetadata for the model

    Raises:
        ValueError: If model not found
    """
    # Check aliases first
    resolved_name = MODEL_ALIASES.get(model_name, model_name)

    if resolved_name not in SUPPORTED_MODELS:
        supported = list(SUPPORTED_MODELS.keys()) + list(MODEL_ALIASES.keys())
        raise ValueError(
            f"Model '{model_name}' not supported. Supported models: {supported}"
        )

    return SUPPORTED_MODELS[resolved_name]


def get_model_config(
    model_name: str,
    dtype: str = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> ModelConfig:
    """Create a ModelConfig from model name.

    Args:
        model_name: Model name or alias
        dtype: Data type for inference
        device: Device for inference

    Returns:
        ModelConfig instance
    """
    metadata = get_model_metadata(model_name)

    return ModelConfig(
        model_name=metadata.hf_name,
        dtype=dtype,
        device=device,
        n_layers=metadata.n_layers,
        n_heads=metadata.n_heads,
        n_kv_heads=metadata.n_kv_heads,
        d_model=metadata.d_model,
        mlp_type=metadata.mlp_type,
        supports_rome_memit=metadata.supports_rome_memit,
        supports_chinese_bench=metadata.supports_chinese_bench,
        supports_icl_study=metadata.supports_icl_study,
    )


def list_supported_models() -> list[dict[str, Any]]:
    """List all supported models with their capabilities.

    Returns:
        List of model info dicts
    """
    result = []
    for name, metadata in SUPPORTED_MODELS.items():
        result.append({
            "name": name,
            "display_name": metadata.display_name,
            "n_layers": metadata.n_layers,
            "n_heads": metadata.n_heads,
            "mlp_type": metadata.mlp_type.value,
            "supports_rome_memit": metadata.supports_rome_memit,
            "supports_chinese_bench": metadata.supports_chinese_bench,
            "supports_icl_study": metadata.supports_icl_study,
            "vram_fp16_gb": metadata.vram_fp16_gb,
        })
    return result
