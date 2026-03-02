"""MechLens model loader.

Load models as TransformerLens HookedTransformer with support for all 4 model families.
Per contract section 1 and R1 design decisions.
"""

import logging
import os
from typing import Any

import torch
from transformer_lens import HookedTransformer

# Use HuggingFace mirror for China mainland access
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from mechlens.config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    SUPPORTED_MODELS,
    get_model_metadata,
)
from mechlens.types import ModelConfig, ModelLoadError

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    dtype: str = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    use_hf_model: bool = False,
    trust_remote_code: bool = True,
) -> HookedTransformer:
    """Load a model as TransformerLens HookedTransformer.

    Supports all 4 model families:
    - Qwen2.5-0.5B, Qwen2.5-7B (SwiGLU MLP, GQA)
    - Llama 3.1-8B (SwiGLU MLP, GQA)
    - Pythia-1.4B (GELU MLP, MHA)

    Args:
        model_name: Model name from SUPPORTED_MODELS or alias
        dtype: Data type for inference (float16, bfloat16, int8)
        device: Device for inference (cuda, cpu)
        use_hf_model: Force use of HuggingFace model fallback
        trust_remote_code: Trust remote code for custom models (Qwen)

    Returns:
        HookedTransformer instance

    Raises:
        ModelLoadError: If model cannot be loaded
    """
    try:
        # Resolve model name and get metadata
        metadata = get_model_metadata(model_name)
        hf_name = metadata.hf_name

        logger.info(f"Loading model: {hf_name} (dtype={dtype}, device={device})")

        # Determine torch dtype
        torch_dtype = _get_torch_dtype(dtype)

        # Check VRAM requirements
        _check_vram_requirements(metadata, dtype, device)

        # Load model via TransformerLens
        if use_hf_model or _needs_hf_fallback(hf_name):
            model = _load_with_hf_fallback(
                hf_name,
                torch_dtype=torch_dtype,
                device=device,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = _load_direct(
                hf_name,
                torch_dtype=torch_dtype,
                device=device,
            )

        logger.info(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
        return model

    except Exception as e:
        raise ModelLoadError(f"Failed to load model '{model_name}': {e}") from e


def _get_torch_dtype(dtype: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if dtype in dtype_map:
        return dtype_map[dtype]
    elif dtype == "int8":
        # int8 quantization handled separately
        return torch.float16  # Load as fp16, then quantize
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _check_vram_requirements(metadata: Any, dtype: str, device: str) -> None:
    """Check if device has sufficient VRAM."""
    if device in ("cpu", "mps"):
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return

    # Get available VRAM
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Get required VRAM
    if dtype == "int8" and metadata.vram_int8_gb:
        required_gb = metadata.vram_int8_gb
    else:
        required_gb = metadata.vram_fp16_gb

    # Add buffer for activation cache (~20%)
    required_gb *= 1.2

    if gpu_memory_gb < required_gb:
        logger.warning(
            f"GPU has {gpu_memory_gb:.1f}GB VRAM, model requires ~{required_gb:.1f}GB. "
            "May encounter OOM errors."
        )


def _needs_hf_fallback(model_name: str) -> bool:
    """Check if model needs HuggingFace fallback.

    Some newer model checkpoints may not have native TransformerLens support
    and need to be loaded via the hf_model parameter.
    """
    # Qwen2.5-7B and larger Qwen models need HF fallback
    # TransformerLens has native support for smaller Qwen models but
    # may not support all sizes natively
    if "Qwen" in model_name:
        # Check if model size > 0.5B — these often need fallback
        for size_str in ["7B", "14B", "32B", "72B"]:
            if size_str in model_name:
                return True

    return False


def _load_direct(
    hf_name: str,
    torch_dtype: torch.dtype,
    device: str,
) -> HookedTransformer:
    """Load model directly via TransformerLens, with automatic HF fallback."""
    try:
        model = HookedTransformer.from_pretrained(
            hf_name,
            torch_dtype=torch_dtype,
            device=device,
        )
        # Validate that hook points are available
        _validate_hooks(model)
        return model
    except Exception as e:
        logger.warning(
            f"Native TransformerLens loading failed for {hf_name}: {e}. "
            "Falling back to HuggingFace model wrapper."
        )
        return _load_with_hf_fallback(
            hf_name,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=True,
        )


def _validate_hooks(model: HookedTransformer) -> None:
    """Validate that standard hook points exist on the model.

    Raises RuntimeError if critical hook points are missing.
    """
    test_hooks = [
        "blocks.0.hook_resid_post",
        "blocks.0.hook_mlp_out",
        "blocks.0.attn.hook_result",
    ]
    available = set(model.hook_dict.keys()) if hasattr(model, "hook_dict") else set()

    if not available:
        raise RuntimeError("Model has no hook_dict — TransformerLens wrapping may have failed")

    missing = [h for h in test_hooks if h not in available]
    if missing:
        raise RuntimeError(
            f"Critical hook points missing: {missing}. "
            f"Available hooks (first 10): {sorted(available)[:10]}"
        )


def _load_with_hf_fallback(
    hf_name: str,
    torch_dtype: torch.dtype,
    device: str,
    trust_remote_code: bool = True,
) -> HookedTransformer:
    """Load model via HuggingFace fallback.

    Per R1: Use `hf_model` parameter for models with incomplete
    native TransformerLens support.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Using HuggingFace fallback for {hf_name}")

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        trust_remote_code=trust_remote_code,
    )

    # Wrap with TransformerLens
    model = HookedTransformer.from_pretrained(
        hf_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device=device,
    )

    return model


def get_model_config(model: HookedTransformer) -> ModelConfig:
    """Extract ModelConfig from loaded model.

    Args:
        model: Loaded HookedTransformer

    Returns:
        ModelConfig with model parameters
    """
    cfg = model.cfg

    # Determine MLP type from model config
    from mechlens.types import MLPType

    # Check if model uses SwiGLU (gate projection exists)
    if hasattr(cfg, "d_mlp") and cfg.d_mlp and hasattr(cfg, "act_fn"):
        if cfg.act_fn in ["silu", "swiglu", "gelu_new"]:
            mlp_type = MLPType.SWIGLU
        else:
            mlp_type = MLPType.GELU
    else:
        mlp_type = MLPType.GELU

    # Get model capabilities from registry
    model_name = cfg.model_name if hasattr(cfg, "model_name") else str(cfg.tokenizer_name)
    try:
        metadata = get_model_metadata(model_name)
        supports_rome_memit = metadata.supports_rome_memit
        supports_chinese_bench = metadata.supports_chinese_bench
        supports_icl_study = metadata.supports_icl_study
    except ValueError:
        # Model not in registry, use defaults
        supports_rome_memit = False
        supports_chinese_bench = False
        supports_icl_study = False

    return ModelConfig(
        model_name=model_name,
        dtype=str(cfg.dtype),
        device=str(cfg.device),
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=getattr(cfg, "n_key_value_heads", cfg.n_heads),
        d_model=cfg.d_model,
        mlp_type=mlp_type,
        supports_rome_memit=supports_rome_memit,
        supports_chinese_bench=supports_chinese_bench,
        supports_icl_study=supports_icl_study,
    )


def list_available_models() -> list[str]:
    """List all available model names."""
    return list(SUPPORTED_MODELS.keys())
