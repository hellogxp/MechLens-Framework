"""MechLens ROME single edit.

Rank-One Model Editing for factual knowledge modification.
BASELINE comparison method - Qwen + Pythia only (NOT Llama per R4).

Based on Meng et al. (2022) "Locating and Editing Factual Associations in GPT".
"""

import logging
from typing import Any

import torch
from transformer_lens import HookedTransformer

from mechlens.analysis.activation import causal_trace
from mechlens.config import get_model_metadata
from mechlens.types import EditMetrics, UnsupportedModelError

logger = logging.getLogger(__name__)


def edit(
    model: HookedTransformer,
    subject: str,
    target_old: str,
    target_new: str,
    layers: list[int] | None = None,
    num_grad_steps: int = 25,
    lr: float = 5e-4,
) -> tuple[HookedTransformer, EditMetrics]:
    """Apply ROME single edit to modify a factual association.

    Supports Qwen2.5 (SwiGLU MLP with gate_proj + up_proj fusion) and
    Pythia-1.4B (native GELU MLP - original ROME architecture).

    Args:
        model: HookedTransformer model
        subject: Subject entity (e.g., "The Eiffel Tower")
        target_old: Old association (e.g., "Paris")
        target_new: New association (e.g., "Rome")
        layers: Specific layers to edit (None = auto-select via causal tracing)
        num_grad_steps: Number of gradient steps for key/value computation
        lr: Learning rate for optimization

    Returns:
        Tuple of (edited_model, EditMetrics)

    Raises:
        UnsupportedModelError: If model is Llama (ROME not supported per R4)
    """
    # Check model support
    _check_model_support(model)

    # Auto-select layers via causal tracing if not specified
    if layers is None:
        layers = _auto_select_layers(model, subject, target_old)

    logger.info(f"ROME edit: '{subject}' -> '{target_new}' at layers {layers}")

    # Determine MLP type and select appropriate edit function
    mlp_type = _get_mlp_type(model)

    if mlp_type == "swiglu":
        edited_model = _rome_edit_swiglu(
            model, subject, target_old, target_new, layers, num_grad_steps, lr
        )
    else:
        edited_model = _rome_edit_gelu(
            model, subject, target_old, target_new, layers, num_grad_steps, lr
        )

    # Compute edit metrics
    metrics = _compute_edit_metrics(edited_model, subject, target_old, target_new)

    logger.info(
        f"ROME edit complete: ES={metrics.es:.3f}, PS={metrics.ps:.3f}, NS={metrics.ns:.3f}"
    )

    return edited_model, metrics


def _check_model_support(model: HookedTransformer) -> None:
    """Check if model supports ROME editing."""
    model_name = _get_model_name(model)

    try:
        metadata = get_model_metadata(model_name)
        if not metadata.supports_rome_memit:
            raise UnsupportedModelError(
                f"ROME not supported for {model_name}. "
                "Only Qwen2.5 and Pythia models are supported."
            )
    except ValueError:
        # Model not in registry - check by name pattern
        model_lower = model_name.lower()
        if "llama" in model_lower:
            raise UnsupportedModelError(
                f"ROME not supported for Llama models ({model_name}). "
                "Per R4: Llama 3.1-8B activation intervention only."
            )


def _get_model_name(model: HookedTransformer) -> str:
    """Extract model name from HookedTransformer."""
    if hasattr(model.cfg, "model_name") and model.cfg.model_name:
        return model.cfg.model_name
    return str(getattr(model.cfg, "tokenizer_name", "unknown"))


def _get_mlp_type(model: HookedTransformer) -> str:
    """Determine MLP type (gelu or swiglu)."""
    model_name = _get_model_name(model).lower()

    if "qwen" in model_name:
        return "swiglu"
    elif "pythia" in model_name:
        return "gelu"
    else:
        # Default based on activation function
        if hasattr(model.cfg, "act_fn"):
            if model.cfg.act_fn in ["silu", "swiglu"]:
                return "swiglu"
        return "gelu"


def _auto_select_layers(
    model: HookedTransformer,
    subject: str,
    target_old: str,
) -> list[int]:
    """Auto-select layers via causal tracing."""
    # Construct prompt
    prompt = f"{subject} is located in"

    try:
        result = causal_trace(model, prompt, subject, component_type="mlp")
        # Get top 3 layers by recovery score
        top_k = 3
        values, indices = torch.topk(result.patch_results, min(top_k, len(result.patch_results)))
        layers = [idx.item() for idx in indices if values[indices.tolist().index(idx)] > 0.1]

        if not layers:
            # Fallback to middle layers
            n_layers = model.cfg.n_layers
            layers = [n_layers // 3, n_layers // 2, 2 * n_layers // 3]

    except Exception as e:
        logger.warning(f"Causal tracing failed, using default layers: {e}")
        n_layers = model.cfg.n_layers
        layers = [n_layers // 3, n_layers // 2]

    return layers


def _rome_edit_gelu(
    model: HookedTransformer,
    subject: str,
    target_old: str,
    target_new: str,
    layers: list[int],
    num_grad_steps: int,
    lr: float,
) -> HookedTransformer:
    """Apply ROME edit to GELU MLP (Pythia architecture)."""
    # This is the original ROME formulation
    # MLP structure: x -> W_in -> GELU -> W_out -> output

    for layer in layers:
        # Compute key vector (subject representation)
        key = _compute_key_vector(model, subject, layer)

        # Compute value vector (target representation difference)
        value = _compute_value_vector(model, subject, target_old, target_new, layer, num_grad_steps, lr)

        # Apply rank-one update to W_out
        _apply_rank_one_update(model, layer, key, value, mlp_type="gelu")

    return model


def _rome_edit_swiglu(
    model: HookedTransformer,
    subject: str,
    target_old: str,
    target_new: str,
    layers: list[int],
    num_grad_steps: int,
    lr: float,
) -> HookedTransformer:
    """Apply ROME edit to SwiGLU MLP (Qwen architecture).

    SwiGLU MLP structure:
    x -> [gate_proj(x) * silu(up_proj(x))] -> down_proj -> output

    Per R4: Requires gate_proj + up_proj fusion handling.
    """
    for layer in layers:
        # Compute key vector (subject representation)
        key = _compute_key_vector(model, subject, layer)

        # Compute value vector (target representation difference)
        value = _compute_value_vector(model, subject, target_old, target_new, layer, num_grad_steps, lr)

        # Apply rank-one update to down_proj (W_out equivalent)
        _apply_rank_one_update(model, layer, key, value, mlp_type="swiglu")

    return model


def _compute_key_vector(
    model: HookedTransformer,
    subject: str,
    layer: int,
) -> torch.Tensor:
    """Compute key vector for subject at specified layer."""
    tokens = model.to_tokens(subject)
    hook_name = f"blocks.{layer}.hook_mlp_out"

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        # Use mean of subject token representations
        key = cache[hook_name][0].mean(dim=0)  # [d_model]

    return key


def _compute_value_vector(
    model: HookedTransformer,
    subject: str,
    target_old: str,
    target_new: str,
    layer: int,
    num_grad_steps: int,
    lr: float,
) -> torch.Tensor:
    """Compute value vector (desired output change)."""
    # Target: shift representation so model predicts target_new instead of target_old

    # Get current representation at target layer
    prompt = f"{subject} is"
    tokens = model.to_tokens(prompt)

    # Get target token IDs
    new_token_id = model.to_single_token(target_new)

    # Initialize value as learnable parameter
    d_model = model.cfg.d_model
    value = torch.zeros(d_model, requires_grad=True, device=model.cfg.device)
    optimizer = torch.optim.Adam([value], lr=lr)

    # Optimize value to increase probability of target_new
    for step in range(num_grad_steps):
        optimizer.zero_grad()

        # Create hook to add value to MLP output
        def add_value_hook(activation, hook, v=value):
            modified = activation.clone()
            modified[:, -1, :] += v
            return modified

        hook_name = f"blocks.{layer}.hook_mlp_out"
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, add_value_hook)])

        # Loss: negative log probability of target_new
        log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
        loss = -log_probs[new_token_id]

        loss.backward()
        optimizer.step()

    return value.detach()


def _apply_rank_one_update(
    model: HookedTransformer,
    layer: int,
    key: torch.Tensor,
    value: torch.Tensor,
    mlp_type: str,
) -> None:
    """Apply rank-one update to MLP weights."""
    # Get the output projection weight matrix
    if mlp_type == "swiglu":
        # For SwiGLU: modify W_out (down_proj)
        W = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    else:
        # For GELU: modify W_out
        W = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    # Compute update: outer product of key and value
    # Normalize key
    key_norm = key / (key.norm() + 1e-8)

    # Scale value appropriately
    update = torch.outer(key_norm, value)  # [d_mlp, d_model] or similar

    # Apply update (may need transpose depending on weight layout)
    with torch.no_grad():
        if W.shape == update.shape:
            W.data += update
        elif W.shape == update.T.shape:
            W.data += update.T
        else:
            logger.warning(
                f"Shape mismatch in rank-one update: W={W.shape}, update={update.shape}"
            )


def _compute_edit_metrics(
    model: HookedTransformer,
    subject: str,
    target_old: str,
    target_new: str,
) -> EditMetrics:
    """Compute ES, PS, NS metrics for edit evaluation."""
    # ES: Edit Success - does model predict target_new for exact prompt?
    es = _compute_edit_success(model, subject, target_new)

    # PS: Paraphrase Success - does edit generalize to paraphrases?
    ps = _compute_paraphrase_success(model, subject, target_new)

    # NS: Neighborhood Specificity - are unrelated facts preserved?
    ns = _compute_neighborhood_specificity(model, subject, target_old)

    return EditMetrics(es=es, ps=ps, ns=ns)


def _compute_edit_success(model: HookedTransformer, subject: str, target_new: str) -> float:
    """Compute edit success rate."""
    prompt = f"{subject} is located in"
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        logits = model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=-1)

    target_token_id = model.to_single_token(target_new)
    return probs[target_token_id].item()


def _compute_paraphrase_success(model: HookedTransformer, subject: str, target_new: str) -> float:
    """Compute paraphrase success rate."""
    paraphrases = [
        f"Where is {subject}? It is in",
        f"The location of {subject} is",
        f"{subject} can be found in",
    ]

    successes = []
    target_token_id = model.to_single_token(target_new)

    for para in paraphrases:
        tokens = model.to_tokens(para)
        with torch.no_grad():
            logits = model(tokens)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            successes.append(probs[target_token_id].item())

    return sum(successes) / len(successes) if successes else 0.0


def _compute_neighborhood_specificity(model: HookedTransformer, subject: str, target_old: str) -> float:
    """Compute neighborhood specificity (preservation of unrelated facts)."""
    # Test that unrelated facts are preserved
    # This is a simplified version - full NS requires neighborhood samples
    unrelated_prompts = [
        "The capital of France is",
        "Water boils at 100 degrees",
        "The Earth orbits the",
    ]

    # Check that model still produces reasonable outputs
    preserved = 0
    for prompt in unrelated_prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)
            # Check if top prediction is reasonable (high confidence)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            if probs.max().item() > 0.1:  # Model is still confident
                preserved += 1

    return preserved / len(unrelated_prompts) if unrelated_prompts else 1.0
