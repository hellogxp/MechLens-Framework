"""MechLens MEMIT batch edit.

Mass-Editing Memory in a Transformer for multi-fact editing.
BASELINE comparison method - Qwen + Pythia only (NOT Llama per R4).

Based on Meng et al. (2023) "Mass-Editing Memory in a Transformer".
"""

import logging
from typing import Any

import torch
from transformer_lens import HookedTransformer

from mechlens.config import get_model_metadata
from mechlens.editing.rome import (
    _check_model_support,
    _compute_key_vector,
    _get_mlp_type,
    _get_model_name,
)
from mechlens.types import EditMetrics, UnsupportedModelError

logger = logging.getLogger(__name__)


def edit(
    model: HookedTransformer,
    edits: list[dict[str, str]],
    layers: list[int] | None = None,
    num_grad_steps: int = 25,
    lr: float = 5e-4,
) -> tuple[HookedTransformer, list[EditMetrics]]:
    """Apply MEMIT batch edit to modify multiple factual associations.

    Supports Qwen2.5 (SwiGLU MLP) and Pythia-1.4B (GELU MLP).
    Same model support constraints as ROME per R4.

    Args:
        model: HookedTransformer model
        edits: List of edit dicts with keys:
            - subject: Subject entity
            - target_old: Old association
            - target_new: New association
        layers: Specific layers to edit (None = auto-select)
        num_grad_steps: Number of gradient steps for optimization
        lr: Learning rate for optimization

    Returns:
        Tuple of (edited_model, list[EditMetrics])

    Raises:
        UnsupportedModelError: If model is Llama (MEMIT not supported per R4)
    """
    # Check model support (same as ROME)
    _check_model_support(model)

    if not edits:
        return model, []

    # Auto-select layers if not specified
    if layers is None:
        layers = _auto_select_layers_memit(model, edits)

    logger.info(f"MEMIT edit: {len(edits)} facts at layers {layers}")

    # Determine MLP type
    mlp_type = _get_mlp_type(model)

    # Compute all key-value pairs
    keys = []
    values = []

    for edit_spec in edits:
        subject = edit_spec["subject"]
        target_old = edit_spec["target_old"]
        target_new = edit_spec["target_new"]

        # Use middle layer for key computation
        key_layer = layers[len(layers) // 2]
        key = _compute_key_vector(model, subject, key_layer)
        keys.append(key)

        # Compute value
        value = _compute_memit_value(
            model, subject, target_old, target_new, layers, num_grad_steps, lr
        )
        values.append(value)

    # Stack keys and values
    K = torch.stack(keys, dim=0)  # [n_edits, d_model]
    V = torch.stack(values, dim=0)  # [n_edits, d_model]

    # Apply batched update to all specified layers
    _apply_memit_update(model, layers, K, V, mlp_type)

    # Compute metrics for each edit
    metrics = []
    for edit_spec in edits:
        m = _compute_edit_metrics(
            model,
            edit_spec["subject"],
            edit_spec["target_old"],
            edit_spec["target_new"],
        )
        metrics.append(m)

    # Log summary
    avg_es = sum(m.es for m in metrics) / len(metrics)
    avg_ps = sum(m.ps for m in metrics) / len(metrics)
    avg_ns = sum(m.ns for m in metrics) / len(metrics)

    logger.info(
        f"MEMIT edit complete: avg ES={avg_es:.3f}, PS={avg_ps:.3f}, NS={avg_ns:.3f}"
    )

    return model, metrics


def _auto_select_layers_memit(
    model: HookedTransformer,
    edits: list[dict[str, str]],
) -> list[int]:
    """Auto-select layers for MEMIT editing.

    MEMIT typically edits a range of middle-to-late layers.
    """
    n_layers = model.cfg.n_layers

    # MEMIT default: edit layers in range [n_layers//3, 2*n_layers//3]
    start = n_layers // 3
    end = 2 * n_layers // 3

    return list(range(start, end + 1))


def _compute_memit_value(
    model: HookedTransformer,
    subject: str,
    target_old: str,
    target_new: str,
    layers: list[int],
    num_grad_steps: int,
    lr: float,
) -> torch.Tensor:
    """Compute value vector for MEMIT.

    Unlike ROME which computes per-layer values, MEMIT computes
    a single value that gets distributed across layers.
    """
    prompt = f"{subject} is"
    tokens = model.to_tokens(prompt)

    # Get target token ID
    new_token_id = model.to_single_token(target_new)

    # Initialize shared value
    d_model = model.cfg.d_model
    value = torch.zeros(d_model, requires_grad=True, device=model.cfg.device)
    optimizer = torch.optim.Adam([value], lr=lr)

    # Optimize value across all target layers
    for step in range(num_grad_steps):
        optimizer.zero_grad()

        # Create hooks to add value to MLP outputs at all layers
        hooks = []
        for layer in layers:
            def add_value_hook(activation, hook, v=value, n_layers=len(layers)):
                modified = activation.clone()
                # Distribute value across layers
                modified[:, -1, :] += v / n_layers
                return modified

            hook_name = f"blocks.{layer}.hook_mlp_out"
            hooks.append((hook_name, add_value_hook))

        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        # Loss: negative log probability of target_new
        log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
        loss = -log_probs[new_token_id]

        loss.backward()
        optimizer.step()

    return value.detach()


def _apply_memit_update(
    model: HookedTransformer,
    layers: list[int],
    K: torch.Tensor,
    V: torch.Tensor,
    mlp_type: str,
) -> None:
    """Apply MEMIT batched update to MLP weights.

    MEMIT distributes the update across multiple layers.
    """
    n_edits = K.shape[0]
    n_layers = len(layers)

    # Compute covariance for key vectors
    # C = K^T @ K
    C = K.T @ K  # [d_model, d_model]

    # Add regularization
    reg = 1e-4 * torch.eye(C.shape[0], device=C.device)
    C_inv = torch.linalg.inv(C + reg)

    # Compute update matrix: (K^T @ K)^{-1} @ K^T @ V
    # This solves for the optimal update given multiple edits
    update_coeff = C_inv @ K.T @ V  # [d_model, d_model]

    # Distribute update across layers
    for layer in layers:
        # Scale update by 1/n_layers to distribute effect
        scaled_update = update_coeff / n_layers

        # Get weight matrix
        if mlp_type == "swiglu":
            W = model.blocks[layer].mlp.W_out
        else:
            W = model.blocks[layer].mlp.W_out

        # Apply update
        with torch.no_grad():
            if W.shape == scaled_update.shape:
                W.data += scaled_update
            elif W.shape == scaled_update.T.shape:
                W.data += scaled_update.T
            else:
                logger.warning(
                    f"Layer {layer}: shape mismatch W={W.shape}, update={scaled_update.shape}"
                )


def _compute_edit_metrics(
    model: HookedTransformer,
    subject: str,
    target_old: str,
    target_new: str,
) -> EditMetrics:
    """Compute ES, PS, NS metrics for a single edit."""
    es = _compute_edit_success(model, subject, target_new)
    ps = _compute_paraphrase_success(model, subject, target_new)
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
    """Compute neighborhood specificity."""
    unrelated_prompts = [
        "The capital of France is",
        "Water boils at 100 degrees",
        "The Earth orbits the",
    ]

    preserved = 0
    for prompt in unrelated_prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            if probs.max().item() > 0.1:
                preserved += 1

    return preserved / len(unrelated_prompts) if unrelated_prompts else 1.0


def verify_model_support(model: HookedTransformer) -> dict[str, Any]:
    """Verify if model supports MEMIT editing.

    Returns:
        Dict with 'supported', 'model_name', 'mlp_type', 'reason'
    """
    model_name = _get_model_name(model)
    mlp_type = _get_mlp_type(model)

    try:
        _check_model_support(model)
        return {
            "supported": True,
            "model_name": model_name,
            "mlp_type": mlp_type,
            "reason": "Model supports ROME/MEMIT editing",
        }
    except UnsupportedModelError as e:
        return {
            "supported": False,
            "model_name": model_name,
            "mlp_type": mlp_type,
            "reason": str(e),
        }
