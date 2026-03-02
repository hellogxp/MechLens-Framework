"""MechLens logit lens analysis.

Layer-wise vocabulary projections from residual stream through unembedding.
Per contract specification and "Interpreting Language Models with Contrastive Explanations".
"""

import logging

import torch
from transformer_lens import HookedTransformer

from mechlens.types import ActivationData

logger = logging.getLogger(__name__)


def compute_logit_lens(
    model: HookedTransformer,
    input_text: str,
) -> torch.Tensor:
    """Compute logit lens projections for all layers.

    Projects residual stream at each layer through the final layer norm
    and unembedding matrix to see what the model "thinks" at each layer.

    Args:
        model: HookedTransformer model
        input_text: Input text to analyze

    Returns:
        Logit lens tensor [layers, seq, vocab]
    """
    n_layers = model.cfg.n_layers

    # Cache residual stream at each layer
    hook_points = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]
    _, cache = model.run_with_cache(input_text, names_filter=hook_points)

    logits_list = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0]  # [seq, d_model]

        # Apply final layer norm
        normed = model.ln_final(resid)

        # Project to vocabulary
        logits = normed @ model.W_U
        if model.b_U is not None:
            logits = logits + model.b_U

        logits_list.append(logits)

    result = torch.stack(logits_list, dim=0)  # [layers, seq, vocab]

    logger.info(f"Computed logit lens: {result.shape}")
    return result


def get_top_predictions(
    model: HookedTransformer,
    logit_lens: torch.Tensor,
    position: int,
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    """Get top-k token predictions at each layer for a given position.

    Args:
        model: HookedTransformer model
        logit_lens: Logit lens tensor [layers, seq, vocab]
        position: Token position to analyze
        top_k: Number of top predictions per layer

    Returns:
        List of [(token, probability)] for each layer
    """
    n_layers = logit_lens.shape[0]
    results = []

    for layer in range(n_layers):
        logits = logit_lens[layer, position, :]  # [vocab]
        probs = torch.softmax(logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, top_k)
        layer_preds = [
            (model.to_single_str_token(idx.item()), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        results.append(layer_preds)

    return results


def track_prediction_evolution(
    model: HookedTransformer,
    logit_lens: torch.Tensor,
    position: int,
    target_token: str | None = None,
) -> dict[str, list[float]]:
    """Track how predictions evolve across layers.

    Args:
        model: HookedTransformer model
        logit_lens: Logit lens tensor [layers, seq, vocab]
        position: Token position to analyze
        target_token: Optional specific token to track (default: final prediction)

    Returns:
        Dict with 'layers', 'target_prob', 'entropy', 'top1_stable'
    """
    n_layers = logit_lens.shape[0]

    # Determine target token (top prediction at final layer if not specified)
    if target_token is None:
        final_logits = logit_lens[-1, position, :]
        target_idx = final_logits.argmax().item()
    else:
        target_idx = model.to_single_token(target_token)

    target_probs = []
    entropies = []
    top1_tokens = []

    for layer in range(n_layers):
        logits = logit_lens[layer, position, :]
        probs = torch.softmax(logits, dim=-1)

        # Track target probability
        target_probs.append(probs[target_idx].item())

        # Compute entropy
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        entropies.append(entropy)

        # Track top-1 token
        top1_tokens.append(probs.argmax().item())

    # Compute top-1 stability (how often top-1 matches final prediction)
    top1_stable = [1.0 if t == target_idx else 0.0 for t in top1_tokens]

    return {
        "layers": list(range(n_layers)),
        "target_prob": target_probs,
        "entropy": entropies,
        "top1_stable": top1_stable,
    }


def compute_tuned_lens(
    model: HookedTransformer,
    input_text: str,
    tuned_lens_weights: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute tuned lens projections (if weights available).

    Tuned lens uses learned affine transformations at each layer
    instead of directly applying the unembedding.

    Args:
        model: HookedTransformer model
        input_text: Input text to analyze
        tuned_lens_weights: Optional pre-trained tuned lens weights per layer

    Returns:
        Tuned lens tensor [layers, seq, vocab], or standard logit lens if no weights
    """
    if tuned_lens_weights is None:
        logger.info("No tuned lens weights provided, falling back to standard logit lens")
        return compute_logit_lens(model, input_text)

    n_layers = model.cfg.n_layers
    hook_points = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]
    _, cache = model.run_with_cache(input_text, names_filter=hook_points)

    logits_list = []
    for layer in range(n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0]  # [seq, d_model]

        # Apply tuned lens transformation
        if layer in tuned_lens_weights:
            transformed = resid @ tuned_lens_weights[layer]
        else:
            transformed = resid

        # Apply final layer norm and unembedding
        normed = model.ln_final(transformed)
        logits = normed @ model.W_U
        if model.b_U is not None:
            logits = logits + model.b_U

        logits_list.append(logits)

    return torch.stack(logits_list, dim=0)


def from_activation_data(
    model: HookedTransformer,
    activation_data: ActivationData,
) -> torch.Tensor:
    """Compute logit lens from pre-computed activation data.

    Args:
        model: HookedTransformer model
        activation_data: ActivationData containing residual_stream

    Returns:
        Logit lens tensor [layers, seq, vocab]
    """
    if activation_data.logit_lens is not None:
        return activation_data.logit_lens

    # Compute from residual stream
    residual_stream = activation_data.residual_stream
    n_layers = residual_stream.shape[0]

    logits_list = []
    for layer in range(n_layers):
        resid = residual_stream[layer]  # [seq, d_model]
        normed = model.ln_final(resid)
        logits = normed @ model.W_U
        if model.b_U is not None:
            logits = logits + model.b_U
        logits_list.append(logits)

    return torch.stack(logits_list, dim=0)
