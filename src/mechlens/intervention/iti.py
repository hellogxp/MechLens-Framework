"""Inference-Time Intervention (ITI) for truthfulness steering.

Implements the ITI approach (Li et al., 2023): learn truthfulness directions
from contrastive activation pairs, then steer activations along those
directions during generation.

This provides a positive baseline that contrasts with the negative results
from fixed activation scaling: directional intervention succeeds because
it targets task-relevant subspaces, while magnitude-only scaling lacks
this directional specificity.
"""

import logging
from dataclasses import dataclass, field

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

logger = logging.getLogger(__name__)


@dataclass
class ITIDirections:
    """Learned ITI truthfulness directions per layer."""

    directions: dict[int, torch.Tensor]  # layer -> unit direction vector [d_model]
    projection_magnitudes: dict[int, float]  # layer -> mean projection magnitude
    n_correct: int = 0
    n_incorrect: int = 0
    layers: list[int] = field(default_factory=list)


def learn_iti_directions(
    model: HookedTransformer,
    correct_prompts: list[str],
    incorrect_prompts: list[str],
    layers: list[int] | None = None,
) -> ITIDirections:
    """Learn truthfulness directions from contrastive prompt pairs.

    For each layer, computes the mean activation difference between
    correct and incorrect prompts at the last token position, then
    normalizes to a unit direction vector.

    Args:
        model: HookedTransformer model
        correct_prompts: List of prompts with correct/truthful completions
        incorrect_prompts: List of prompts with incorrect/hallucinated completions
        layers: Layers to learn directions for (None = all)

    Returns:
        ITIDirections with per-layer unit direction vectors
    """
    n_layers = model.cfg.n_layers
    if layers is None:
        layers = list(range(n_layers))

    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

    logger.info(
        f"Learning ITI directions: {len(correct_prompts)} correct, "
        f"{len(incorrect_prompts)} incorrect, {len(layers)} layers"
    )

    # Collect last-token activations for correct prompts
    correct_acts = _collect_last_token_activations(model, correct_prompts, hook_names)

    # Collect last-token activations for incorrect prompts
    incorrect_acts = _collect_last_token_activations(model, incorrect_prompts, hook_names)

    # Compute per-layer directions: mean(correct) - mean(incorrect)
    directions = {}
    projection_magnitudes = {}

    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Stack and average: [n_samples, d_model] -> [d_model]
        correct_mean = torch.stack(correct_acts[hook_name]).mean(dim=0)
        incorrect_mean = torch.stack(incorrect_acts[hook_name]).mean(dim=0)

        # Direction: correct - incorrect (pointing toward truthfulness)
        diff = correct_mean - incorrect_mean
        magnitude = diff.norm(p=2).item()

        if magnitude > 1e-8:
            # Normalize to unit vector
            directions[layer] = (diff / diff.norm(p=2)).cpu()
        else:
            # Degenerate case: no separability at this layer
            directions[layer] = torch.zeros_like(diff).cpu()
            logger.warning(f"Layer {layer}: near-zero direction (magnitude={magnitude:.6f})")

        projection_magnitudes[layer] = magnitude

    # Log top layers by magnitude
    sorted_layers = sorted(layers, key=lambda l: projection_magnitudes[l], reverse=True)
    top_5 = sorted_layers[:5]
    logger.info(
        f"Top layers by direction magnitude: "
        + ", ".join(f"L{l}={projection_magnitudes[l]:.4f}" for l in top_5)
    )

    return ITIDirections(
        directions=directions,
        projection_magnitudes=projection_magnitudes,
        n_correct=len(correct_prompts),
        n_incorrect=len(incorrect_prompts),
        layers=layers,
    )


def create_iti_steering_hook(
    direction: torch.Tensor,
    coefficient: float,
) -> callable:
    """Create a hook that adds a scaled direction to the residual stream.

    The hook adds `coefficient * direction` to the last token position
    of the residual stream activation during forward pass.

    Args:
        direction: Unit direction vector [d_model]
        coefficient: Steering strength (higher = more intervention)

    Returns:
        Hook function compatible with TransformerLens
    """
    # Pre-compute the steering vector
    steering_vector = (coefficient * direction).clone()

    def hook_fn(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # activation shape: [batch, seq_len, d_model]
        modified = activation.clone()

        # Move steering vector to same device/dtype as activation
        sv = steering_vector.to(device=activation.device, dtype=activation.dtype)

        # Add steering to all token positions (follows ITI paper approach)
        modified = modified + sv.unsqueeze(0).unsqueeze(0)

        return modified

    return hook_fn


def generate_with_iti(
    model: HookedTransformer,
    input_text: str,
    iti_directions: ITIDirections,
    coefficient: float = 1.0,
    layers: list[int] | None = None,
    max_new_tokens: int = 100,
) -> tuple[str, str]:
    """Generate text with ITI steering applied.

    Args:
        model: HookedTransformer model
        input_text: Input prompt
        iti_directions: Learned ITI directions
        coefficient: Steering coefficient (higher = stronger intervention)
        layers: Which layers to steer (None = use all learned layers)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (original_output, steered_output)
    """
    if layers is None:
        layers = iti_directions.layers

    tokens = model.to_tokens(input_text)

    # Generate original output
    with torch.no_grad():
        orig_ids = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    original_output = model.to_string(orig_ids[0, tokens.shape[1]:]).strip()

    # Build steering hooks
    hooks = []
    for layer in layers:
        if layer not in iti_directions.directions:
            continue
        direction = iti_directions.directions[layer]
        hook_point = f"blocks.{layer}.hook_resid_post"
        hook_fn = create_iti_steering_hook(direction, coefficient)
        hooks.append((hook_point, hook_fn))

    if not hooks:
        logger.warning("No valid steering hooks created, returning original output")
        return original_output, original_output

    # Generate with steering hooks (token-by-token for autoregressive generation)
    generated = tokens.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(generated, fwd_hooks=hooks)
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)

            if model.tokenizer.eos_token_id is not None:
                if next_token.item() == model.tokenizer.eos_token_id:
                    break

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    steered_output = model.to_string(generated[0, tokens.shape[1]:]).strip()

    return original_output, steered_output


def select_top_layers(
    iti_directions: ITIDirections,
    top_k: int = 5,
) -> list[int]:
    """Select the top-K layers by direction magnitude for steering.

    Layers with larger projection magnitudes have more separability
    between truthful and untruthful activations, making them better
    candidates for intervention.

    Args:
        iti_directions: Learned directions
        top_k: Number of top layers to select

    Returns:
        List of layer indices sorted by magnitude (descending)
    """
    sorted_layers = sorted(
        iti_directions.layers,
        key=lambda l: iti_directions.projection_magnitudes.get(l, 0),
        reverse=True,
    )
    return sorted_layers[:top_k]


def compute_iti_metrics(
    model: HookedTransformer,
    input_text: str,
    iti_directions: ITIDirections,
    coefficient: float,
    layers: list[int] | None = None,
) -> dict:
    """Compute metrics for ITI intervention (KL divergence, prob change).

    Args:
        model: HookedTransformer model
        input_text: Input text
        iti_directions: Learned directions
        coefficient: Steering coefficient
        layers: Layers to steer (None = all learned)

    Returns:
        Dict with kl_divergence, prob_change, logit_diff
    """
    import torch.nn.functional as F

    if layers is None:
        layers = iti_directions.layers

    tokens = model.to_tokens(input_text)

    # Build hooks
    hooks = []
    for layer in layers:
        if layer not in iti_directions.directions:
            continue
        direction = iti_directions.directions[layer]
        hook_point = f"blocks.{layer}.hook_resid_post"
        hook_fn = create_iti_steering_hook(direction, coefficient)
        hooks.append((hook_point, hook_fn))

    with torch.no_grad():
        original_logits = model(tokens)
        steered_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    # Cast to float32 for numerical stability
    orig_probs = torch.softmax(original_logits[0, -1, :].float(), dim=-1)
    steer_probs = torch.softmax(steered_logits[0, -1, :].float(), dim=-1)

    kl_div = F.kl_div(
        steer_probs.log(), orig_probs, reduction="sum", log_target=False
    ).item()
    kl_div = max(0.0, kl_div)

    original_top = original_logits[0, -1, :].argmax()
    logit_diff = (
        original_logits[0, -1, original_top].float()
        - steered_logits[0, -1, original_top].float()
    ).item()

    prob_change = (orig_probs[original_top] - steer_probs[original_top]).item()

    return {
        "kl_divergence": kl_div,
        "logit_diff": logit_diff,
        "prob_change": prob_change,
        "coefficient": coefficient,
        "n_steered_layers": len(hooks),
    }


def serialize_directions(iti_directions: ITIDirections) -> dict:
    """Serialize ITI directions to JSON-compatible dict.

    Args:
        iti_directions: Learned directions

    Returns:
        JSON-serializable dict
    """
    return {
        "directions": {
            str(k): v.tolist() for k, v in iti_directions.directions.items()
        },
        "projection_magnitudes": {
            str(k): v for k, v in iti_directions.projection_magnitudes.items()
        },
        "n_correct": iti_directions.n_correct,
        "n_incorrect": iti_directions.n_incorrect,
        "layers": iti_directions.layers,
    }


def _collect_last_token_activations(
    model: HookedTransformer,
    prompts: list[str],
    hook_names: list[str],
) -> dict[str, list[torch.Tensor]]:
    """Collect last-token residual stream activations for a list of prompts.

    Args:
        model: HookedTransformer model
        prompts: List of text prompts
        hook_names: List of hook names to cache

    Returns:
        Dict mapping hook_name -> list of [d_model] tensors (one per prompt)
    """
    all_acts = {name: [] for name in hook_names}

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)

        for name in hook_names:
            # Take last token position: [1, seq_len, d_model] -> [d_model]
            last_token_act = cache[name][0, -1, :].detach().cpu()
            all_acts[name].append(last_token_act)

    return all_acts
