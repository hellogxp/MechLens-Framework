"""MechLens injection intervention.

Replace specified components' activations with externally provided activations.
Per contract section 7 - PRIMARY intervention paradigm.
"""

import logging

import torch
from transformer_lens import HookedTransformer

from mechlens.analysis.activation import analyze as analyze_activation
from mechlens.models.hook_manager import create_injection_hook, get_component_hook_point
from mechlens.types import (
    ActivationData,
    InterventionError,
    InterventionResult,
    InterventionTarget,
    ShapeMismatchError,
)

logger = logging.getLogger(__name__)


def inject(
    model: HookedTransformer,
    input_text: str,
    targets: list[InterventionTarget],
    source_activations: dict[str, torch.Tensor],
    max_new_tokens: int = 50,
) -> InterventionResult:
    """Replace specified components' activations with provided tensors.

    Works uniformly across all 4 model families (Qwen, Llama, Pythia).

    Args:
        model: HookedTransformer model
        input_text: Input text
        targets: List of InterventionTarget specifying components to replace
        source_activations: Dict mapping "layer_{i}_{component}" to replacement tensor
        max_new_tokens: Maximum new tokens to generate

    Returns:
        InterventionResult with original and intervened outputs

    Raises:
        InterventionError: If target out of range
        ShapeMismatchError: If source tensor shape doesn't match target
    """
    _validate_targets(model, targets)

    # Get expected shapes for validation
    tokens = model.to_tokens(input_text)
    _validate_activations(model, tokens, targets, source_activations)

    # Get original output
    with torch.no_grad():
        original_output = _generate(model, tokens, max_new_tokens)

    # Build injection hooks
    hooks = []
    for target in targets:
        key = _get_activation_key(target)
        if key not in source_activations:
            raise InterventionError(f"Missing source activation for {key}")

        source = source_activations[key]
        target_with_source = InterventionTarget(
            layer=target.layer,
            component_type=target.component_type,
            component_id=target.component_id,
            source_activation=source,
        )

        hook_point = get_component_hook_point(target.layer, target.component_type)
        hook_fn = create_injection_hook(target_with_source, source)
        hooks.append((hook_point, hook_fn))

    # Get intervened output
    with torch.no_grad():
        intervened_output = _generate_with_hooks(model, tokens, hooks, max_new_tokens)

    # Compute activation difference
    activation_diff = _compute_activation_diff(model, input_text, hooks)

    # Compute metrics
    metrics = _compute_metrics(model, tokens, hooks)

    logger.info(
        f"Injection complete: {len(targets)} targets, "
        f"KL divergence = {metrics.get('kl_divergence', 'N/A'):.4f}"
    )

    return InterventionResult(
        original_output=original_output,
        intervened_output=intervened_output,
        activation_diff=activation_diff,
        metrics=metrics,
    )


def _get_activation_key(target: InterventionTarget) -> str:
    """Get dictionary key for target activation."""
    component = target.component_type.value
    if target.component_id is not None:
        return f"layer_{target.layer}_{component}_{target.component_id}"
    return f"layer_{target.layer}_{component}"


def _validate_targets(model: HookedTransformer, targets: list[InterventionTarget]) -> None:
    """Validate intervention targets."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    for target in targets:
        if target.layer < 0 or target.layer >= n_layers:
            raise InterventionError(
                f"Layer {target.layer} out of range [0, {n_layers})"
            )

        if target.component_type.value == "attn_head" and target.component_id is not None:
            if target.component_id < 0 or target.component_id >= n_heads:
                raise InterventionError(
                    f"Head {target.component_id} out of range [0, {n_heads})"
                )


def _validate_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    targets: list[InterventionTarget],
    source_activations: dict[str, torch.Tensor],
) -> None:
    """Validate source activation shapes match targets."""
    # Get actual activation shapes
    n_layers = model.cfg.n_layers
    hook_points = []
    for layer in range(n_layers):
        hook_points.extend([
            f"blocks.{layer}.hook_resid_post",
            f"blocks.{layer}.hook_mlp_out",
            f"blocks.{layer}.attn.hook_result",
        ])

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_points)

    for target in targets:
        key = _get_activation_key(target)
        if key not in source_activations:
            raise InterventionError(f"Missing source activation for {key}")

        source = source_activations[key]
        hook_point = get_component_hook_point(target.layer, target.component_type)

        if hook_point in cache:
            expected_shape = cache[hook_point][0].shape

            # For specific component_id, check sub-tensor shape
            if target.component_id is not None:
                if target.component_type.value == "attn_head":
                    # Expected shape is [seq, d_head] for a single head
                    d_head = model.cfg.d_head
                    expected_shape = (expected_shape[0], d_head)

            if source.shape != expected_shape:
                raise ShapeMismatchError(
                    f"Source activation shape {source.shape} doesn't match "
                    f"expected shape {expected_shape} for {key}"
                )


def _generate(
    model: HookedTransformer,
    tokens: torch.Tensor,
    max_new_tokens: int,
) -> str:
    """Generate text from tokens."""
    output_ids = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return model.to_string(output_ids[0])


def _generate_with_hooks(
    model: HookedTransformer,
    tokens: torch.Tensor,
    hooks: list[tuple[str, callable]],
    max_new_tokens: int,
) -> str:
    """Generate text with hooks applied."""
    generated = tokens.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(generated, fwd_hooks=hooks)
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)

            if model.tokenizer.eos_token_id is not None:
                if next_token.item() == model.tokenizer.eos_token_id:
                    break

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    return model.to_string(generated[0])


def _compute_activation_diff(
    model: HookedTransformer,
    input_text: str,
    hooks: list[tuple[str, callable]],
) -> ActivationData | None:
    """Compute activation difference between original and intervened."""
    original = analyze_activation(model, input_text)

    tokens = model.to_tokens(input_text)
    n_layers = model.cfg.n_layers

    hook_points = []
    for layer in range(n_layers):
        hook_points.extend([
            f"blocks.{layer}.hook_resid_post",
            f"blocks.{layer}.hook_mlp_out",
            f"blocks.{layer}.attn.hook_result",
        ])

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_points,
            fwd_hooks=hooks,
        )

    residual_list = []
    mlp_list = []
    attn_list = []

    for layer in range(n_layers):
        residual_list.append(cache[f"blocks.{layer}.hook_resid_post"][0])
        mlp_list.append(cache[f"blocks.{layer}.hook_mlp_out"][0])
        attn_list.append(cache[f"blocks.{layer}.attn.hook_result"][0])

    intervened = ActivationData(
        residual_stream=torch.stack(residual_list, dim=0),
        mlp_output=torch.stack(mlp_list, dim=0),
        attn_output=torch.stack(attn_list, dim=0),
    )

    diff = ActivationData(
        residual_stream=intervened.residual_stream - original.residual_stream,
        mlp_output=intervened.mlp_output - original.mlp_output,
        attn_output=intervened.attn_output - original.attn_output,
    )

    return diff


def _compute_metrics(
    model: HookedTransformer,
    tokens: torch.Tensor,
    hooks: list[tuple[str, callable]],
) -> dict:
    """Compute intervention metrics."""
    with torch.no_grad():
        original_logits = model(tokens)
        original_probs = torch.softmax(original_logits[0, -1, :], dim=-1)

        intervened_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        intervened_probs = torch.softmax(intervened_logits[0, -1, :], dim=-1)

    kl_div = torch.sum(
        original_probs * (torch.log(original_probs + 1e-10) - torch.log(intervened_probs + 1e-10))
    ).item()

    original_top = original_logits[0, -1, :].argmax()
    logit_diff = (
        original_logits[0, -1, original_top] - intervened_logits[0, -1, original_top]
    ).item()

    prob_change = (original_probs[original_top] - intervened_probs[original_top]).item()

    return {
        "kl_divergence": kl_div,
        "logit_diff": logit_diff,
        "prob_change": prob_change,
        "original_top_prob": original_probs[original_top].item(),
        "intervened_top_prob": intervened_probs[original_top].item(),
    }


def extract_activations_for_injection(
    model: HookedTransformer,
    input_text: str,
    targets: list[InterventionTarget],
) -> dict[str, torch.Tensor]:
    """Extract activations from a model run for use in injection.

    Args:
        model: HookedTransformer model
        input_text: Input text to extract activations from
        targets: List of targets specifying which activations to extract

    Returns:
        Dict mapping "layer_{i}_{component}" to activation tensor
    """
    tokens = model.to_tokens(input_text)

    # Build hook points list
    hook_points = set()
    for target in targets:
        hook_point = get_component_hook_point(target.layer, target.component_type)
        hook_points.add(hook_point)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=list(hook_points))

    result = {}
    for target in targets:
        hook_point = get_component_hook_point(target.layer, target.component_type)
        activation = cache[hook_point][0]  # Remove batch dimension

        key = _get_activation_key(target)

        if target.component_id is not None:
            # Extract specific component
            if target.component_type.value == "attn_head":
                # activation shape: [seq, n_heads, d_head]
                result[key] = activation[:, target.component_id, :]
            else:
                # MLP neuron: [seq, d_mlp]
                result[key] = activation[:, target.component_id]
        else:
            result[key] = activation

    return result
