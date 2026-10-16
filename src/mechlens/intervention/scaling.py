"""MechLens scaling intervention.

Scale (amplify or dampen) specified components' activations by a factor.
Per contract section 7 - PRIMARY intervention paradigm.
"""

import logging

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from mechlens.analysis.activation import analyze as analyze_activation
from mechlens.models.hook_manager import create_scaling_hook, get_component_hook_point
from mechlens.types import ActivationData, InterventionError, InterventionResult, InterventionTarget

logger = logging.getLogger(__name__)


def scale(
    model: HookedTransformer,
    input_text: str,
    targets: list[InterventionTarget],
    factor: float,
    max_new_tokens: int = 50,
) -> InterventionResult:
    """Scale specified components' activations by a factor.

    Works uniformly across all 4 model families (Qwen, Llama, Pythia).

    Args:
        model: HookedTransformer model
        input_text: Input text
        targets: List of InterventionTarget specifying components to scale
        factor: Scaling factor (0.0 = ablation, 1.0 = no change, 2.0 = amplify)
        max_new_tokens: Maximum new tokens to generate

    Returns:
        InterventionResult with original and intervened outputs

    Raises:
        InterventionError: If target out of range or factor < 0
    """
    if factor < 0.0:
        raise InterventionError(f"Scaling factor must be >= 0.0, got {factor}")

    _validate_targets(model, targets)

    tokens = model.to_tokens(input_text)

    # Get original output
    with torch.no_grad():
        original_output = _generate(model, tokens, max_new_tokens)

    # Build scaling hooks
    hooks = []
    for target in targets:
        hook_point = get_component_hook_point(target.layer, target.component_type)
        hook_fn = create_scaling_hook(target, factor)
        hooks.append((hook_point, hook_fn))

    # Get intervened output
    with torch.no_grad():
        intervened_output = _generate_with_hooks(model, tokens, hooks, max_new_tokens)

    # Compute activation difference (may fail for some models due to cache issues)
    try:
        activation_diff = _compute_activation_diff(model, input_text, hooks)
    except (KeyError, RuntimeError) as e:
        logger.debug(f"Activation diff skipped (non-critical): {e}")
        activation_diff = None

    # Compute metrics
    try:
        metrics = _compute_metrics(model, tokens, hooks, factor)
    except (KeyError, RuntimeError) as e:
        logger.debug(f"Metrics computation skipped (non-critical): {e}")
        metrics = {"scaling_factor": factor}

    logger.info(
        f"Scaling complete: {len(targets)} targets, factor={factor}, "
        f"KL divergence = {metrics.get('kl_divergence', 0.0):.4f}"
    )

    return InterventionResult(
        original_output=original_output,
        intervened_output=intervened_output,
        activation_diff=activation_diff,
        metrics=metrics,
    )


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
            f"blocks.{layer}.hook_attn_out",
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
        attn_list.append(cache[f"blocks.{layer}.hook_attn_out"][0])

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
    factor: float,
) -> dict:
    """Compute intervention metrics."""
    with torch.no_grad():
        original_logits = model(tokens)
        intervened_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    # Cast to float32 for numerical stability (fp16 models produce NaN otherwise)
    orig_probs = torch.softmax(original_logits[0, -1, :].float(), dim=-1)
    intv_probs = torch.softmax(intervened_logits[0, -1, :].float(), dim=-1)

    kl_div = F.kl_div(
        intv_probs.log(), orig_probs, reduction="sum", log_target=False
    ).item()
    kl_div = max(0.0, kl_div)  # clamp numerical artifacts

    original_top = original_logits[0, -1, :].argmax()
    logit_diff = (
        original_logits[0, -1, original_top].float() - intervened_logits[0, -1, original_top].float()
    ).item()

    prob_change = (orig_probs[original_top] - intv_probs[original_top]).item()

    return {
        "kl_divergence": kl_div,
        "logit_diff": logit_diff,
        "prob_change": prob_change,
        "scaling_factor": factor,
        "original_top_prob": orig_probs[original_top].item(),
        "intervened_top_prob": intv_probs[original_top].item(),
    }
