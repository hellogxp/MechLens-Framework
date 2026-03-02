"""MechLens activation analysis.

Extract activation distributions and perform causal tracing.
Per contract sections 3 and 4.
"""

import logging
from typing import Literal

import torch
from transformer_lens import HookedTransformer

from mechlens.types import ActivationData, CausalTraceResult

logger = logging.getLogger(__name__)


def analyze(
    model: HookedTransformer,
    input_text: str,
    include_logit_lens: bool = False,
) -> ActivationData:
    """Extract activation distributions from model.

    Args:
        model: HookedTransformer model
        input_text: Input text to analyze
        include_logit_lens: Whether to compute logit lens projections

    Returns:
        ActivationData with residual stream, MLP, and attention outputs
    """
    n_layers = model.cfg.n_layers

    # Build hook points list
    hook_points = []
    for layer in range(n_layers):
        hook_points.extend([
            f"blocks.{layer}.hook_resid_post",
            f"blocks.{layer}.hook_mlp_out",
            f"blocks.{layer}.attn.hook_result",
        ])

    # Run model with cache
    _, cache = model.run_with_cache(input_text, names_filter=hook_points)

    # Extract activations per layer
    residual_list = []
    mlp_list = []
    attn_list = []

    for layer in range(n_layers):
        # Residual stream after layer
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        residual_list.append(resid[0])  # [seq, d_model]

        # MLP output
        mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]
        mlp_list.append(mlp_out[0])  # [seq, d_model]

        # Attention output
        attn_out = cache[f"blocks.{layer}.attn.hook_result"]
        attn_list.append(attn_out[0])  # [seq, d_model]

    residual_stream = torch.stack(residual_list, dim=0)  # [layers, seq, d_model]
    mlp_output = torch.stack(mlp_list, dim=0)  # [layers, seq, d_model]
    attn_output = torch.stack(attn_list, dim=0)  # [layers, seq, d_model]

    # Compute logit lens projections if requested
    logit_lens = None
    if include_logit_lens:
        logit_lens = _compute_logit_lens(model, residual_stream)

    logger.info(
        f"Extracted activations: residual {residual_stream.shape}, "
        f"mlp {mlp_output.shape}, attn {attn_output.shape}"
    )

    return ActivationData(
        residual_stream=residual_stream,
        mlp_output=mlp_output,
        attn_output=attn_output,
        logit_lens=logit_lens,
    )


def _compute_logit_lens(
    model: HookedTransformer,
    residual_stream: torch.Tensor,
) -> torch.Tensor:
    """Compute logit lens projections.

    Project residual stream at each layer through unembedding to get
    per-layer vocabulary predictions.

    Args:
        model: HookedTransformer model
        residual_stream: [layers, seq, d_model]

    Returns:
        Logit lens tensor [layers, seq, vocab]
    """
    n_layers, seq_len, d_model = residual_stream.shape

    # Apply layer norm and unembedding
    logits_list = []
    for layer in range(n_layers):
        resid = residual_stream[layer]  # [seq, d_model]

        # Apply final layer norm
        normed = model.ln_final(resid)

        # Apply unembedding: [seq, d_model] @ [d_model, vocab] = [seq, vocab]
        logits = normed @ model.W_U

        if model.b_U is not None:
            logits = logits + model.b_U

        logits_list.append(logits)

    return torch.stack(logits_list, dim=0)  # [layers, seq, vocab]


def causal_trace(
    model: HookedTransformer,
    input_text: str,
    subject: str,
    component_type: Literal["mlp", "attn", "resid"] = "mlp",
    noise_std: float = 0.1,
) -> CausalTraceResult:
    """Perform causal tracing to identify important layers/components.

    Based on Meng et al. (2022) "Locating and Editing Factual Associations".

    Args:
        model: HookedTransformer model
        input_text: Input text containing the subject
        subject: Subject to trace (must appear in input_text)
        component_type: Component type to trace ("mlp", "attn", "resid")
        noise_std: Standard deviation of Gaussian noise for corruption

    Returns:
        CausalTraceResult with base/corrupted outputs and patch results
    """
    n_layers = model.cfg.n_layers
    tokens = model.to_tokens(input_text)
    token_strs = model.to_str_tokens(input_text)

    # Find subject token positions
    subject_tokens = model.to_str_tokens(subject)
    subject_start = _find_subject_position(token_strs, subject_tokens)

    if subject_start is None:
        raise ValueError(f"Subject '{subject}' not found in input")

    subject_end = subject_start + len(subject_tokens)

    # Get base output (clean run)
    with torch.no_grad():
        base_logits = model(tokens)
        base_output = _get_predicted_token(model, base_logits)

    # Get corrupted output (noise on subject embeddings)
    def corrupt_hook(activation, hook):
        # Add noise to subject token embeddings
        noise = torch.randn_like(activation[:, subject_start:subject_end, :]) * noise_std
        activation[:, subject_start:subject_end, :] += noise
        return activation

    with torch.no_grad():
        corrupted_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[("hook_embed", corrupt_hook)],
        )
        corrupted_output = _get_predicted_token(model, corrupted_logits)

    # Determine target token index (last position)
    target_idx = tokens.shape[1] - 1

    # Patch each layer/component and measure recovery
    patch_results = torch.zeros(n_layers, dtype=torch.float32)

    hook_component = _get_hook_component(component_type)

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.{hook_component}"

        # Cache clean activation
        _, clean_cache = model.run_with_cache(tokens, names_filter=[hook_name])
        clean_activation = clean_cache[hook_name]

        # Run corrupted with this layer patched to clean
        def patch_hook(activation, hook, clean_act=clean_activation):
            activation[:, subject_start:subject_end, :] = clean_act[:, subject_start:subject_end, :]
            return activation

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    ("hook_embed", corrupt_hook),
                    (hook_name, patch_hook),
                ],
            )

        # Measure recovery
        base_prob = _get_target_prob(base_logits, target_idx)
        corrupted_prob = _get_target_prob(corrupted_logits, target_idx)
        patched_prob = _get_target_prob(patched_logits, target_idx)

        # Recovery score: how much patching this layer recovers the original
        if base_prob - corrupted_prob > 1e-6:
            recovery = (patched_prob - corrupted_prob) / (base_prob - corrupted_prob)
        else:
            recovery = 0.0

        patch_results[layer] = recovery

    logger.info(
        f"Causal trace complete: top layer = {patch_results.argmax().item()}, "
        f"max recovery = {patch_results.max().item():.3f}"
    )

    return CausalTraceResult(
        base_output=base_output,
        corrupted_output=corrupted_output,
        patch_results=patch_results,
        component_type=component_type,
        target_token_idx=target_idx,
    )


def _find_subject_position(
    token_strs: list[str],
    subject_tokens: list[str],
) -> int | None:
    """Find starting position of subject tokens in token list."""
    n_tokens = len(token_strs)
    n_subject = len(subject_tokens)

    for i in range(n_tokens - n_subject + 1):
        # Check if subject tokens match at this position
        # Use approximate matching since tokenization may differ
        match = True
        for j, subj_tok in enumerate(subject_tokens):
            if subj_tok.strip() not in token_strs[i + j]:
                match = False
                break
        if match:
            return i

    return None


def _get_hook_component(component_type: str) -> str:
    """Get hook component name for component type."""
    if component_type == "mlp":
        return "hook_mlp_out"
    elif component_type == "attn":
        return "attn.hook_result"
    elif component_type == "resid":
        return "hook_resid_post"
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def _get_predicted_token(model: HookedTransformer, logits: torch.Tensor) -> str:
    """Get predicted token string from logits."""
    last_logits = logits[0, -1, :]  # [vocab]
    token_id = last_logits.argmax().item()
    return model.to_single_str_token(token_id)


def _get_target_prob(logits: torch.Tensor, target_idx: int) -> float:
    """Get probability of top token at target position."""
    probs = torch.softmax(logits[0, target_idx, :], dim=-1)
    return probs.max().item()


def get_layer_importance(
    causal_result: CausalTraceResult,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Get top-k most important layers from causal trace.

    Args:
        causal_result: CausalTraceResult from causal_trace()
        top_k: Number of top layers to return

    Returns:
        List of (layer_idx, recovery_score) tuples sorted by importance
    """
    patch_results = causal_result.patch_results
    values, indices = torch.topk(patch_results, min(top_k, len(patch_results)))

    return [(idx.item(), val.item()) for idx, val in zip(indices, values)]
