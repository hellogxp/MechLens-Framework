"""Improved Causal Tracing with Noise-Based Perturbation (v2).

Key improvements over v1 (activation.causal_trace):
1. Adaptive noise calibrated to embedding standard deviation
2. Multi-run averaging for stable scores on small models
3. KL divergence metric for more sensitive measurement
4. Head-level granularity in addition to layer-level
5. Better handling of uniform/near-zero scores
"""
import logging
from typing import Literal

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from mechlens.types import CausalTraceResult

logger = logging.getLogger(__name__)


def _calibrate_noise(
    model: HookedTransformer,
    tokens: torch.Tensor,
    noise_factor: float = 3.0,
) -> float:
    """Calibrate noise level based on embedding standard deviation.

    Instead of fixed noise_std, compute std of actual embeddings
    and scale by noise_factor. This ensures corruption is meaningful
    regardless of model scale.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=["hook_embed"])
        embed = cache["hook_embed"]
        embed_std = embed.std().item()
    return embed_std * noise_factor


def _kl_divergence(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    target_idx: int,
) -> float:
    """Compute KL divergence at target position.

    KL(P || Q) where P = clean distribution, Q = corrupted/patched distribution.
    More sensitive than simple probability comparison.
    Computes in float32 for numerical stability with fp16 models.
    """
    # Cast to float32 to avoid fp16 precision issues
    p = F.softmax(logits_p[0, target_idx, :].float(), dim=-1)
    q = F.softmax(logits_q[0, target_idx, :].float(), dim=-1)
    kl = F.kl_div(q.log(), p, reduction="sum", log_target=False).item()
    # kl_div can return negative values due to numerical issues; clamp to 0
    return max(0.0, kl)


def run_causal_tracing_v2(
    model: HookedTransformer,
    input_text: str,
    subject: str,
    component_type: Literal["mlp", "attn", "resid"] = "mlp",
    noise_factor: float = 3.0,
    n_runs: int = 5,
    use_kl: bool = True,
) -> CausalTraceResult:
    """Improved causal tracing with adaptive noise and multi-run averaging.

    Args:
        model: HookedTransformer model
        input_text: Input text containing the subject
        subject: Subject to trace (must appear in input_text)
        component_type: Component type ('mlp', 'attn', 'resid')
        noise_factor: Multiplier for embedding std to get noise level
        n_runs: Number of runs to average (improves stability)
        use_kl: Use KL divergence metric instead of probability

    Returns:
        CausalTraceResult with averaged patch results
    """
    n_layers = model.cfg.n_layers
    tokens = model.to_tokens(input_text)
    token_strs = model.to_str_tokens(input_text)

    # Find subject token positions
    # to_str_tokens prepends BOS; strip it for matching
    subject_tokens = model.to_str_tokens(subject)
    bos = model.tokenizer.bos_token if hasattr(model.tokenizer, 'bos_token') and model.tokenizer.bos_token else None
    if bos and len(subject_tokens) > 0 and subject_tokens[0] == bos:
        subject_tokens = subject_tokens[1:]
    subject_pos = _find_subject_position(token_strs, subject_tokens)

    if subject_pos is None:
        raise ValueError(f"Subject not found in input: {subject}")

    subject_start, subject_end = subject_pos
    target_idx = tokens.shape[1] - 1

    # Calibrate noise to model scale
    noise_std = _calibrate_noise(model, tokens, noise_factor)
    logger.info(f"Calibrated noise_std = {noise_std:.4f} (factor={noise_factor})")

    # Get clean baseline
    with torch.no_grad():
        base_logits = model(tokens)
        base_output = _get_predicted_token(model, base_logits)

    # Multi-run averaging
    all_patch_results = []
    corrupted_output = None

    hook_component = _get_hook_component(component_type)

    for run_idx in range(n_runs):
        # Create corruption hook with fresh noise each run
        def make_corrupt_hook(std, s_start, s_end):
            def corrupt_hook(activation, hook):
                noise = torch.randn_like(
                    activation[:, s_start:s_end, :]
                ) * std
                activation[:, s_start:s_end, :] += noise
                return activation
            return corrupt_hook

        corrupt_fn = make_corrupt_hook(noise_std, subject_start, subject_end)

        # Get corrupted baseline
        with torch.no_grad():
            corrupted_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[("hook_embed", corrupt_fn)],
            )
            if corrupted_output is None:
                corrupted_output = _get_predicted_token(model, corrupted_logits)

        # Measure base vs corrupted gap
        if use_kl:
            base_score = _kl_divergence(base_logits, corrupted_logits, target_idx)
        else:
            base_score = (
                _get_target_prob(base_logits, target_idx)
                - _get_target_prob(corrupted_logits, target_idx)
            )

        # Patch each layer
        run_results = torch.zeros(n_layers, dtype=torch.float32)

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.{hook_component}"

            # Cache clean activation for this layer
            _, clean_cache = model.run_with_cache(
                tokens, names_filter=[hook_name]
            )
            clean_act = clean_cache[hook_name]

            # Run corrupted with this layer patched to clean
            def make_patch_hook(clean_activation, s_start, s_end):
                def patch_hook(activation, hook):
                    activation[:, s_start:s_end, :] = \
                        clean_activation[:, s_start:s_end, :]
                    return activation
                return patch_hook

            patch_fn = make_patch_hook(clean_act, subject_start, subject_end)

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        ("hook_embed", corrupt_fn),
                        (hook_name, patch_fn),
                    ],
                )

            # Compute recovery score
            if use_kl:
                patched_kl = _kl_divergence(
                    base_logits, patched_logits, target_idx
                )
                if base_score > 1e-6:
                    recovery = (base_score - patched_kl) / base_score
                else:
                    recovery = 0.0
            else:
                patched_prob = _get_target_prob(patched_logits, target_idx)
                corrupted_prob = _get_target_prob(corrupted_logits, target_idx)
                base_prob = _get_target_prob(base_logits, target_idx)
                if base_prob - corrupted_prob > 1e-6:
                    recovery = (patched_prob - corrupted_prob) / (base_prob - corrupted_prob)
                else:
                    recovery = 0.0

            run_results[layer] = recovery

        all_patch_results.append(run_results)

    # Average across runs
    avg_results = torch.stack(all_patch_results).mean(dim=0)

    # Check for uniform scores and warn
    score_std = avg_results.std().item()
    score_max = avg_results.max().item()
    if score_std < 0.01 and score_max > 0:
        logger.warning(
            f"Near-uniform causal trace scores (std={score_std:.4f}). "
            f"Consider increasing noise_factor or n_runs."
        )

    logger.info(
        f"Causal trace v2 complete ({n_runs} runs averaged): "
        f"top layer = {avg_results.argmax().item()}, "
        f"max recovery = {avg_results.max().item():.3f}, "
        f"score std = {score_std:.4f}"
    )

    return CausalTraceResult(
        base_output=base_output,
        corrupted_output=corrupted_output or "",
        patch_results=avg_results,
        component_type=component_type,
        target_token_idx=target_idx,
    )


def run_head_level_tracing(
    model: HookedTransformer,
    input_text: str,
    subject: str,
    noise_factor: float = 3.0,
    n_runs: int = 3,
) -> dict:
    """Run head-level causal tracing to identify specific attention heads.

    Unlike layer-level tracing, this identifies individual heads that are
    critical for factual recall. Essential for targeted interventions.

    Args:
        model: HookedTransformer model
        input_text: Input text
        subject: Subject entity to trace
        noise_factor: Noise calibration factor
        n_runs: Number of averaging runs

    Returns:
        Dict with head_scores [n_layers, n_heads], top_heads list
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    tokens = model.to_tokens(input_text)
    token_strs = model.to_str_tokens(input_text)

    subject_tokens = model.to_str_tokens(subject)
    bos = model.tokenizer.bos_token if hasattr(model.tokenizer, 'bos_token') and model.tokenizer.bos_token else None
    if bos and len(subject_tokens) > 0 and subject_tokens[0] == bos:
        subject_tokens = subject_tokens[1:]
    subject_pos = _find_subject_position(token_strs, subject_tokens)
    if subject_pos is None:
        raise ValueError(f"Subject not found: {subject}")
    subject_start, subject_end = subject_pos
    target_idx = tokens.shape[1] - 1

    noise_std = _calibrate_noise(model, tokens, noise_factor)

    with torch.no_grad():
        base_logits = model(tokens)

    all_head_scores = []

    for run_idx in range(n_runs):
        def make_corrupt_hook(std, s_start, s_end):
            def _corrupt(activation, hook):
                noise = torch.randn_like(
                    activation[:, s_start:s_end, :]
                ) * std
                activation[:, s_start:s_end, :] += noise
                return activation
            return _corrupt

        corrupt_fn = make_corrupt_hook(noise_std, subject_start, subject_end)

        with torch.no_grad():
            corrupted_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[("hook_embed", corrupt_fn)],
            )

        base_kl = _kl_divergence(base_logits, corrupted_logits, target_idx)

        head_scores = torch.zeros(n_layers, n_heads)

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.attn.hook_z"

            _, clean_cache = model.run_with_cache(
                tokens, names_filter=[hook_name]
            )
            clean_act = clean_cache[hook_name]  # [batch, seq, n_heads, d_head]

            for head in range(n_heads):
                def make_head_patch(clean_activation, h, s_start, s_end):
                    def _patch(activation, hook):
                        activation[:, s_start:s_end, h, :] = \
                            clean_activation[:, s_start:s_end, h, :]
                        return activation
                    return _patch

                patch_fn = make_head_patch(clean_act, head, subject_start, subject_end)

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[
                            ("hook_embed", corrupt_fn),
                            (hook_name, patch_fn),
                        ],
                    )

                patched_kl = _kl_divergence(
                    base_logits, patched_logits, target_idx
                )
                if base_kl > 1e-6:
                    recovery = (base_kl - patched_kl) / base_kl
                else:
                    recovery = 0.0

                head_scores[layer, head] = recovery

        all_head_scores.append(head_scores)

    avg_scores = torch.stack(all_head_scores).mean(dim=0)

    # Get top heads
    flat_scores = avg_scores.flatten()
    top_k = min(20, flat_scores.numel())
    top_values, top_indices = torch.topk(flat_scores, top_k)

    top_heads = []
    for val, idx in zip(top_values, top_indices):
        layer = idx.item() // n_heads
        head = idx.item() % n_heads
        top_heads.append({
            "layer": layer,
            "head": head,
            "recovery_score": val.item(),
        })

    logger.info(
        f"Head-level tracing: top head = L{top_heads[0]['layer']}H{top_heads[0]['head']} "
        f"(score={top_heads[0]['recovery_score']:.3f})"
    )

    return {
        "head_scores": avg_scores,
        "top_heads": top_heads,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }


def _find_subject_position(token_strs, subject_tokens):
    """Find starting and ending position of subject tokens using character-level matching.

    Handles tokenizers that split subjects differently (e.g., Pythia
    tokenizes 'ocean' as ['o', 'cean'] while the subject tokenizer
    produces ['ocean']). Falls back to character-level reconstruction
    for robust cross-model matching.

    Returns:
        Tuple of (start_idx, end_idx) or None if not found.
        end_idx is exclusive (i.e., subject spans token_strs[start:end]).
    """
    # First try exact token-level matching (fast path)
    n_tokens = len(token_strs)
    n_subject = len(subject_tokens)
    for i in range(n_tokens - n_subject + 1):
        match = True
        for j, subj_tok in enumerate(subject_tokens):
            if subj_tok.strip() not in token_strs[i + j]:
                match = False
                break
        if match:
            return (i, i + n_subject)

    # Fallback: character-level matching for different tokenizations.
    # Reconstruct subject string and search for it in the concatenated
    # token strings, tracking which token index each character maps to.
    subject_str = "".join(t.strip() for t in subject_tokens).lower()
    if not subject_str:
        return None

    # Build a mapping: for each character position in the joined string,
    # record which token index it belongs to
    char_to_token = []
    for tok_idx, tok in enumerate(token_strs):
        stripped = tok.replace(" ", "").replace("\u0120", "").replace("Ġ", "")
        for ch in stripped:
            char_to_token.append(tok_idx)

    joined = "".join(
        tok.replace(" ", "").replace("\u0120", "").replace("Ġ", "")
        for tok in token_strs
    ).lower()

    pos = joined.find(subject_str)
    if pos >= 0 and pos < len(char_to_token):
        start_tok = char_to_token[pos]
        end_char = pos + len(subject_str) - 1
        if end_char < len(char_to_token):
            end_tok = char_to_token[end_char] + 1  # exclusive
        else:
            end_tok = start_tok + 1
        return (start_tok, end_tok)

    return None


def _get_hook_component(component_type) -> str:
    """Get hook component name."""
    ct = component_type.value if hasattr(component_type, "value") else str(component_type).lower()
    if ct in ("mlp", "mlp_neuron"):
        return "hook_mlp_out"
    elif ct in ("attn", "attn_head"):
        return "hook_attn_out"
    elif ct == "resid":
        return "hook_resid_post"
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def _get_predicted_token(model, logits):
    """Get predicted token string from logits."""
    last_logits = logits[0, -1, :]
    token_id = last_logits.argmax().item()
    return model.to_single_str_token(token_id)


def _get_target_prob(logits, target_idx):
    """Get probability of top token at target position."""
    probs = torch.softmax(logits[0, target_idx, :], dim=-1)
    return probs.max().item()
