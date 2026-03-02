"""MechLens attention pattern analysis.

Extract attention patterns across all layers for visualization and circuit analysis.
Per contract section 2 and data-model.md AttentionData specification.
"""

import logging

import torch
from transformer_lens import HookedTransformer

from mechlens.types import AttentionData

logger = logging.getLogger(__name__)


def analyze(
    model: HookedTransformer,
    input_text: str,
    include_qk_scores: bool = False,
) -> AttentionData:
    """Extract attention patterns across all layers.

    Handles both GQA (Qwen/Llama: n_kv_heads < n_heads) and
    MHA (Pythia: n_kv_heads == n_heads) architectures.

    Args:
        model: HookedTransformer model
        input_text: Input text to analyze
        include_qk_scores: Whether to include raw QK scores

    Returns:
        AttentionData with patterns and head labels
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Determine hook points to cache
    hook_points = [f"blocks.{layer}.attn.hook_pattern" for layer in range(n_layers)]
    if include_qk_scores:
        hook_points.extend(
            [f"blocks.{layer}.attn.hook_attn_scores" for layer in range(n_layers)]
        )

    # Run model with cache
    _, cache = model.run_with_cache(input_text, names_filter=hook_points)

    # Extract attention patterns: [layers, heads, seq, seq]
    patterns_list = []
    for layer in range(n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        # pattern shape: [batch, n_heads, seq, seq]
        patterns_list.append(pattern[0])  # Remove batch dimension

    patterns = torch.stack(patterns_list, dim=0)  # [layers, heads, seq, seq]

    # Generate head labels
    head_labels = [f"L{layer}H{head}" for layer in range(n_layers) for head in range(n_heads)]

    # Extract QK scores if requested
    qk_scores = None
    if include_qk_scores:
        qk_list = []
        for layer in range(n_layers):
            scores = cache[f"blocks.{layer}.attn.hook_attn_scores"]
            qk_list.append(scores[0])  # Remove batch dimension
        qk_scores = torch.stack(qk_list, dim=0)

    logger.info(
        f"Extracted attention patterns: {patterns.shape} "
        f"({n_layers} layers, {n_heads} heads)"
    )

    return AttentionData(
        patterns=patterns,
        head_labels=head_labels,
        qk_scores=qk_scores,
    )


def get_head_pattern(
    attention_data: AttentionData,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Get attention pattern for a specific head.

    Args:
        attention_data: AttentionData from analyze()
        layer: Layer index
        head: Head index

    Returns:
        Attention pattern tensor [seq, seq]
    """
    return attention_data.patterns[layer, head]


def compute_attention_entropy(
    attention_data: AttentionData,
    layer: int | None = None,
) -> torch.Tensor:
    """Compute attention entropy per head.

    Higher entropy indicates more uniform attention distribution.
    Lower entropy indicates more focused attention.

    Args:
        attention_data: AttentionData from analyze()
        layer: Specific layer (None = all layers)

    Returns:
        Entropy tensor [layers, heads] or [heads] if layer specified
    """
    if layer is not None:
        patterns = attention_data.patterns[layer:layer+1]
    else:
        patterns = attention_data.patterns

    # Compute entropy: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    log_patterns = torch.log(patterns + eps)
    entropy = -torch.sum(patterns * log_patterns, dim=-1).mean(dim=-1)

    if layer is not None:
        return entropy[0]  # [heads]
    return entropy  # [layers, heads]


def find_induction_heads(
    model: HookedTransformer,
    input_text: str,
    threshold: float = 0.8,
) -> list[tuple[int, int, float]]:
    """Identify potential induction heads.

    Induction heads show a specific pattern: attending to tokens that follow
    tokens similar to the current token (AB...AB pattern).

    Args:
        model: HookedTransformer model
        input_text: Input text (should contain repeated patterns)
        threshold: Minimum induction score

    Returns:
        List of (layer, head, induction_score) tuples
    """
    attention_data = analyze(model, input_text)
    patterns = attention_data.patterns

    n_layers, n_heads, seq_len, _ = patterns.shape
    induction_heads = []

    # Check for induction pattern: strong attention to positions that are
    # offset by a fixed amount from repeated tokens
    for layer in range(n_layers):
        for head in range(n_heads):
            pattern = patterns[layer, head]

            # Simple induction score: check diagonal offset pattern
            # Strong induction heads attend to position i-1 when at position i
            # for repeated sequences
            if seq_len < 4:
                continue

            # Compute average attention to "previous occurrence" positions
            # This is a simplified heuristic
            off_diagonal = torch.diagonal(pattern, offset=-1)
            avg_off_diag = off_diagonal.mean().item()

            # Normalized by expected uniform attention
            induction_score = avg_off_diag * seq_len

            if induction_score > threshold:
                induction_heads.append((layer, head, induction_score))

    # Sort by induction score
    induction_heads.sort(key=lambda x: x[2], reverse=True)

    logger.info(f"Found {len(induction_heads)} potential induction heads")
    return induction_heads
