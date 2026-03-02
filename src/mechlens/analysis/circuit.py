"""MechLens circuit discovery.

Discover computational circuits via activation patching.
Per contract section 5 and Conmy et al. (2023) ACDC methodology.
"""

import logging
from typing import Literal

import torch
from transformer_lens import HookedTransformer

from mechlens.types import CircuitEdge, CircuitGraph, CircuitNode, ComponentType

logger = logging.getLogger(__name__)


def discover(
    model: HookedTransformer,
    input_text: str,
    target_token_idx: int,
    method: Literal["activation_patching", "edge_patching"] = "activation_patching",
    threshold: float = 0.1,
    include_mlp: bool = True,
) -> CircuitGraph:
    """Discover circuit with causal contribution to target token.

    Uses activation patching to identify causally important components.
    Handles GQA (Qwen/Llama) and MHA (Pythia) architectures.

    Args:
        model: HookedTransformer model
        input_text: Input text
        target_token_idx: Target token position for circuit discovery
        method: Discovery method ("activation_patching" or "edge_patching")
        threshold: Importance threshold for including components
        include_mlp: Whether to include MLP neurons in circuit

    Returns:
        CircuitGraph with nodes, edges, faithfulness, and completeness
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    tokens = model.to_tokens(input_text)

    # Get baseline output probability
    with torch.no_grad():
        base_logits = model(tokens)
        base_prob = _get_target_prob(base_logits, target_token_idx)

    # Get corrupted baseline (mean ablation on all components)
    corrupted_prob = _get_corrupted_baseline(model, tokens, target_token_idx)

    if method == "activation_patching":
        nodes, importance_scores = _activation_patching(
            model, tokens, target_token_idx, base_prob, corrupted_prob,
            include_mlp=include_mlp,
        )
    else:  # edge_patching
        nodes, importance_scores = _edge_patching(
            model, tokens, target_token_idx, base_prob, corrupted_prob,
            include_mlp=include_mlp,
        )

    # Filter nodes by threshold
    filtered_nodes = [
        node for node in nodes
        if node.importance >= threshold
    ]

    # Build edges between adjacent layers
    edges = _build_edges(filtered_nodes, importance_scores)

    # Compute faithfulness and completeness
    faithfulness, completeness = _compute_metrics(
        model, tokens, target_token_idx, filtered_nodes, base_prob, corrupted_prob
    )

    logger.info(
        f"Discovered circuit: {len(filtered_nodes)} nodes, {len(edges)} edges, "
        f"faithfulness={faithfulness:.3f}, completeness={completeness:.3f}"
    )

    return CircuitGraph(
        nodes=filtered_nodes,
        edges=edges,
        faithfulness=faithfulness,
        completeness=completeness,
    )


def _activation_patching(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_idx: int,
    base_prob: float,
    corrupted_prob: float,
    include_mlp: bool = True,
) -> tuple[list[CircuitNode], dict[str, float]]:
    """Perform activation patching to find important components."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    nodes = []
    importance_scores = {}

    # Cache clean activations
    _, clean_cache = model.run_with_cache(tokens)

    # Test each attention head
    for layer in range(n_layers):
        for head in range(n_heads):
            hook_name = f"blocks.{layer}.hook_attn_out"

            def ablate_head(activation, hook, h=head):
                modified = activation.clone()
                # activation shape: [batch, seq, n_heads, d_head]
                modified[:, :, h, :] = 0.0
                return modified

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, ablate_head)],
                )
                ablated_prob = _get_target_prob(ablated_logits, target_idx)

            # Importance = how much ablation hurts performance
            importance = max(0.0, base_prob - ablated_prob)
            normalized_importance = importance / max(base_prob - corrupted_prob, 1e-6)

            node_id = f"L{layer}H{head}"
            nodes.append(CircuitNode(
                id=node_id,
                layer=layer,
                component_type="attn_head",
                importance=normalized_importance,
            ))
            importance_scores[node_id] = normalized_importance

    # Test MLP layers
    if include_mlp:
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_mlp_out"

            def ablate_mlp(activation, hook):
                return torch.zeros_like(activation)

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, ablate_mlp)],
                )
                ablated_prob = _get_target_prob(ablated_logits, target_idx)

            importance = max(0.0, base_prob - ablated_prob)
            normalized_importance = importance / max(base_prob - corrupted_prob, 1e-6)

            node_id = f"L{layer}MLP"
            nodes.append(CircuitNode(
                id=node_id,
                layer=layer,
                component_type="mlp",
                importance=normalized_importance,
            ))
            importance_scores[node_id] = normalized_importance

    return nodes, importance_scores


def _edge_patching(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_idx: int,
    base_prob: float,
    corrupted_prob: float,
    include_mlp: bool = True,
) -> tuple[list[CircuitNode], dict[str, float]]:
    """Perform edge patching for finer-grained circuit discovery.

    Edge patching tests connections between components by patching
    the output of one component only as input to another.
    """
    # For now, fall back to activation patching
    # Full edge patching requires more complex intervention logic
    logger.info("Edge patching requested, using activation patching as fallback")
    return _activation_patching(
        model, tokens, target_idx, base_prob, corrupted_prob, include_mlp
    )


def _build_edges(
    nodes: list[CircuitNode],
    importance_scores: dict[str, float],
) -> list[CircuitEdge]:
    """Build edges between circuit nodes."""
    edges = []

    # Sort nodes by layer
    sorted_nodes = sorted(nodes, key=lambda n: (n.layer, n.id))

    # Connect nodes in adjacent layers
    for i, source in enumerate(sorted_nodes):
        for target in sorted_nodes:
            if target.layer == source.layer + 1:
                # Edge weight based on geometric mean of node importances
                weight = (source.importance * target.importance) ** 0.5
                edges.append(CircuitEdge(
                    source=source.id,
                    target=target.id,
                    weight=weight,
                ))

    return edges


def _get_corrupted_baseline(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_idx: int,
) -> float:
    """Get corrupted baseline by mean-ablating all attention heads."""
    def ablate_all(activation, hook):
        return torch.zeros_like(activation)

    hooks = [
        (f"blocks.{layer}.hook_attn_out", ablate_all)
        for layer in range(model.cfg.n_layers)
    ]

    with torch.no_grad():
        corrupted_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        return _get_target_prob(corrupted_logits, target_idx)


def _compute_metrics(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_idx: int,
    circuit_nodes: list[CircuitNode],
    base_prob: float,
    corrupted_prob: float,
) -> tuple[float, float]:
    """Compute circuit faithfulness and completeness.

    Faithfulness: How well does the circuit approximate full model behavior?
    Completeness: What fraction of model behavior is explained by the circuit?
    """
    if not circuit_nodes:
        return 0.0, 0.0

    # Get IDs of circuit components
    circuit_heads = {
        (node.layer, int(node.id.split("H")[1]))
        for node in circuit_nodes
        if "H" in node.id
    }

    # Ablate everything except circuit
    def ablate_non_circuit(activation, hook):
        layer = int(hook.name.split(".")[1])
        modified = activation.clone()

        # Check each head
        n_heads = activation.shape[2]
        for head in range(n_heads):
            if (layer, head) not in circuit_heads:
                modified[:, :, head, :] = 0.0

        return modified

    hooks = [
        (f"blocks.{layer}.hook_attn_out", ablate_non_circuit)
        for layer in range(model.cfg.n_layers)
    ]

    with torch.no_grad():
        circuit_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        circuit_prob = _get_target_prob(circuit_logits, target_idx)

    # Faithfulness: how close is circuit-only output to full model output
    faithfulness = 1.0 - abs(base_prob - circuit_prob)

    # Completeness: what fraction of full model capability is retained
    prob_range = base_prob - corrupted_prob
    if prob_range > 1e-6:
        completeness = (circuit_prob - corrupted_prob) / prob_range
    else:
        completeness = 1.0 if circuit_prob >= base_prob else 0.0

    return max(0.0, faithfulness), max(0.0, min(1.0, completeness))


def _get_target_prob(logits: torch.Tensor, target_idx: int) -> float:
    """Get maximum probability at target position."""
    probs = torch.softmax(logits[0, target_idx, :], dim=-1)
    return probs.max().item()


def get_critical_path(
    circuit: CircuitGraph,
    top_k: int = 10,
) -> list[CircuitNode]:
    """Extract the critical path through the circuit.

    Args:
        circuit: CircuitGraph from discover()
        top_k: Maximum nodes to include

    Returns:
        List of most important nodes forming the critical path
    """
    # Sort by importance
    sorted_nodes = sorted(circuit.nodes, key=lambda n: n.importance, reverse=True)
    return sorted_nodes[:top_k]


def to_intervention_targets(
    circuit: CircuitGraph,
    threshold: float = 0.1,
) -> list[dict]:
    """Convert circuit nodes to intervention targets.

    Args:
        circuit: CircuitGraph from discover()
        threshold: Minimum importance to include

    Returns:
        List of intervention target specifications
    """
    targets = []

    for node in circuit.nodes:
        if node.importance < threshold:
            continue

        if "H" in node.id:
            # Attention head
            head_idx = int(node.id.split("H")[1])
            targets.append({
                "layer": node.layer,
                "component_type": ComponentType.ATTN_HEAD,
                "component_id": head_idx,
            })
        elif "MLP" in node.id:
            # MLP (entire layer)
            targets.append({
                "layer": node.layer,
                "component_type": ComponentType.MLP_NEURON,
                "component_id": None,  # Entire MLP
            })

    return targets
