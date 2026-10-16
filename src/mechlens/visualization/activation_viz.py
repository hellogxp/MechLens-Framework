"""MechLens activation visualization.

Render activation distributions, logit lens, and causal trace heatmaps.
Per contract section 10.
"""

from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mechlens.types import ActivationData, CausalTraceResult
from mechlens.visualization import (
    COLORSCALES,
    apply_mechlens_style,
    create_figure,
)


def render(
    data: ActivationData | CausalTraceResult,
    layer: int | None = None,
    view: Literal["distribution", "logit_lens", "causal_trace"] = "distribution",
    tokens: list[str] | None = None,
    for_paper: bool = False,
) -> go.Figure:
    """Render activation data as heatmap or chart.

    Args:
        data: ActivationData or CausalTraceResult
        layer: Specific layer to visualize
        view: Visualization type
        tokens: Token labels
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    if isinstance(data, CausalTraceResult):
        return _render_causal_trace(data, for_paper)

    if view == "logit_lens" and data.logit_lens is not None:
        return _render_logit_lens(data, tokens, for_paper)
    elif view == "causal_trace":
        raise ValueError("CausalTraceResult required for causal_trace view")
    else:
        return _render_distribution(data, layer, tokens, for_paper)


def _render_distribution(
    data: ActivationData,
    layer: int | None,
    tokens: list[str] | None,
    for_paper: bool,
) -> go.Figure:
    """Render activation distribution heatmap."""
    resid = data.residual_stream  # [layers, seq, d_model]
    n_layers, seq_len, d_model = resid.shape

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    if layer is not None:
        # Single layer: show full activation pattern
        activation = resid[layer].cpu().numpy()  # [seq, d_model]

        # Reduce dimensionality for visualization (show top PCA components or stats)
        # For now, show L2 norm per position
        norms = np.linalg.norm(activation, axis=1)

        fig = create_figure(
            title=f"Activation Norms at Layer {layer}",
            for_paper=for_paper,
        )

        fig.add_trace(go.Bar(
            x=tokens,
            y=norms,
            marker_color="#1f77b4",
            name="L2 Norm",
        ))

        fig.update_layout(
            xaxis_title="Token Position",
            yaxis_title="Activation L2 Norm",
        )

    else:
        # All layers: show norm heatmap
        norms = np.linalg.norm(resid.cpu().numpy(), axis=2)  # [layers, seq]

        fig = create_figure(
            title="Activation Norms Across Layers",
            for_paper=for_paper,
        )

        fig.add_trace(go.Heatmap(
            z=norms,
            x=tokens,
            y=[f"L{i}" for i in range(n_layers)],
            colorscale=COLORSCALES["activation"],
            colorbar=dict(title="L2 Norm"),
        ))

        fig.update_layout(
            xaxis_title="Token Position",
            yaxis_title="Layer",
        )

    return apply_mechlens_style(fig)


def _render_logit_lens(
    data: ActivationData,
    tokens: list[str] | None,
    for_paper: bool,
) -> go.Figure:
    """Render logit lens visualization."""
    logit_lens = data.logit_lens  # [layers, seq, vocab]
    if logit_lens is None:
        raise ValueError("logit_lens data not available")

    n_layers, seq_len, vocab_size = logit_lens.shape

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    # Get top-1 probability at each layer/position
    probs = logit_lens.softmax(dim=-1)
    top_probs, top_indices = probs.max(dim=-1)  # [layers, seq]
    top_probs = top_probs.cpu().numpy()

    fig = create_figure(
        title="Logit Lens: Top Token Probability",
        width=800 if not for_paper else 600,
        height=500 if not for_paper else 400,
        for_paper=for_paper,
    )

    fig.add_trace(go.Heatmap(
        z=top_probs,
        x=tokens,
        y=[f"Layer {i}" for i in range(n_layers)],
        colorscale=COLORSCALES["activation"],
        colorbar=dict(title="Top-1 Prob"),
        hovertemplate="Position: %{x}<br>Layer: %{y}<br>Probability: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Token Position",
        yaxis_title="Layer",
    )

    return apply_mechlens_style(fig)


def _render_causal_trace(
    data: CausalTraceResult,
    for_paper: bool,
) -> go.Figure:
    """Render causal tracing results."""
    patch_results = data.patch_results.cpu().numpy()  # [layers]
    n_layers = len(patch_results)

    fig = create_figure(
        title=f"Causal Trace: {data.component_type.upper()} Recovery",
        width=700 if not for_paper else 500,
        height=400 if not for_paper else 300,
        for_paper=for_paper,
    )

    # Bar chart of recovery scores
    colors = ["#2ecc71" if r > 0.5 else "#e74c3c" if r < 0.2 else "#f39c12"
              for r in patch_results]

    fig.add_trace(go.Bar(
        x=list(range(n_layers)),
        y=patch_results,
        marker_color=colors,
        name="Recovery Score",
        hovertemplate="Layer %{x}<br>Recovery: %{y:.3f}<extra></extra>",
    ))

    # Add threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="50% recovery")

    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title="Recovery Score",
        yaxis_range=[0, 1],
    )

    # Add annotations for base and corrupted output
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"Base: {data.base_output}",
        showarrow=False,
        font=dict(size=10),
        bgcolor="white",
        borderpad=4,
    )
    fig.add_annotation(
        x=0.02, y=0.88,
        xref="paper", yref="paper",
        text=f"Corrupted: {data.corrupted_output}",
        showarrow=False,
        font=dict(size=10),
        bgcolor="white",
        borderpad=4,
    )

    return apply_mechlens_style(fig)


def render_component_comparison(
    residual: ActivationData,
    mlp: ActivationData | None = None,
    attn: ActivationData | None = None,
    layer: int = 0,
    tokens: list[str] | None = None,
    for_paper: bool = False,
) -> go.Figure:
    """Compare residual, MLP, and attention contributions at a layer."""
    resid = residual.residual_stream[layer]  # [seq, d_model]
    seq_len = resid.shape[0]

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Residual", "MLP Output", "Attention Output"],
    )

    # Residual norms
    resid_norms = np.linalg.norm(resid.cpu().numpy(), axis=1)
    fig.add_trace(
        go.Bar(x=tokens, y=resid_norms, marker_color="#1f77b4", name="Residual"),
        row=1, col=1,
    )

    # MLP norms
    if residual.mlp_output is not None:
        mlp_out = residual.mlp_output[layer]
        mlp_norms = np.linalg.norm(mlp_out.cpu().numpy(), axis=1)
        fig.add_trace(
            go.Bar(x=tokens, y=mlp_norms, marker_color="#ff7f0e", name="MLP"),
            row=1, col=2,
        )

    # Attention norms
    if residual.attn_output is not None:
        attn_out = residual.attn_output[layer]
        attn_norms = np.linalg.norm(attn_out.cpu().numpy(), axis=1)
        fig.add_trace(
            go.Bar(x=tokens, y=attn_norms, marker_color="#2ca02c", name="Attn"),
            row=1, col=3,
        )

    fig.update_layout(
        title=f"Component Comparison at Layer {layer}",
        height=400,
        width=1000 if not for_paper else 700,
        showlegend=False,
    )

    return apply_mechlens_style(fig)
