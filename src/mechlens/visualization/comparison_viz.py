"""MechLens before/after comparison visualization.

Render side-by-side activation comparisons for intervention effects.
Per contract section 10.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mechlens.types import ActivationData
from mechlens.visualization import (
    COLORSCALES,
    COLORS,
    apply_mechlens_style,
    create_figure,
)


def render(
    before: ActivationData,
    after: ActivationData,
    layer: int | None = None,
    tokens: list[str] | None = None,
    view: str = "side_by_side",
    for_paper: bool = False,
) -> go.Figure:
    """Render before/after activation comparison.

    Args:
        before: ActivationData before intervention
        after: ActivationData after intervention
        layer: Specific layer to compare (None = all layers overview)
        tokens: Token labels
        view: Visualization type ("side_by_side", "diff", "overlay")
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    if view == "diff":
        return _render_diff(before, after, layer, tokens, for_paper)
    elif view == "overlay":
        return _render_overlay(before, after, layer, tokens, for_paper)
    else:
        return _render_side_by_side(before, after, layer, tokens, for_paper)


def _render_side_by_side(
    before: ActivationData,
    after: ActivationData,
    layer: int | None,
    tokens: list[str] | None,
    for_paper: bool,
) -> go.Figure:
    """Render side-by-side comparison."""
    n_layers = before.residual_stream.shape[0]
    seq_len = before.residual_stream.shape[1]

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    if layer is not None:
        # Single layer comparison
        before_resid = before.residual_stream[layer].cpu().numpy()
        after_resid = after.residual_stream[layer].cpu().numpy()

        before_norms = np.linalg.norm(before_resid, axis=1)
        after_norms = np.linalg.norm(after_resid, axis=1)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Before Intervention", "After Intervention"],
        )

        fig.add_trace(
            go.Bar(x=tokens, y=before_norms, marker_color=COLORS["primary"], name="Before"),
            row=1, col=1,
        )

        fig.add_trace(
            go.Bar(x=tokens, y=after_norms, marker_color=COLORS["secondary"], name="After"),
            row=1, col=2,
        )

        fig.update_layout(
            title=f"Activation Comparison at Layer {layer}",
            height=400,
            width=900 if not for_paper else 700,
            showlegend=False,
        )

    else:
        # All layers heatmap comparison
        before_norms = np.linalg.norm(before.residual_stream.cpu().numpy(), axis=2)
        after_norms = np.linalg.norm(after.residual_stream.cpu().numpy(), axis=2)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Before Intervention", "After Intervention"],
        )

        # Find common scale
        vmax = max(before_norms.max(), after_norms.max())

        fig.add_trace(
            go.Heatmap(
                z=before_norms,
                x=tokens,
                y=[f"L{i}" for i in range(n_layers)],
                colorscale=COLORSCALES["activation"],
                zmin=0, zmax=vmax,
                showscale=False,
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=after_norms,
                x=tokens,
                y=[f"L{i}" for i in range(n_layers)],
                colorscale=COLORSCALES["activation"],
                zmin=0, zmax=vmax,
                colorbar=dict(title="L2 Norm"),
            ),
            row=1, col=2,
        )

        fig.update_layout(
            title="Activation Comparison Across Layers",
            height=500,
            width=1000 if not for_paper else 800,
        )

    return apply_mechlens_style(fig)


def _render_diff(
    before: ActivationData,
    after: ActivationData,
    layer: int | None,
    tokens: list[str] | None,
    for_paper: bool,
) -> go.Figure:
    """Render difference heatmap."""
    n_layers = before.residual_stream.shape[0]
    seq_len = before.residual_stream.shape[1]

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    # Compute difference
    diff = after.residual_stream - before.residual_stream

    if layer is not None:
        # Single layer diff
        diff_layer = diff[layer].cpu().numpy()
        diff_norms = np.linalg.norm(diff_layer, axis=1)

        fig = create_figure(
            title=f"Activation Difference at Layer {layer}",
            width=700 if not for_paper else 500,
            height=400 if not for_paper else 300,
            for_paper=for_paper,
        )

        # Color positive/negative changes
        colors = [COLORS["positive"] if d > 0 else COLORS["negative"] for d in diff_norms]

        fig.add_trace(go.Bar(
            x=tokens,
            y=diff_norms,
            marker_color=colors,
            hovertemplate="%{x}<br>Diff: %{y:.4f}<extra></extra>",
        ))

        fig.update_layout(
            xaxis_title="Token",
            yaxis_title="Activation Difference (L2 Norm)",
        )

    else:
        # All layers diff heatmap
        diff_norms = np.linalg.norm(diff.cpu().numpy(), axis=2)

        fig = create_figure(
            title="Activation Difference Across Layers",
            width=800 if not for_paper else 600,
            height=500 if not for_paper else 400,
            for_paper=for_paper,
        )

        fig.add_trace(go.Heatmap(
            z=diff_norms,
            x=tokens,
            y=[f"L{i}" for i in range(n_layers)],
            colorscale=COLORSCALES["diverging"],
            colorbar=dict(title="Diff (L2)"),
        ))

        fig.update_layout(
            xaxis_title="Token",
            yaxis_title="Layer",
        )

    return apply_mechlens_style(fig)


def _render_overlay(
    before: ActivationData,
    after: ActivationData,
    layer: int | None,
    tokens: list[str] | None,
    for_paper: bool,
) -> go.Figure:
    """Render overlaid line comparison."""
    n_layers = before.residual_stream.shape[0]
    seq_len = before.residual_stream.shape[1]

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    if layer is None:
        layer = n_layers // 2  # Default to middle layer

    before_resid = before.residual_stream[layer].cpu().numpy()
    after_resid = after.residual_stream[layer].cpu().numpy()

    before_norms = np.linalg.norm(before_resid, axis=1)
    after_norms = np.linalg.norm(after_resid, axis=1)

    fig = create_figure(
        title=f"Activation Overlay at Layer {layer}",
        width=700 if not for_paper else 500,
        height=400 if not for_paper else 300,
        for_paper=for_paper,
    )

    fig.add_trace(go.Scatter(
        x=tokens,
        y=before_norms,
        mode="lines+markers",
        name="Before",
        line=dict(color=COLORS["primary"]),
        marker=dict(size=8),
    ))

    fig.add_trace(go.Scatter(
        x=tokens,
        y=after_norms,
        mode="lines+markers",
        name="After",
        line=dict(color=COLORS["secondary"], dash="dash"),
        marker=dict(size=8),
    ))

    fig.update_layout(
        xaxis_title="Token",
        yaxis_title="Activation L2 Norm",
        legend=dict(x=0.02, y=0.98),
    )

    return apply_mechlens_style(fig)


def render_multi_layer_diff(
    before: ActivationData,
    after: ActivationData,
    layers: list[int],
    tokens: list[str] | None = None,
    for_paper: bool = False,
) -> go.Figure:
    """Render difference across multiple specific layers.

    Args:
        before: ActivationData before intervention
        after: ActivationData after intervention
        layers: List of layer indices to compare
        tokens: Token labels
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    seq_len = before.residual_stream.shape[1]

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    n_layers = len(layers)
    fig = make_subplots(
        rows=1, cols=n_layers,
        subplot_titles=[f"Layer {l}" for l in layers],
    )

    for i, layer in enumerate(layers, 1):
        before_resid = before.residual_stream[layer].cpu().numpy()
        after_resid = after.residual_stream[layer].cpu().numpy()

        diff = np.linalg.norm(after_resid - before_resid, axis=1)

        fig.add_trace(
            go.Bar(
                x=tokens,
                y=diff,
                marker_color=COLORS["secondary"],
                showlegend=False,
            ),
            row=1, col=i,
        )

    fig.update_layout(
        title="Activation Difference by Layer",
        height=350,
        width=300 * n_layers,
    )

    return apply_mechlens_style(fig)
