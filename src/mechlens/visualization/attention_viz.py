"""MechLens attention visualization.

Render attention patterns as interactive heatmaps.
Per contract section 10.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mechlens.types import AttentionData
from mechlens.visualization import (
    COLORSCALES,
    apply_mechlens_style,
    create_figure,
    get_model_display_name,
)


def render(
    data: AttentionData,
    layer: int | None = None,
    head: int | None = None,
    tokens: list[str] | None = None,
    show_all_heads: bool = False,
    for_paper: bool = False,
) -> go.Figure:
    """Render attention patterns as heatmap.

    Args:
        data: AttentionData from attention.analyze()
        layer: Specific layer to visualize (None = all layers overview)
        head: Specific head to visualize (None = all heads in layer)
        tokens: Token labels for axes
        show_all_heads: Show all heads in a grid
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    patterns = data.patterns  # [layers, heads, seq, seq]
    n_layers, n_heads, seq_len, _ = patterns.shape

    if layer is None:
        # Layer overview: show average attention per layer
        return _render_layer_overview(data, tokens, for_paper)

    if head is None and show_all_heads:
        # Show all heads in selected layer
        return _render_all_heads(data, layer, tokens, for_paper)

    if head is None:
        head = 0

    # Single head attention pattern
    return _render_single_head(data, layer, head, tokens, for_paper)


def _render_single_head(
    data: AttentionData,
    layer: int,
    head: int,
    tokens: list[str] | None,
    for_paper: bool,
) -> go.Figure:
    """Render attention pattern for a single head."""
    pattern = data.patterns[layer, head].cpu().numpy()  # [seq, seq]
    seq_len = pattern.shape[0]

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    fig = create_figure(
        title=f"Attention Pattern: Layer {layer}, Head {head}",
        width=600 if for_paper else 700,
        height=500 if for_paper else 600,
        for_paper=for_paper,
    )

    fig.add_trace(go.Heatmap(
        z=pattern,
        x=tokens,
        y=tokens,
        colorscale=COLORSCALES["attention"],
        showscale=True,
        colorbar=dict(title="Attention"),
        hovertemplate="From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        yaxis=dict(autorange="reversed"),
    )

    return apply_mechlens_style(fig)


def _render_all_heads(
    data: AttentionData,
    layer: int,
    tokens: list[str] | None,
    for_paper: bool,
) -> go.Figure:
    """Render all attention heads in a layer as a grid."""
    patterns = data.patterns[layer]  # [heads, seq, seq]
    n_heads = patterns.shape[0]
    seq_len = patterns.shape[1]

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    # Create subplot grid
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Head {h}" for h in range(n_heads)],
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    for h in range(n_heads):
        row = h // cols + 1
        col = h % cols + 1

        pattern = patterns[h].cpu().numpy()

        fig.add_trace(
            go.Heatmap(
                z=pattern,
                colorscale=COLORSCALES["attention"],
                showscale=(h == 0),
                colorbar=dict(title="Attn", x=1.02) if h == 0 else None,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Layer {layer} - All Attention Heads",
        height=200 * rows + 100,
        width=200 * cols + 150,
    )

    return apply_mechlens_style(fig)


def _render_layer_overview(
    data: AttentionData,
    tokens: list[str] | None,
    for_paper: bool,
) -> go.Figure:
    """Render overview of attention across all layers."""
    patterns = data.patterns  # [layers, heads, seq, seq]
    n_layers, n_heads, seq_len, _ = patterns.shape

    # Compute average attention per layer (across heads and positions)
    layer_avg = patterns.mean(dim=(1, 2, 3)).cpu().numpy()  # [layers]

    # Also compute attention entropy per layer
    import torch
    entropy = -torch.sum(
        patterns * torch.log(patterns + 1e-10), dim=-1
    ).mean(dim=(1, 2)).cpu().numpy()  # [layers]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Average Attention", "Attention Entropy"],
    )

    # Average attention bar chart
    fig.add_trace(
        go.Bar(
            x=list(range(n_layers)),
            y=layer_avg,
            name="Avg Attention",
            marker_color="#1f77b4",
        ),
        row=1, col=1,
    )

    # Entropy line chart
    fig.add_trace(
        go.Scatter(
            x=list(range(n_layers)),
            y=entropy,
            mode="lines+markers",
            name="Entropy",
            line=dict(color="#ff7f0e"),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title="Attention Overview Across Layers",
        height=400,
        width=900 if not for_paper else 700,
    )

    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=1, col=2)
    fig.update_yaxes(title_text="Average Attention", row=1, col=1)
    fig.update_yaxes(title_text="Entropy", row=1, col=2)

    return apply_mechlens_style(fig)


def render_attention_flow(
    data: AttentionData,
    source_position: int,
    tokens: list[str] | None = None,
    for_paper: bool = False,
) -> go.Figure:
    """Render attention flow from a specific source position.

    Shows how attention flows from one token position to others
    across all layers.
    """
    patterns = data.patterns  # [layers, heads, seq, seq]
    n_layers, n_heads, seq_len, _ = patterns.shape

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    # Get attention from source to all positions, averaged across heads
    flow = patterns[:, :, :, source_position].mean(dim=1).cpu().numpy()  # [layers, seq]

    fig = create_figure(
        title=f"Attention Flow from Position {source_position}: '{tokens[source_position]}'",
        width=800 if not for_paper else 600,
        height=500 if not for_paper else 400,
        for_paper=for_paper,
    )

    fig.add_trace(go.Heatmap(
        z=flow,
        x=tokens,
        y=[f"Layer {i}" for i in range(n_layers)],
        colorscale=COLORSCALES["attention"],
        colorbar=dict(title="Attention"),
    ))

    fig.update_layout(
        xaxis_title="Target Position",
        yaxis_title="Layer",
    )

    return apply_mechlens_style(fig)
