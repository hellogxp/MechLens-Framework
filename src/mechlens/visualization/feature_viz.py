"""MechLens SAE feature visualization.

Render feature decomposition results as bar charts and activation maps.
Per contract section 10.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mechlens.types import SAEFeature
from mechlens.visualization import (
    COLORSCALES,
    COLORS,
    apply_mechlens_style,
    create_figure,
)


def render(
    features: list[SAEFeature],
    top_k: int = 20,
    show_descriptions: bool = True,
    for_paper: bool = False,
) -> go.Figure:
    """Render SAE feature decomposition as bar chart.

    Args:
        features: List of SAEFeature from features.decompose()
        top_k: Number of top features to show
        show_descriptions: Whether to show feature descriptions
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    if not features:
        return _render_empty(for_paper)

    # Take top_k features
    display_features = features[:top_k]

    fig = create_figure(
        title=f"Top {len(display_features)} SAE Features at Layer {features[0].layer}",
        width=800 if not for_paper else 600,
        height=500 if not for_paper else 400,
        for_paper=for_paper,
    )

    # Feature IDs and activations
    feature_ids = [f"F{f.feature_idx}" for f in display_features]
    activations = [f.activation for f in display_features]

    # Color by activation strength
    max_act = max(activations) if activations else 1
    colors = [f"rgba(31, 119, 180, {0.3 + 0.7 * a / max_act})" for a in activations]

    # Hover text with descriptions
    hover_texts = []
    for f in display_features:
        text = f"Feature {f.feature_idx}<br>Activation: {f.activation:.4f}"
        if show_descriptions and f.description:
            text += f"<br>Description: {f.description}"
        hover_texts.append(text)

    fig.add_trace(go.Bar(
        x=feature_ids,
        y=activations,
        marker_color=colors,
        hoverinfo="text",
        hovertext=hover_texts,
    ))

    fig.update_layout(
        xaxis_title="Feature ID",
        yaxis_title="Activation Strength",
        xaxis=dict(tickangle=45 if len(display_features) > 10 else 0),
    )

    return apply_mechlens_style(fig)


def _render_empty(for_paper: bool) -> go.Figure:
    """Render empty feature visualization."""
    fig = create_figure(
        title="SAE Feature Decomposition",
        width=600 if not for_paper else 400,
        height=400 if not for_paper else 300,
        for_paper=for_paper,
    )

    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text="No SAE features found or SAE not available",
        showarrow=False,
        font=dict(size=14, color="gray"),
    )

    return apply_mechlens_style(fig)


def render_feature_comparison(
    features1: list[SAEFeature],
    features2: list[SAEFeature],
    label1: str = "Input 1",
    label2: str = "Input 2",
    top_k: int = 10,
    for_paper: bool = False,
) -> go.Figure:
    """Compare SAE features between two inputs.

    Args:
        features1: Features from first input
        features2: Features from second input
        label1: Label for first input
        label2: Label for second input
        top_k: Number of features to compare
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[label1, label2],
    )

    # First input features
    f1 = features1[:top_k]
    if f1:
        fig.add_trace(
            go.Bar(
                x=[f"F{f.feature_idx}" for f in f1],
                y=[f.activation for f in f1],
                marker_color=COLORS["primary"],
                name=label1,
            ),
            row=1, col=1,
        )

    # Second input features
    f2 = features2[:top_k]
    if f2:
        fig.add_trace(
            go.Bar(
                x=[f"F{f.feature_idx}" for f in f2],
                y=[f.activation for f in f2],
                marker_color=COLORS["secondary"],
                name=label2,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title="Feature Comparison",
        height=400,
        width=900 if not for_paper else 700,
        showlegend=False,
    )

    return apply_mechlens_style(fig)


def render_feature_activation_map(
    activation_map: "torch.Tensor",
    tokens: list[str],
    feature_idx: int,
    layer: int,
    for_paper: bool = False,
) -> go.Figure:
    """Render activation map of a specific feature across tokens.

    Args:
        activation_map: Feature activation per token [seq]
        tokens: Token labels
        feature_idx: Feature index
        layer: Layer number
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    import numpy as np

    activations = activation_map.cpu().numpy() if hasattr(activation_map, 'cpu') else np.array(activation_map)

    fig = create_figure(
        title=f"Feature {feature_idx} Activation at Layer {layer}",
        width=700 if not for_paper else 500,
        height=350 if not for_paper else 250,
        for_paper=for_paper,
    )

    # Color by activation value
    max_val = max(abs(activations.max()), abs(activations.min()), 1e-6)
    colors = [f"rgba(31, 119, 180, {abs(a) / max_val})" for a in activations]

    fig.add_trace(go.Bar(
        x=tokens,
        y=activations,
        marker_color=colors,
        hovertemplate="%{x}<br>Activation: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Token",
        yaxis_title="Feature Activation",
    )

    return apply_mechlens_style(fig)


def render_feature_summary(
    features: list[SAEFeature],
    for_paper: bool = False,
) -> go.Figure:
    """Render summary statistics of feature decomposition.

    Shows distribution of activation strengths and layer coverage.
    """
    if not features:
        return _render_empty(for_paper)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Activation Distribution", "Feature Count by Strength"],
    )

    activations = [f.activation for f in features]

    # Histogram of activations
    fig.add_trace(
        go.Histogram(
            x=activations,
            nbinsx=20,
            marker_color=COLORS["primary"],
            name="Activations",
        ),
        row=1, col=1,
    )

    # Cumulative count by activation threshold
    import numpy as np
    thresholds = np.linspace(0, max(activations), 20)
    counts = [sum(1 for a in activations if a >= t) for t in thresholds]

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=counts,
            mode="lines+markers",
            line=dict(color=COLORS["secondary"]),
            name="Features Above Threshold",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Feature Summary ({len(features)} total features)",
        height=400,
        width=900 if not for_paper else 700,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Activation Strength", row=1, col=1)
    fig.update_xaxes(title_text="Activation Threshold", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Features Above", row=1, col=2)

    return apply_mechlens_style(fig)
