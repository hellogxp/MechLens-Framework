"""MechLens intervention-specific visualization.

Render intervention results with diff, logit shift, and side-by-side views.
Per contract section 10.
"""

from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mechlens.types import InterventionResult
from mechlens.visualization import (
    COLORS,
    apply_mechlens_style,
    create_figure,
)


def render(
    result: InterventionResult,
    view: Literal["diff", "logit_shift", "side_by_side", "summary"] = "summary",
    tokens: list[str] | None = None,
    for_paper: bool = False,
) -> go.Figure:
    """Render intervention result visualization.

    Args:
        result: InterventionResult from intervention
        view: Visualization type
        tokens: Token labels for activation diff
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    if view == "diff":
        return _render_diff_view(result, tokens, for_paper)
    elif view == "logit_shift":
        return _render_logit_shift(result, for_paper)
    elif view == "side_by_side":
        return _render_side_by_side(result, for_paper)
    else:
        return _render_summary(result, for_paper)


def _render_summary(
    result: InterventionResult,
    for_paper: bool,
) -> go.Figure:
    """Render summary of intervention effect."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Output Change", "Metrics"],
        column_widths=[0.6, 0.4],
    )

    # Output comparison as text annotations
    # Using a simple indicator for output change
    output_changed = result.original_output != result.intervened_output

    fig.add_trace(
        go.Indicator(
            mode="delta",
            value=1 if output_changed else 0,
            delta=dict(reference=0),
            title=dict(text="Output Changed"),
            domain=dict(row=0, column=0),
        ),
        row=1, col=1,
    )

    # Metrics bar chart
    metrics = result.metrics
    metric_names = ["kl_divergence", "logit_diff", "prob_change"]
    metric_values = [metrics.get(m, 0) for m in metric_names]
    metric_labels = ["KL Div", "Logit Diff", "Prob Change"]

    fig.add_trace(
        go.Bar(
            x=metric_labels,
            y=metric_values,
            marker_color=[COLORS["primary"], COLORS["secondary"], COLORS["neutral"]],
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title="Intervention Summary",
        height=400,
        width=800 if not for_paper else 600,
    )

    # Add text annotations for outputs
    fig.add_annotation(
        x=0.15, y=0.3,
        xref="paper", yref="paper",
        text=f"Original: {result.original_output[:50]}...",
        showarrow=False,
        font=dict(size=10),
        bgcolor="white",
    )
    fig.add_annotation(
        x=0.15, y=0.15,
        xref="paper", yref="paper",
        text=f"Intervened: {result.intervened_output[:50]}...",
        showarrow=False,
        font=dict(size=10),
        bgcolor="white",
    )

    return apply_mechlens_style(fig)


def _render_diff_view(
    result: InterventionResult,
    tokens: list[str] | None,
    for_paper: bool,
) -> go.Figure:
    """Render activation difference heatmap."""
    if result.activation_diff is None:
        return _render_no_diff(for_paper)

    diff = result.activation_diff.residual_stream  # [layers, seq, d_model]
    n_layers, seq_len, _ = diff.shape

    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_len)]

    # Compute L2 norm of difference
    diff_norms = np.linalg.norm(diff.cpu().numpy(), axis=2)  # [layers, seq]

    fig = create_figure(
        title="Activation Difference (L2 Norm)",
        width=800 if not for_paper else 600,
        height=500 if not for_paper else 400,
        for_paper=for_paper,
    )

    fig.add_trace(go.Heatmap(
        z=diff_norms,
        x=tokens,
        y=[f"Layer {i}" for i in range(n_layers)],
        colorscale="RdBu",
        colorbar=dict(title="Diff Norm"),
        hovertemplate="Position: %{x}<br>Layer: %{y}<br>Diff: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Token Position",
        yaxis_title="Layer",
    )

    return apply_mechlens_style(fig)


def _render_logit_shift(
    result: InterventionResult,
    for_paper: bool,
) -> go.Figure:
    """Render logit/probability shifts."""
    metrics = result.metrics

    original_prob = metrics.get("original_top_prob", 0)
    intervened_prob = metrics.get("intervened_top_prob", 0)
    kl_div = metrics.get("kl_divergence", 0)
    logit_diff = metrics.get("logit_diff", 0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Probability Shift", "Divergence Metrics"],
    )

    # Probability comparison
    fig.add_trace(
        go.Bar(
            x=["Original", "Intervened"],
            y=[original_prob, intervened_prob],
            marker_color=[COLORS["primary"], COLORS["secondary"]],
            name="Top Token Prob",
        ),
        row=1, col=1,
    )

    # Divergence metrics
    fig.add_trace(
        go.Bar(
            x=["KL Divergence", "Logit Diff"],
            y=[kl_div, logit_diff],
            marker_color=[COLORS["negative"], COLORS["neutral"]],
            name="Divergence",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title="Logit and Probability Shifts",
        height=400,
        width=800 if not for_paper else 600,
        showlegend=False,
    )

    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)

    return apply_mechlens_style(fig)


def _render_side_by_side(
    result: InterventionResult,
    for_paper: bool,
) -> go.Figure:
    """Render side-by-side output comparison."""
    fig = create_figure(
        title="Output Comparison",
        width=800 if not for_paper else 600,
        height=300 if not for_paper else 250,
        for_paper=for_paper,
    )

    # Create text boxes for outputs
    original = result.original_output
    intervened = result.intervened_output

    # Highlight differences
    changed = original != intervened
    change_indicator = "CHANGED" if changed else "SAME"
    indicator_color = COLORS["negative"] if changed else COLORS["positive"]

    fig.add_annotation(
        x=0.25, y=0.7,
        xref="paper", yref="paper",
        text=f"<b>Original Output:</b><br>{original[:100]}{'...' if len(original) > 100 else ''}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor=COLORS["primary"],
        borderwidth=2,
        borderpad=10,
        align="left",
    )

    fig.add_annotation(
        x=0.75, y=0.7,
        xref="paper", yref="paper",
        text=f"<b>Intervened Output:</b><br>{intervened[:100]}{'...' if len(intervened) > 100 else ''}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor=COLORS["secondary"],
        borderwidth=2,
        borderpad=10,
        align="left",
    )

    fig.add_annotation(
        x=0.5, y=0.15,
        xref="paper", yref="paper",
        text=f"<b>Status: {change_indicator}</b>",
        showarrow=False,
        font=dict(size=16, color=indicator_color),
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return apply_mechlens_style(fig)


def _render_no_diff(for_paper: bool) -> go.Figure:
    """Render placeholder when no activation diff available."""
    fig = create_figure(
        title="Activation Difference",
        width=600 if not for_paper else 400,
        height=400 if not for_paper else 300,
        for_paper=for_paper,
    )

    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text="Activation difference not available",
        showarrow=False,
        font=dict(size=14, color="gray"),
    )

    return apply_mechlens_style(fig)


def render_batch_results(
    results: list[InterventionResult],
    for_paper: bool = False,
) -> go.Figure:
    """Render summary of batch intervention results.

    Args:
        results: List of InterventionResult from batch_intervene
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    if not results:
        return _render_no_diff(for_paper)

    # Aggregate metrics
    kl_divs = [r.metrics.get("kl_divergence", 0) for r in results]
    prob_changes = [r.metrics.get("prob_change", 0) for r in results]
    output_changed = [r.original_output != r.intervened_output for r in results]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["KL Divergence", "Probability Change", "Output Changed"],
    )

    # KL divergence histogram
    fig.add_trace(
        go.Histogram(x=kl_divs, marker_color=COLORS["primary"], name="KL Div"),
        row=1, col=1,
    )

    # Probability change histogram
    fig.add_trace(
        go.Histogram(x=prob_changes, marker_color=COLORS["secondary"], name="Prob Change"),
        row=1, col=2,
    )

    # Output change pie chart
    changed_count = sum(output_changed)
    unchanged_count = len(output_changed) - changed_count

    fig.add_trace(
        go.Pie(
            values=[changed_count, unchanged_count],
            labels=["Changed", "Unchanged"],
            marker_colors=[COLORS["negative"], COLORS["positive"]],
        ),
        row=1, col=3,
    )

    fig.update_layout(
        title=f"Batch Intervention Results ({len(results)} samples)",
        height=400,
        width=1000 if not for_paper else 800,
        showlegend=False,
    )

    return apply_mechlens_style(fig)
