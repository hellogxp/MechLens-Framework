"""MechLens visualization utilities.

Common Plotly theme, figure export helpers, and color palette constants.
Per contract section 10 and R2 design decisions.
"""

import io
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

# Color palette for consistent styling across visualizations
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "highlight": "#9467bd",
    "background": "#ffffff",
    "grid": "#e5e5e5",
    "text": "#333333",
}

# Colorscales for heatmaps
COLORSCALES = {
    "attention": "Blues",
    "activation": "RdBu",
    "causal_trace": "Viridis",
    "circuit": "Plasma",
    "feature": "Greens",
    "comparison": "RdYlGn",
    "diverging": "RdBu",
}

# Model display names for consistent labeling
MODEL_DISPLAY_NAMES = {
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-7B": "Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B": "Llama 3.1-8B",
    "EleutherAI/pythia-1.4b": "Pythia-1.4B",
}

# Default figure dimensions
DEFAULT_WIDTH = 1000
DEFAULT_HEIGHT = 600
PAPER_WIDTH = 800  # For publication figures
PAPER_HEIGHT = 500

# Export settings
EXPORT_DPI = 300
EXPORT_FORMATS = ["pdf", "svg", "png"]


def get_mechlens_template() -> dict[str, Any]:
    """Get the MechLens Plotly template.

    Returns:
        Template dict for Plotly figures
    """
    return {
        "layout": {
            "font": {
                "family": "Arial, sans-serif",
                "size": 12,
                "color": COLORS["text"],
            },
            "paper_bgcolor": COLORS["background"],
            "plot_bgcolor": COLORS["background"],
            "colorway": [
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["positive"],
                COLORS["negative"],
                COLORS["highlight"],
            ],
            "xaxis": {
                "gridcolor": COLORS["grid"],
                "linecolor": COLORS["grid"],
                "zerolinecolor": COLORS["grid"],
            },
            "yaxis": {
                "gridcolor": COLORS["grid"],
                "linecolor": COLORS["grid"],
                "zerolinecolor": COLORS["grid"],
            },
            "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
        }
    }


def apply_mechlens_style(fig: go.Figure) -> go.Figure:
    """Apply MechLens styling to a Plotly figure.

    Args:
        fig: Plotly figure to style

    Returns:
        Styled figure
    """
    template = get_mechlens_template()

    fig.update_layout(
        font=template["layout"]["font"],
        paper_bgcolor=template["layout"]["paper_bgcolor"],
        plot_bgcolor=template["layout"]["plot_bgcolor"],
        margin=template["layout"]["margin"],
    )

    fig.update_xaxes(
        gridcolor=template["layout"]["xaxis"]["gridcolor"],
        linecolor=template["layout"]["xaxis"]["linecolor"],
    )

    fig.update_yaxes(
        gridcolor=template["layout"]["yaxis"]["gridcolor"],
        linecolor=template["layout"]["yaxis"]["linecolor"],
    )

    return fig


def create_figure(
    title: str = "",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    for_paper: bool = False,
) -> go.Figure:
    """Create a new figure with MechLens styling.

    Args:
        title: Figure title
        width: Figure width in pixels
        height: Figure height in pixels
        for_paper: Use paper-optimized dimensions

    Returns:
        New Plotly figure
    """
    if for_paper:
        width = PAPER_WIDTH
        height = PAPER_HEIGHT

    fig = go.Figure()
    fig.update_layout(
        title=title,
        width=width,
        height=height,
    )

    return apply_mechlens_style(fig)


def export_figure(
    fig: go.Figure,
    path: str | Path,
    format: str = "pdf",
    scale: float = 2.0,
) -> None:
    """Export a figure to file for paper inclusion.

    Args:
        fig: Plotly figure to export
        path: Output path (extension added if not present)
        format: Export format (pdf, svg, png)
        scale: Scale factor for raster formats
    """
    path = Path(path)

    # Add extension if not present
    if not path.suffix:
        path = path.with_suffix(f".{format}")

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Export based on format
    if format == "pdf":
        fig.write_image(str(path), format="pdf", scale=scale)
    elif format == "svg":
        fig.write_image(str(path), format="svg")
    elif format == "png":
        fig.write_image(str(path), format="png", scale=scale)
    else:
        raise ValueError(f"Unsupported format: {format}. Use: {EXPORT_FORMATS}")


def figure_to_bytes(fig: go.Figure, format: str = "png", scale: float = 2.0) -> bytes:
    """Convert a figure to bytes for embedding.

    Args:
        fig: Plotly figure
        format: Output format
        scale: Scale factor

    Returns:
        Figure as bytes
    """
    buffer = io.BytesIO()
    fig.write_image(buffer, format=format, scale=scale)
    buffer.seek(0)
    return buffer.read()


def get_model_display_name(model_name: str) -> str:
    """Get display name for a model.

    Args:
        model_name: Full model name

    Returns:
        Short display name
    """
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def format_layer_label(layer: int, component: str = "", head: int | None = None) -> str:
    """Format a layer/component label.

    Args:
        layer: Layer index
        component: Component type (attn, mlp, resid)
        head: Head index (optional)

    Returns:
        Formatted label
    """
    if head is not None:
        return f"L{layer}H{head}"
    elif component:
        return f"L{layer}.{component}"
    else:
        return f"L{layer}"


def create_heatmap_figure(
    data,
    x_labels: list[str],
    y_labels: list[str],
    title: str = "",
    colorscale: str = "RdBu",
    for_paper: bool = False,
) -> go.Figure:
    """Create a heatmap figure with MechLens styling.

    Args:
        data: 2D array of values
        x_labels: X-axis labels
        y_labels: Y-axis labels
        title: Figure title
        colorscale: Plotly colorscale name
        for_paper: Use paper-optimized dimensions

    Returns:
        Plotly figure with heatmap
    """
    fig = create_figure(title=title, for_paper=for_paper)

    fig.add_trace(
        go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            colorbar={"title": "Value"},
        )
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
    )

    return fig


def create_bar_chart(
    values: list[float],
    labels: list[str],
    title: str = "",
    orientation: str = "v",
    color: str | None = None,
    for_paper: bool = False,
) -> go.Figure:
    """Create a bar chart with MechLens styling.

    Args:
        values: Bar values
        labels: Bar labels
        title: Figure title
        orientation: 'v' for vertical, 'h' for horizontal
        color: Bar color (uses primary if None)
        for_paper: Use paper-optimized dimensions

    Returns:
        Plotly figure with bar chart
    """
    fig = create_figure(title=title, for_paper=for_paper)

    if color is None:
        color = COLORS["primary"]

    if orientation == "v":
        fig.add_trace(go.Bar(x=labels, y=values, marker_color=color))
    else:
        fig.add_trace(go.Bar(x=values, y=labels, orientation="h", marker_color=color))

    return fig


# Register MechLens template with Plotly
pio.templates["mechlens"] = go.layout.Template(layout=get_mechlens_template()["layout"])
pio.templates.default = "mechlens"
