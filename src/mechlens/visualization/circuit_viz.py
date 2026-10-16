"""MechLens circuit graph visualization.

Render circuit graphs with NetworkX layout and interactive nodes/edges.
Per contract section 10.
"""

from typing import Literal

import networkx as nx
import plotly.graph_objects as go

from mechlens.types import CircuitGraph
from mechlens.visualization import (
    COLORS,
    apply_mechlens_style,
    create_figure,
)


def render(
    graph: CircuitGraph,
    layout: Literal["layered", "spring", "circular"] = "layered",
    show_edge_weights: bool = True,
    highlight_nodes: list[str] | None = None,
    for_paper: bool = False,
) -> go.Figure:
    """Render circuit graph as interactive visualization.

    Args:
        graph: CircuitGraph from circuit.discover()
        layout: Graph layout algorithm
        show_edge_weights: Whether to show edge weight labels
        highlight_nodes: Node IDs to highlight
        for_paper: Use paper-quality settings

    Returns:
        Plotly Figure
    """
    if not graph.nodes:
        return _render_empty_graph(for_paper)

    # Build NetworkX graph
    G = nx.DiGraph()

    for node in graph.nodes:
        G.add_node(node.id, layer=node.layer, importance=node.importance,
                   component_type=node.component_type)

    for edge in graph.edges:
        G.add_edge(edge.source, edge.target, weight=edge.weight)

    # Compute layout
    pos = _compute_layout(G, layout)

    # Create figure
    fig = create_figure(
        title=f"Circuit Graph (Faithfulness: {graph.faithfulness:.2f}, Completeness: {graph.completeness:.2f})",
        width=900 if not for_paper else 700,
        height=600 if not for_paper else 500,
        for_paper=for_paper,
    )

    # Add edges
    edge_traces = _create_edge_traces(G, pos, show_edge_weights)
    for trace in edge_traces:
        fig.add_trace(trace)

    # Add nodes
    node_trace = _create_node_trace(G, pos, highlight_nodes)
    fig.add_trace(node_trace)

    # Add layer labels
    _add_layer_labels(fig, G, pos)

    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    return apply_mechlens_style(fig)


def _compute_layout(G: nx.DiGraph, layout: str) -> dict:
    """Compute node positions using specified layout."""
    if layout == "layered":
        # Custom layered layout based on node layer attribute
        pos = {}
        layers = {}
        for node, data in G.nodes(data=True):
            layer = data.get("layer", 0)
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)

        for layer, nodes in layers.items():
            for i, node in enumerate(sorted(nodes)):
                x = layer
                y = (i - len(nodes) / 2) * 1.5
                pos[node] = (x, y)

    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50)

    elif layout == "circular":
        pos = nx.circular_layout(G)

    else:
        pos = nx.spring_layout(G)

    return pos


def _create_edge_traces(G: nx.DiGraph, pos: dict, show_weights: bool) -> list:
    """Create edge traces with arrows."""
    traces = []

    for edge in G.edges(data=True):
        source, target, data = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        weight = data.get("weight", 0.5)

        # Edge line
        traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(
                width=max(1, weight * 5),
                color=f"rgba(100, 100, 100, {min(1, weight + 0.3)})",
            ),
            hoverinfo="none",
        ))

        # Arrow marker at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        traces.append(go.Scatter(
            x=[mx],
            y=[my],
            mode="markers",
            marker=dict(
                symbol="triangle-right",
                size=8,
                color="gray",
                angle=_compute_angle(x0, y0, x1, y1),
            ),
            hoverinfo="text",
            hovertext=f"{source} → {target}<br>Weight: {weight:.3f}",
        ))

    return traces


def _compute_angle(x0: float, y0: float, x1: float, y1: float) -> float:
    """Compute angle for arrow direction."""
    import math
    dx = x1 - x0
    dy = y1 - y0
    return math.degrees(math.atan2(dy, dx))


def _create_node_trace(
    G: nx.DiGraph,
    pos: dict,
    highlight_nodes: list[str] | None,
) -> go.Scatter:
    """Create node trace with importance-based sizing."""
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        importance = data.get("importance", 0.5)
        component_type = data.get("component_type", "unknown")

        node_text.append(f"{node}<br>Type: {component_type}<br>Importance: {importance:.3f}")

        # Color by component type
        if component_type == "attn_head":
            color = COLORS["primary"]
        elif component_type == "mlp":
            color = COLORS["secondary"]
        else:
            color = COLORS["neutral"]

        # Highlight if specified
        if highlight_nodes and node in highlight_nodes:
            color = COLORS["highlight"]

        node_color.append(color)
        node_size.append(15 + importance * 30)

    return go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color="white"),
        ),
        text=[n for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=10),
        hoverinfo="text",
        hovertext=node_text,
    )


def _add_layer_labels(fig: go.Figure, G: nx.DiGraph, pos: dict) -> None:
    """Add layer labels to the figure."""
    layers = set(data.get("layer", 0) for _, data in G.nodes(data=True))

    for layer in layers:
        # Find y range for this layer
        y_vals = [pos[n][1] for n, d in G.nodes(data=True) if d.get("layer") == layer]
        if y_vals:
            fig.add_annotation(
                x=layer,
                y=min(y_vals) - 1,
                text=f"Layer {layer}",
                showarrow=False,
                font=dict(size=12, color="gray"),
            )


def _render_empty_graph(for_paper: bool) -> go.Figure:
    """Render empty graph placeholder."""
    fig = create_figure(
        title="Circuit Graph (No nodes found)",
        width=600 if not for_paper else 400,
        height=400 if not for_paper else 300,
        for_paper=for_paper,
    )

    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text="No circuit nodes found above threshold",
        showarrow=False,
        font=dict(size=16, color="gray"),
    )

    return apply_mechlens_style(fig)


def render_critical_path(
    graph: CircuitGraph,
    top_k: int = 10,
    for_paper: bool = False,
) -> go.Figure:
    """Render the critical path through the circuit.

    Shows the most important nodes in order of their contribution.
    """
    # Get top-k nodes by importance
    sorted_nodes = sorted(graph.nodes, key=lambda n: n.importance, reverse=True)[:top_k]

    fig = create_figure(
        title="Critical Path: Top Components by Importance",
        width=700 if not for_paper else 500,
        height=400 if not for_paper else 300,
        for_paper=for_paper,
    )

    # Bar chart of importance
    node_ids = [n.id for n in sorted_nodes]
    importances = [n.importance for n in sorted_nodes]
    colors = [COLORS["primary"] if "H" in n.id else COLORS["secondary"] for n in sorted_nodes]

    fig.add_trace(go.Bar(
        x=node_ids,
        y=importances,
        marker_color=colors,
        hovertemplate="%{x}<br>Importance: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Component",
        yaxis_title="Importance",
    )

    return apply_mechlens_style(fig)
