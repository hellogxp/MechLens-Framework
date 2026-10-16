"""MechLens analysis modules.

Provides attention pattern extraction, activation analysis, causal tracing,
logit lens projections, circuit discovery, contrastive analysis, and SAE
feature decomposition.
"""

from mechlens.analysis import activation, attention, circuit, contrastive, features, logit_lens
from mechlens.analysis.activation import analyze as analyze_activation
from mechlens.analysis.activation import causal_trace
from mechlens.analysis.attention import analyze as analyze_attention
from mechlens.analysis.circuit import discover as discover_circuit
from mechlens.analysis.contrastive import run_contrastive_analysis
from mechlens.analysis.features import decompose as decompose_features
from mechlens.analysis.logit_lens import compute_logit_lens

__all__ = [
    # Submodules
    "attention",
    "activation",
    "logit_lens",
    "circuit",
    "contrastive",
    "features",
    # Convenience functions
    "analyze_attention",
    "analyze_activation",
    "causal_trace",
    "compute_logit_lens",
    "discover_circuit",
    "run_contrastive_analysis",
    "decompose_features",
]
