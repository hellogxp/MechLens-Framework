"""MechLens SAE feature decomposition.

Decompose activations into monosemantic features using Sparse Autoencoders.
Per contract section 6 and Bricken et al. (2023) methodology.
"""

import logging
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from mechlens.types import SAEFeature

logger = logging.getLogger(__name__)


def decompose(
    model: HookedTransformer,
    input_text: str,
    layer: int,
    sae_path: str | None = None,
    top_k: int = 20,
) -> list[SAEFeature]:
    """Decompose activations at a layer into monosemantic features.

    Uses Sparse Autoencoder (SAE) to decompose residual stream activations
    into interpretable features. Pythia SAEs are well-available via SAELens;
    Qwen/Llama SAEs may require community models or graceful fallback.

    Args:
        model: HookedTransformer model
        input_text: Input text to analyze
        layer: Target layer for decomposition
        sae_path: Path to SAE weights (None = use pretrained from SAELens)
        top_k: Number of top features to return

    Returns:
        List of SAEFeature sorted by activation strength
    """
    # Get residual stream activation at target layer
    hook_name = f"blocks.{layer}.hook_resid_post"
    _, cache = model.run_with_cache(input_text, names_filter=[hook_name])
    activation = cache[hook_name][0]  # [seq, d_model]

    # Try to load SAE
    sae = _load_sae(model, layer, sae_path)

    if sae is None:
        logger.warning(f"No SAE available for layer {layer}, returning empty features")
        return []

    # Decompose activation using SAE
    features = _decompose_with_sae(sae, activation, layer, top_k)

    logger.info(f"Decomposed into {len(features)} features at layer {layer}")
    return features


def _load_sae(
    model: HookedTransformer,
    layer: int,
    sae_path: str | None,
) -> "SAE | None":
    """Load SAE for the given model and layer."""
    if sae_path is not None:
        return _load_sae_from_path(sae_path)

    # Try SAELens for well-supported models
    model_name = _get_model_name(model)

    if "pythia" in model_name.lower():
        return _load_saelens_sae(model_name, layer)
    elif "qwen" in model_name.lower() or "llama" in model_name.lower():
        # Community SAEs may be available
        logger.info(f"SAELens support limited for {model_name}, attempting community SAE")
        return _try_community_sae(model_name, layer)
    else:
        logger.warning(f"No SAE support for model {model_name}")
        return None


def _load_sae_from_path(sae_path: str) -> "SAE | None":
    """Load SAE from a local path."""
    path = Path(sae_path)
    if not path.exists():
        logger.warning(f"SAE path not found: {sae_path}")
        return None

    try:
        # Load as a simple encoder-decoder SAE
        state_dict = torch.load(sae_path, map_location="cpu")
        return SimpleSAE.from_state_dict(state_dict)
    except Exception as e:
        logger.warning(f"Failed to load SAE from {sae_path}: {e}")
        return None


def _load_saelens_sae(model_name: str, layer: int) -> "SAE | None":
    """Load SAE from SAELens library."""
    try:
        from sae_lens import SAE as SAELensSAE

        # SAELens model naming convention
        if "pythia-1.4b" in model_name.lower():
            sae_id = f"pythia-1.4b-deduped-res-sm"
        elif "pythia-70m" in model_name.lower():
            sae_id = f"pythia-70m-deduped-res-sm"
        else:
            logger.warning(f"No SAELens SAE for {model_name}")
            return None

        sae = SAELensSAE.from_pretrained(
            release=sae_id,
            sae_id=f"blocks.{layer}.hook_resid_post",
        )
        return SAELensWrapper(sae)

    except ImportError:
        logger.warning("SAELens not installed, cannot load pretrained SAE")
        return None
    except Exception as e:
        logger.warning(f"Failed to load SAELens SAE: {e}")
        return None


def _try_community_sae(model_name: str, layer: int) -> "SAE | None":
    """Try to load community-trained SAE for newer models."""
    # Placeholder for community SAE loading
    # In practice, this would check HuggingFace or other repositories
    logger.info(f"No community SAE found for {model_name} layer {layer}")
    return None


def _get_model_name(model: HookedTransformer) -> str:
    """Extract model name from HookedTransformer."""
    if hasattr(model.cfg, "model_name"):
        return model.cfg.model_name
    return str(model.cfg.tokenizer_name)


def _decompose_with_sae(
    sae,
    activation: torch.Tensor,
    layer: int,
    top_k: int,
) -> list[SAEFeature]:
    """Decompose activation using SAE encoder."""
    # Encode activation to get feature activations
    # activation: [seq, d_model] -> features: [seq, n_features]
    feature_acts = sae.encode(activation)

    # Average across sequence positions for overall feature importance
    avg_acts = feature_acts.mean(dim=0)  # [n_features]

    # Get top-k features by average activation
    top_values, top_indices = torch.topk(avg_acts.abs(), min(top_k, len(avg_acts)))

    features = []
    for idx, val in zip(top_indices, top_values):
        feature_idx = idx.item()
        activation_strength = val.item()

        # Get description if available
        description = sae.get_feature_description(feature_idx) if hasattr(sae, "get_feature_description") else None

        features.append(SAEFeature(
            feature_idx=feature_idx,
            layer=layer,
            activation=activation_strength,
            description=description,
            top_examples=[],  # Would require additional analysis
        ))

    return features


class SimpleSAE:
    """Simple Sparse Autoencoder implementation."""

    def __init__(self, W_enc: torch.Tensor, W_dec: torch.Tensor, b_enc: torch.Tensor, b_dec: torch.Tensor):
        self.W_enc = W_enc  # [d_model, n_features]
        self.W_dec = W_dec  # [n_features, d_model]
        self.b_enc = b_enc  # [n_features]
        self.b_dec = b_dec  # [d_model]

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "SimpleSAE":
        return cls(
            W_enc=state_dict["W_enc"],
            W_dec=state_dict["W_dec"],
            b_enc=state_dict["b_enc"],
            b_dec=state_dict["b_dec"],
        )

    def encode(self, activation: torch.Tensor) -> torch.Tensor:
        """Encode activation to feature space."""
        # activation: [seq, d_model]
        # output: [seq, n_features]
        pre_acts = activation @ self.W_enc + self.b_enc
        return torch.relu(pre_acts)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to activation space."""
        return features @ self.W_dec + self.b_dec


class SAELensWrapper:
    """Wrapper for SAELens SAE to provide consistent interface."""

    def __init__(self, sae):
        self.sae = sae

    def encode(self, activation: torch.Tensor) -> torch.Tensor:
        """Encode activation using SAELens SAE."""
        return self.sae.encode(activation)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features using SAELens SAE."""
        return self.sae.decode(features)

    def get_feature_description(self, feature_idx: int) -> str | None:
        """Get description for a feature if available."""
        # SAELens may have feature descriptions via Neuronpedia
        return None


def get_feature_activation_map(
    model: HookedTransformer,
    input_text: str,
    layer: int,
    feature_idx: int,
    sae_path: str | None = None,
) -> torch.Tensor:
    """Get activation of a specific feature across all tokens.

    Args:
        model: HookedTransformer model
        input_text: Input text
        layer: Target layer
        feature_idx: Feature index to analyze
        sae_path: Path to SAE weights

    Returns:
        Feature activation tensor [seq]
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    _, cache = model.run_with_cache(input_text, names_filter=[hook_name])
    activation = cache[hook_name][0]

    sae = _load_sae(model, layer, sae_path)
    if sae is None:
        raise ValueError(f"No SAE available for layer {layer}")

    feature_acts = sae.encode(activation)  # [seq, n_features]
    return feature_acts[:, feature_idx]  # [seq]


def compare_features(
    model: HookedTransformer,
    input_text1: str,
    input_text2: str,
    layer: int,
    sae_path: str | None = None,
    top_k: int = 10,
) -> dict[str, list[SAEFeature]]:
    """Compare feature activations between two inputs.

    Args:
        model: HookedTransformer model
        input_text1: First input text
        input_text2: Second input text
        layer: Target layer
        sae_path: Path to SAE weights
        top_k: Number of top features per input

    Returns:
        Dict with 'input1', 'input2', 'shared', 'unique1', 'unique2' feature lists
    """
    features1 = decompose(model, input_text1, layer, sae_path, top_k * 2)
    features2 = decompose(model, input_text2, layer, sae_path, top_k * 2)

    # Get feature indices
    indices1 = {f.feature_idx for f in features1[:top_k]}
    indices2 = {f.feature_idx for f in features2[:top_k]}

    shared = indices1 & indices2
    unique1 = indices1 - indices2
    unique2 = indices2 - indices1

    return {
        "input1": features1[:top_k],
        "input2": features2[:top_k],
        "shared": [f for f in features1 if f.feature_idx in shared],
        "unique1": [f for f in features1 if f.feature_idx in unique1],
        "unique2": [f for f in features2 if f.feature_idx in unique2],
    }
