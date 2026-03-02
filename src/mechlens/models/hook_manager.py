"""MechLens hook management.

Activation hook registration/cleanup utilities for HookedTransformer.
Supports intervention infrastructure across all 4 model families.
Per contract and R3 design decisions.
"""

import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from mechlens.types import ComponentType, InterventionTarget

logger = logging.getLogger(__name__)


# Hook point name templates for different components
# These work across all 4 model families via TransformerLens abstraction
HOOK_POINTS = {
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "resid_mid": "blocks.{layer}.hook_resid_mid",
    "attn_out": "blocks.{layer}.attn.hook_result",
    "attn_pattern": "blocks.{layer}.attn.hook_pattern",
    "attn_scores": "blocks.{layer}.attn.hook_attn_scores",
    "attn_z": "blocks.{layer}.attn.hook_z",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
    "mlp_pre": "blocks.{layer}.mlp.hook_pre",
    "mlp_post": "blocks.{layer}.mlp.hook_post",
}


@dataclass
class HookSpec:
    """Specification for a hook."""

    name: str
    hook_point: str
    hook_fn: Callable
    enabled: bool = True


@dataclass
class HookManager:
    """Manage hooks for a HookedTransformer model."""

    model: HookedTransformer
    hooks: dict[str, HookSpec] = field(default_factory=dict)
    _active_hooks: list[tuple[str, Callable]] = field(default_factory=list)

    def get_hook_point(self, layer: int, component: str) -> str:
        """Get hook point name for a layer/component.

        Args:
            layer: Layer index
            component: Component type (resid_pre, attn_out, mlp_out, etc.)

        Returns:
            Hook point name string
        """
        if component not in HOOK_POINTS:
            raise ValueError(f"Unknown component: {component}. Valid: {list(HOOK_POINTS.keys())}")

        return HOOK_POINTS[component].format(layer=layer)

    def register_hook(
        self,
        name: str,
        layer: int,
        component: str,
        hook_fn: Callable[[torch.Tensor, HookPoint], torch.Tensor],
    ) -> None:
        """Register a hook.

        Args:
            name: Unique name for the hook
            layer: Layer index
            component: Component type
            hook_fn: Hook function (activation, hook) -> modified_activation
        """
        hook_point = self.get_hook_point(layer, component)

        self.hooks[name] = HookSpec(
            name=name,
            hook_point=hook_point,
            hook_fn=hook_fn,
        )

        logger.debug(f"Registered hook '{name}' at {hook_point}")

    def remove_hook(self, name: str) -> None:
        """Remove a registered hook by name."""
        if name in self.hooks:
            del self.hooks[name]
            logger.debug(f"Removed hook '{name}'")

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        self.hooks.clear()
        logger.debug("Cleared all hooks")

    def get_active_hooks(self) -> list[tuple[str, Callable]]:
        """Get list of active hooks for run_with_hooks.

        Returns:
            List of (hook_point, hook_fn) tuples
        """
        active = []
        for spec in self.hooks.values():
            if spec.enabled:
                active.append((spec.hook_point, spec.hook_fn))
        return active

    @contextmanager
    def hooks_context(self):
        """Context manager for running with registered hooks.

        Usage:
            with hook_manager.hooks_context():
                output = model.run_with_hooks(...)
        """
        try:
            yield self.get_active_hooks()
        finally:
            pass  # Hooks are managed externally


def create_ablation_hook(
    target: InterventionTarget,
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    """Create an ablation hook that zeros out activations.

    Args:
        target: Intervention target specification

    Returns:
        Hook function
    """

    def hook_fn(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        # Clone to avoid modifying original
        modified = activation.clone()

        if target.component_type == ComponentType.ATTN_HEAD:
            if target.component_id is not None:
                # Zero out specific head
                # activation shape: [batch, pos, n_heads, d_head] or [batch, n_heads, pos, pos]
                if len(modified.shape) == 4:
                    if modified.shape[2] == hook.model.cfg.n_heads:
                        # [batch, pos, n_heads, d_head]
                        modified[:, :, target.component_id, :] = 0.0
                    else:
                        # [batch, n_heads, pos, pos] (attention pattern)
                        modified[:, target.component_id, :, :] = 0.0
            else:
                # Zero out all heads
                modified.zero_()

        elif target.component_type == ComponentType.MLP_NEURON:
            if target.component_id is not None:
                # Zero out specific neuron
                # activation shape: [batch, pos, d_mlp]
                modified[:, :, target.component_id] = 0.0
            else:
                # Zero out entire MLP
                modified.zero_()

        elif target.component_type == ComponentType.RESID:
            # Zero out residual stream
            modified.zero_()

        return modified

    return hook_fn


def create_scaling_hook(
    target: InterventionTarget,
    factor: float,
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    """Create a scaling hook that multiplies activations.

    Args:
        target: Intervention target specification
        factor: Scaling factor (0.0 = ablation, 1.0 = no change, 2.0 = amplify)

    Returns:
        Hook function
    """
    if factor < 0.0:
        raise ValueError(f"Scaling factor must be >= 0.0, got {factor}")

    def hook_fn(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        modified = activation.clone()

        if target.component_type == ComponentType.ATTN_HEAD:
            if target.component_id is not None:
                if len(modified.shape) == 4:
                    if modified.shape[2] == hook.model.cfg.n_heads:
                        modified[:, :, target.component_id, :] *= factor
                    else:
                        modified[:, target.component_id, :, :] *= factor
            else:
                modified *= factor

        elif target.component_type == ComponentType.MLP_NEURON:
            if target.component_id is not None:
                modified[:, :, target.component_id] *= factor
            else:
                modified *= factor

        elif target.component_type == ComponentType.RESID:
            modified *= factor

        return modified

    return hook_fn


def create_injection_hook(
    target: InterventionTarget,
    source_activation: torch.Tensor,
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    """Create an injection hook that replaces activations.

    Args:
        target: Intervention target specification
        source_activation: Replacement activation tensor

    Returns:
        Hook function
    """

    def hook_fn(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        modified = activation.clone()

        # Validate shape compatibility
        if target.component_type == ComponentType.ATTN_HEAD:
            if target.component_id is not None:
                if len(modified.shape) == 4:
                    if modified.shape[2] == hook.model.cfg.n_heads:
                        expected_shape = modified[:, :, target.component_id, :].shape
                    else:
                        expected_shape = modified[:, target.component_id, :, :].shape
                    if source_activation.shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch: expected {expected_shape}, "
                            f"got {source_activation.shape}"
                        )
                    if modified.shape[2] == hook.model.cfg.n_heads:
                        modified[:, :, target.component_id, :] = source_activation
                    else:
                        modified[:, target.component_id, :, :] = source_activation
            else:
                if source_activation.shape != modified.shape:
                    raise ValueError(
                        f"Shape mismatch: expected {modified.shape}, "
                        f"got {source_activation.shape}"
                    )
                modified = source_activation

        elif target.component_type == ComponentType.MLP_NEURON:
            if target.component_id is not None:
                expected_shape = modified[:, :, target.component_id].shape
                if source_activation.shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch: expected {expected_shape}, "
                        f"got {source_activation.shape}"
                    )
                modified[:, :, target.component_id] = source_activation
            else:
                if source_activation.shape != modified.shape:
                    raise ValueError(
                        f"Shape mismatch: expected {modified.shape}, "
                        f"got {source_activation.shape}"
                    )
                modified = source_activation

        elif target.component_type == ComponentType.RESID:
            if source_activation.shape != modified.shape:
                raise ValueError(
                    f"Shape mismatch: expected {modified.shape}, "
                    f"got {source_activation.shape}"
                )
            modified = source_activation

        return modified

    return hook_fn


def extract_activations(
    model: HookedTransformer,
    input_text: str,
    layers: list[int] | None = None,
    components: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract activations from specified layers/components.

    Args:
        model: HookedTransformer model
        input_text: Input text to process
        layers: Layer indices to extract (None = all)
        components: Component types to extract (None = all standard)

    Returns:
        Dict mapping hook_point -> activation tensor
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    if components is None:
        components = ["resid_pre", "resid_post", "attn_out", "mlp_out"]

    # Build list of hook points to cache
    hook_points = []
    for layer in layers:
        for component in components:
            hook_point = HOOK_POINTS[component].format(layer=layer)
            hook_points.append(hook_point)

    # Run with cache
    _, cache = model.run_with_cache(input_text, names_filter=hook_points)

    return dict(cache)


def get_component_hook_point(
    layer: int,
    component_type: ComponentType,
) -> str:
    """Get the appropriate hook point for a component type.

    Args:
        layer: Layer index
        component_type: Type of component

    Returns:
        Hook point name
    """
    if component_type == ComponentType.ATTN_HEAD:
        return HOOK_POINTS["attn_out"].format(layer=layer)
    elif component_type == ComponentType.MLP_NEURON:
        return HOOK_POINTS["mlp_out"].format(layer=layer)
    elif component_type == ComponentType.RESID:
        return HOOK_POINTS["resid_post"].format(layer=layer)
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def validate_hook_points(
    model: HookedTransformer,
    hook_points: list[str],
) -> tuple[list[str], list[str]]:
    """Validate that hook points exist on the model.

    Args:
        model: HookedTransformer model
        hook_points: List of hook point names to validate

    Returns:
        Tuple of (valid_hooks, missing_hooks)
    """
    available = set(model.hook_dict.keys()) if hasattr(model, "hook_dict") else set()
    valid = [h for h in hook_points if h in available]
    missing = [h for h in hook_points if h not in available]
    if missing:
        logger.warning(f"Missing hook points: {missing}")
    return valid, missing


def compose_hooks(
    hooks: list[Callable[[torch.Tensor, HookPoint], torch.Tensor]],
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    """Compose multiple hooks into a single hook.

    Hooks are applied in order (first to last).

    Args:
        hooks: List of hook functions

    Returns:
        Composed hook function
    """

    def composed_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        result = activation
        for hook_fn in hooks:
            result = hook_fn(result, hook)
        return result

    return composed_hook
