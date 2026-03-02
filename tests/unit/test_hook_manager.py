"""Unit tests for MechLens hook_manager module.

Tests hook point naming, hook creation, and validation without requiring GPU.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from mechlens.models.hook_manager import (
    HOOK_POINTS,
    HookManager,
    HookSpec,
    compose_hooks,
    create_ablation_hook,
    create_scaling_hook,
    get_component_hook_point,
    validate_hook_points,
)
from mechlens.types import ComponentType, InterventionTarget


class TestHookPoints:
    """Test hook point naming convention."""

    def test_all_hook_points_defined(self):
        expected = [
            "resid_pre", "resid_post", "resid_mid",
            "attn_out", "attn_pattern", "attn_scores", "attn_z",
            "mlp_out", "mlp_pre", "mlp_post",
        ]
        for key in expected:
            assert key in HOOK_POINTS, f"Missing hook point: {key}"

    def test_hook_point_format(self):
        for key, template in HOOK_POINTS.items():
            assert "{layer}" in template, f"Hook point '{key}' missing {{layer}} placeholder"
            # Verify formatting works
            formatted = template.format(layer=5)
            assert "5" in formatted
            assert "{" not in formatted

    def test_specific_hook_names(self):
        assert HOOK_POINTS["resid_post"].format(layer=0) == "blocks.0.hook_resid_post"
        assert HOOK_POINTS["mlp_out"].format(layer=3) == "blocks.3.hook_mlp_out"
        assert HOOK_POINTS["attn_out"].format(layer=7) == "blocks.7.attn.hook_result"
        assert HOOK_POINTS["attn_z"].format(layer=12) == "blocks.12.attn.hook_z"


class TestGetComponentHookPoint:
    """Test get_component_hook_point function."""

    def test_attn_head(self):
        hp = get_component_hook_point(5, ComponentType.ATTN_HEAD)
        assert hp == "blocks.5.attn.hook_result"

    def test_mlp_neuron(self):
        hp = get_component_hook_point(10, ComponentType.MLP_NEURON)
        assert hp == "blocks.10.hook_mlp_out"

    def test_resid(self):
        hp = get_component_hook_point(0, ComponentType.RESID)
        assert hp == "blocks.0.hook_resid_post"

    def test_invalid_component(self):
        with pytest.raises(ValueError, match="Unknown component type"):
            get_component_hook_point(0, "invalid")


class TestCreateScalingHook:
    """Test scaling hook creation."""

    def test_negative_factor_raises(self):
        target = InterventionTarget(layer=0, component_type=ComponentType.RESID)
        with pytest.raises(ValueError, match="must be >= 0.0"):
            create_scaling_hook(target, factor=-1.0)

    def test_resid_scaling(self):
        target = InterventionTarget(layer=0, component_type=ComponentType.RESID)
        hook_fn = create_scaling_hook(target, factor=2.0)
        # Simulate activation
        activation = torch.ones(1, 5, 128)
        mock_hook = MagicMock()
        result = hook_fn(activation, mock_hook)
        assert torch.allclose(result, torch.ones(1, 5, 128) * 2.0)

    def test_mlp_scaling_all(self):
        target = InterventionTarget(layer=0, component_type=ComponentType.MLP_NEURON)
        hook_fn = create_scaling_hook(target, factor=0.5)
        activation = torch.ones(1, 5, 128)
        mock_hook = MagicMock()
        result = hook_fn(activation, mock_hook)
        assert torch.allclose(result, torch.ones(1, 5, 128) * 0.5)

    def test_identity_scaling(self):
        target = InterventionTarget(layer=0, component_type=ComponentType.RESID)
        hook_fn = create_scaling_hook(target, factor=1.0)
        activation = torch.randn(1, 5, 128)
        mock_hook = MagicMock()
        result = hook_fn(activation, mock_hook)
        assert torch.allclose(result, activation)


class TestCreateAblationHook:
    """Test ablation hook creation."""

    def test_resid_ablation(self):
        target = InterventionTarget(layer=0, component_type=ComponentType.RESID)
        hook_fn = create_ablation_hook(target)
        activation = torch.randn(1, 5, 128)
        mock_hook = MagicMock()
        result = hook_fn(activation, mock_hook)
        assert torch.allclose(result, torch.zeros_like(activation))

    def test_mlp_ablation_all(self):
        target = InterventionTarget(layer=0, component_type=ComponentType.MLP_NEURON)
        hook_fn = create_ablation_hook(target)
        activation = torch.randn(1, 5, 128)
        mock_hook = MagicMock()
        result = hook_fn(activation, mock_hook)
        assert torch.allclose(result, torch.zeros_like(activation))


class TestValidateHookPoints:
    """Test validate_hook_points function."""

    def test_all_valid(self):
        model = MagicMock()
        model.hook_dict = {
            "blocks.0.hook_resid_post": MagicMock(),
            "blocks.0.hook_mlp_out": MagicMock(),
            "blocks.0.attn.hook_result": MagicMock(),
        }
        valid, missing = validate_hook_points(
            model,
            ["blocks.0.hook_resid_post", "blocks.0.hook_mlp_out"],
        )
        assert len(valid) == 2
        assert len(missing) == 0

    def test_some_missing(self):
        model = MagicMock()
        model.hook_dict = {
            "blocks.0.hook_resid_post": MagicMock(),
        }
        valid, missing = validate_hook_points(
            model,
            ["blocks.0.hook_resid_post", "blocks.0.hook_mlp_out"],
        )
        assert valid == ["blocks.0.hook_resid_post"]
        assert missing == ["blocks.0.hook_mlp_out"]

    def test_no_hook_dict(self):
        model = MagicMock(spec=[])  # No hook_dict attribute
        valid, missing = validate_hook_points(
            model,
            ["blocks.0.hook_resid_post"],
        )
        assert len(valid) == 0
        assert len(missing) == 1


class TestHookManager:
    """Test HookManager class."""

    def _make_manager(self):
        model = MagicMock()
        model.cfg.n_layers = 24
        model.cfg.n_heads = 14
        return HookManager(model=model)

    def test_get_hook_point(self):
        mgr = self._make_manager()
        assert mgr.get_hook_point(0, "mlp_out") == "blocks.0.hook_mlp_out"
        assert mgr.get_hook_point(5, "attn_out") == "blocks.5.attn.hook_result"

    def test_get_hook_point_invalid(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="Unknown component"):
            mgr.get_hook_point(0, "nonexistent")

    def test_register_and_get_hooks(self):
        mgr = self._make_manager()
        fn = lambda act, hook: act
        mgr.register_hook("test_hook", layer=3, component="mlp_out", hook_fn=fn)
        active = mgr.get_active_hooks()
        assert len(active) == 1
        assert active[0][0] == "blocks.3.hook_mlp_out"

    def test_remove_hook(self):
        mgr = self._make_manager()
        fn = lambda act, hook: act
        mgr.register_hook("test_hook", layer=0, component="mlp_out", hook_fn=fn)
        mgr.remove_hook("test_hook")
        assert len(mgr.get_active_hooks()) == 0

    def test_clear_hooks(self):
        mgr = self._make_manager()
        fn = lambda act, hook: act
        mgr.register_hook("h1", layer=0, component="mlp_out", hook_fn=fn)
        mgr.register_hook("h2", layer=1, component="attn_out", hook_fn=fn)
        mgr.clear_hooks()
        assert len(mgr.get_active_hooks()) == 0


class TestComposeHooks:
    """Test hook composition."""

    def test_compose_two_hooks(self):
        def double(act, hook):
            return act * 2

        def add_one(act, hook):
            return act + 1

        composed = compose_hooks([double, add_one])
        activation = torch.tensor([1.0, 2.0, 3.0])
        mock_hook = MagicMock()
        result = composed(activation, mock_hook)
        # double first: [2, 4, 6], then add_one: [3, 5, 7]
        expected = torch.tensor([3.0, 5.0, 7.0])
        assert torch.allclose(result, expected)

    def test_compose_empty(self):
        composed = compose_hooks([])
        activation = torch.tensor([1.0, 2.0])
        mock_hook = MagicMock()
        result = composed(activation, mock_hook)
        assert torch.allclose(result, activation)
