"""Unit tests for MechLens intervention modules.

Tests intervention logic without requiring GPU or actual models.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from mechlens.types import (
    ComponentType,
    InterventionError,
    InterventionTarget,
)


class TestInterventionTargetValidation:
    """Test intervention target validation logic."""

    def test_valid_target_resid(self):
        """Residual targets don't need component_id."""
        t = InterventionTarget(layer=5, component_type=ComponentType.RESID)
        assert t.component_id is None

    def test_valid_target_attn_head(self):
        t = InterventionTarget(
            layer=3, component_type=ComponentType.ATTN_HEAD, component_id=7
        )
        assert t.component_id == 7

    def test_valid_target_mlp(self):
        t = InterventionTarget(
            layer=0, component_type=ComponentType.MLP_NEURON, component_id=42
        )
        assert t.component_id == 42


class TestScalingValidation:
    """Test scaling module validation without loading models."""

    def test_negative_factor_raises(self):
        """Scaling factor < 0 should raise InterventionError."""
        from mechlens.intervention.scaling import scale

        mock_model = MagicMock()
        targets = [InterventionTarget(layer=0, component_type=ComponentType.RESID)]

        with pytest.raises(InterventionError, match="must be >= 0.0"):
            scale(mock_model, "test", targets, factor=-1.0)

    def test_target_layer_out_of_range(self):
        """Layer index beyond model range should raise InterventionError."""
        from mechlens.intervention.scaling import scale

        mock_model = MagicMock()
        mock_model.cfg.n_layers = 24
        mock_model.cfg.n_heads = 14
        targets = [InterventionTarget(layer=30, component_type=ComponentType.RESID)]

        with pytest.raises(InterventionError, match="out of range"):
            scale(mock_model, "test", targets, factor=0.5)

    def test_target_head_out_of_range(self):
        """Head index beyond model range should raise InterventionError."""
        from mechlens.intervention.scaling import scale

        mock_model = MagicMock()
        mock_model.cfg.n_layers = 24
        mock_model.cfg.n_heads = 14
        targets = [
            InterventionTarget(
                layer=0, component_type=ComponentType.ATTN_HEAD, component_id=20
            )
        ]

        with pytest.raises(InterventionError, match="out of range"):
            scale(mock_model, "test", targets, factor=0.5)


class TestAblationValidation:
    """Test ablation module validation."""

    def test_target_layer_out_of_range(self):
        from mechlens.intervention.ablation import ablate

        mock_model = MagicMock()
        mock_model.cfg.n_layers = 24
        mock_model.cfg.n_heads = 14
        targets = [InterventionTarget(layer=50, component_type=ComponentType.RESID)]

        with pytest.raises(InterventionError, match="out of range"):
            ablate(mock_model, "test", targets)


class TestStrategyModule:
    """Test intervention strategy module."""

    def test_import_strategy(self):
        """Verify strategy module can be imported."""
        from mechlens.intervention.strategy import (
            save,
            load,
            list_strategies,
            compare,
        )
        # Just verify imports work

    def test_import_iti(self):
        """Verify ITI module can be imported."""
        from mechlens.intervention.iti import (
            learn_iti_directions,
            create_iti_steering_hook,
            generate_with_iti,
        )
        # Just verify imports work
