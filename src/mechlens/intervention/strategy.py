"""MechLens intervention strategy management.

Save, load, and compare intervention strategies.
Per contract section 8.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mechlens.types import ComparisonResult, ComponentType, InterventionTarget, InterventionType

logger = logging.getLogger(__name__)

# Default strategies directory
STRATEGIES_DIR = Path("strategies")


def save(
    name: str,
    intervention_type: str | InterventionType,
    targets: list[InterventionTarget],
    params: dict[str, Any],
    results_summary: dict[str, Any] | None = None,
    strategies_dir: Path | str | None = None,
) -> str:
    """Save an intervention strategy for later reuse/comparison.

    Args:
        name: Human-readable strategy name (e.g., "ablate_L5_H3_hallucination")
        intervention_type: "ablation" | "scaling" | "injection"
        targets: Target specification
        params: Type-specific parameters (e.g., {"factor": 0.5} for scaling)
        results_summary: Optional summary metrics from prior execution
        strategies_dir: Directory for strategy files (default: strategies/)

    Returns:
        Strategy ID (UUID string)

    Storage:
        JSON file at strategies/{name}.json
    """
    if strategies_dir is None:
        strategies_dir = STRATEGIES_DIR
    strategies_dir = Path(strategies_dir)
    strategies_dir.mkdir(parents=True, exist_ok=True)

    strategy_id = str(uuid.uuid4())

    if isinstance(intervention_type, InterventionType):
        intervention_type = intervention_type.value

    strategy = {
        "id": strategy_id,
        "name": name,
        "intervention_type": intervention_type,
        "targets": _serialize_targets(targets),
        "params": params,
        "results_summary": results_summary,
        "created_at": datetime.now().isoformat(),
    }

    filepath = strategies_dir / f"{name}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(strategy, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved strategy '{name}' with ID {strategy_id}")
    return strategy_id


def load(
    strategy_id_or_name: str,
    strategies_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load a saved strategy.

    Args:
        strategy_id_or_name: Strategy ID (UUID) or name
        strategies_dir: Directory for strategy files

    Returns:
        Strategy dict with id, name, intervention_type, targets, params, results_summary
    """
    if strategies_dir is None:
        strategies_dir = STRATEGIES_DIR
    strategies_dir = Path(strategies_dir)

    # Try loading by name first
    filepath = strategies_dir / f"{strategy_id_or_name}.json"
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    # Search by ID
    for path in strategies_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            strategy = json.load(f)
            if strategy.get("id") == strategy_id_or_name:
                return strategy

    raise FileNotFoundError(f"Strategy not found: {strategy_id_or_name}")


def list_strategies(
    strategies_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List all saved strategies.

    Args:
        strategies_dir: Directory for strategy files

    Returns:
        List of strategy summary dicts
    """
    if strategies_dir is None:
        strategies_dir = STRATEGIES_DIR
    strategies_dir = Path(strategies_dir)

    if not strategies_dir.exists():
        return []

    strategies = []
    for path in strategies_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            strategy = json.load(f)
            strategies.append({
                "id": strategy["id"],
                "name": strategy["name"],
                "intervention_type": strategy["intervention_type"],
                "n_targets": len(strategy.get("targets", [])),
                "created_at": strategy.get("created_at"),
            })

    return sorted(strategies, key=lambda s: s.get("created_at", ""), reverse=True)


def delete(
    strategy_id_or_name: str,
    strategies_dir: Path | str | None = None,
) -> bool:
    """Delete a saved strategy.

    Args:
        strategy_id_or_name: Strategy ID or name
        strategies_dir: Directory for strategy files

    Returns:
        True if deleted, False if not found
    """
    if strategies_dir is None:
        strategies_dir = STRATEGIES_DIR
    strategies_dir = Path(strategies_dir)

    # Try by name first
    filepath = strategies_dir / f"{strategy_id_or_name}.json"
    if filepath.exists():
        filepath.unlink()
        logger.info(f"Deleted strategy: {strategy_id_or_name}")
        return True

    # Search by ID
    for path in strategies_dir.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            strategy = json.load(f)
            if strategy.get("id") == strategy_id_or_name:
                path.unlink()
                logger.info(f"Deleted strategy: {strategy.get('name')}")
                return True

    return False


def compare(
    strategy_ids: list[str],
    samples: list[dict[str, str]],
    model_runner: callable,
    strategies_dir: Path | str | None = None,
) -> ComparisonResult:
    """Compare multiple strategies on same samples.

    Args:
        strategy_ids: List of strategy IDs or names to compare
        samples: List of {"input_text": str, "expected": str}
        model_runner: Function (strategy, samples) -> list[dict] with per-sample metrics
        strategies_dir: Directory for strategy files

    Returns:
        ComparisonResult with rankings
    """
    strategies = []
    per_strategy_metrics = []

    for sid in strategy_ids:
        strategy = load(sid, strategies_dir)
        strategies.append(strategy)

        # Run intervention with this strategy
        metrics = model_runner(strategy, samples)
        per_strategy_metrics.append(metrics)

    # Compute ranking based on average KL divergence (lower is better for suppression)
    rankings = []
    for i, (strategy, metrics) in enumerate(zip(strategies, per_strategy_metrics)):
        avg_kl = sum(m.get("kl_divergence", 0) for m in metrics) / len(metrics) if metrics else 0
        rankings.append((strategy["id"], avg_kl))

    rankings.sort(key=lambda x: x[1])
    ranked_ids = [r[0] for r in rankings]

    # Build diff table
    diff_table = _build_diff_table(strategies, per_strategy_metrics)

    return ComparisonResult(
        strategies=strategies,
        per_strategy_metrics=per_strategy_metrics,
        ranking=ranked_ids,
        diff_table=diff_table,
    )


def _serialize_targets(targets: list[InterventionTarget]) -> list[dict]:
    """Serialize intervention targets to JSON-compatible format."""
    serialized = []
    for target in targets:
        serialized.append({
            "layer": target.layer,
            "component_type": target.component_type.value,
            "component_id": target.component_id,
            "factor": target.factor,
            # Note: source_activation tensors cannot be serialized to JSON
        })
    return serialized


def deserialize_targets(target_dicts: list[dict]) -> list[InterventionTarget]:
    """Deserialize intervention targets from JSON format."""
    targets = []
    for td in target_dicts:
        targets.append(InterventionTarget(
            layer=td["layer"],
            component_type=ComponentType(td["component_type"]),
            component_id=td.get("component_id"),
            factor=td.get("factor"),
        ))
    return targets


def _build_diff_table(
    strategies: list[dict],
    per_strategy_metrics: list[list[dict]],
) -> dict[str, Any]:
    """Build comparison diff table."""
    table = {
        "strategy_names": [s["name"] for s in strategies],
        "metrics": {},
    }

    # Aggregate common metrics
    metric_keys = ["kl_divergence", "logit_diff", "prob_change"]

    for key in metric_keys:
        table["metrics"][key] = []
        for metrics in per_strategy_metrics:
            values = [m.get(key, 0) for m in metrics if key in m]
            avg = sum(values) / len(values) if values else 0
            std = (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5 if values else 0
            table["metrics"][key].append({"mean": avg, "std": std})

    return table
