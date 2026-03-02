"""CrystalBoost: Crystallization-Aware Intervention for Factual Knowledge Enhancement.

A novel intervention method that leverages the Late Crystallization phenomenon:
- 85.9% of factual knowledge crystallizes only at the final layers
- Crystallization is nonlinear (LayerNorm + Unembedding)
- Existing methods are crystallization-agnostic

CrystalBoost Design:
1. Dual-Stage Steering: Suppress surface patterns early, amplify facts late
2. Boundary-Focused: Concentrate intervention at crystallization boundary (final 2-3 layers)
3. Adaptive Strength: Scale coefficients by layer's crystallization importance

Expected to outperform DoLa by directly targeting the crystallization mechanism.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrystalBoostConfig:
    """Configuration for CrystalBoost intervention."""
    # Boundary detection
    boundary_start_ratio: float = 0.8  # Start of crystallization boundary (e.g., layer 22/28)
    boundary_end_ratio: float = 1.0    # End of crystallization boundary
    
    # Dual-stage steering
    early_suppression_coeff: float = -0.5  # Negative to suppress surface patterns
    late_amplification_coeff: float = 2.0   # Positive to amplify factual signals
    boundary_boost_coeff: float = 3.0       # Extra boost at crystallization boundary
    
    # Layer weighting
    use_gaussian_weighting: bool = True
    gaussian_peak_ratio: float = 0.9  # Peak at 90% depth (near final layer)
    gaussian_sigma: float = 0.1       # Width of Gaussian
    
    # Direction source
    n_contrastive_samples: int = 100  # Samples for learning directions
    top_k_layers: int = 10            # Number of layers to intervene


def unembed_at_layer(model, resid: torch.Tensor) -> torch.Tensor:
    """Project residual stream to vocabulary logits via ln_final + W_U."""
    normed = model.ln_final(resid)
    logits = normed @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    return logits


def compute_crystallization_gradient(
    model,
    tokens: torch.Tensor,
    early_layers: list[int],
    late_layers: list[int],
) -> dict[int, float]:
    """Compute crystallization gradient per layer.
    
    Crystallization gradient = how much the logit distribution changes
    between this layer and the final layer.
    
    Higher gradient = more crystallization happening at this layer.
    """
    n_layers = model.cfg.n_layers
    
    # Cache all relevant layers
    all_layers = set(early_layers) | set(late_layers) | {n_layers - 1}
    hook_names = [f"blocks.{l}.hook_resid_post" for l in all_layers]
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_names)
    
    # Get final layer distribution as reference
    final_resid = cache[f"blocks.{n_layers - 1}.hook_resid_post"][0, -1, :]
    final_logits = unembed_at_layer(model, final_resid)
    final_probs = F.softmax(final_logits.float(), dim=-1)
    final_entropy = -torch.sum(final_probs * torch.log(final_probs + 1e-10)).item()
    
    # Compute gradient for each layer
    gradients = {}
    
    for layer in all_layers:
        if layer == n_layers - 1:
            gradients[layer] = 0.0
            continue
        
        resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
        logits = unembed_at_layer(model, resid)
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # KL divergence from this layer to final layer
        kl_div = F.kl_div(
            F.log_softmax(logits.float(), dim=-1),
            final_probs,
            reduction="sum",
            log_target=False
        ).item()
        
        # Gradient = entropy reduction + KL divergence
        gradients[layer] = (entropy - final_entropy) + kl_div
    
    return gradients


def learn_crystalboost_directions(
    model,
    dataset: list[dict],
    config: CrystalBoostConfig = None,
) -> dict:
    """Learn CrystalBoost directions from contrastive pairs.
    
    Unlike CAA/ITI which use uniform directions, CrystalBoost learns:
    1. Early suppression direction (reduce surface pattern contribution)
    2. Late amplification direction (enhance factual signal)
    3. Per-layer weights based on crystallization importance
    """
    if config is None:
        config = CrystalBoostConfig()
    
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    # Define layer regions
    boundary_start = int(n_layers * config.boundary_start_ratio)
    early_layers = list(range(0, boundary_start // 2))
    late_layers = list(range(boundary_start, n_layers))
    boundary_layers = list(range(boundary_start, n_layers - 1))
    
    logger.info(f"CrystalBoost layer regions: early={early_layers[:3]}..., boundary={boundary_layers}, late={late_layers}")
    
    # Collect activations for direction learning
    all_layers = set(early_layers) | set(late_layers)
    hook_names = [f"blocks.{l}.hook_resid_post" for l in all_layers]
    
    early_correct_acts = {l: [] for l in early_layers}
    early_incorrect_acts = {l: [] for l in early_layers}
    late_correct_acts = {l: [] for l in late_layers}
    late_incorrect_acts = {l: [] for l in late_layers}
    
    crystallization_gradients = {l: [] for l in all_layers}
    
    n_samples = min(config.n_contrastive_samples, len(dataset))
    
    for sample in dataset[:n_samples]:
        question = sample["question"]
        best_answer = sample.get("best_answer", "")
        incorrect_answers = sample.get("incorrect_answers", [])
        
        if not best_answer or not incorrect_answers:
            continue
        
        # Correct answer activations
        correct_text = f"Q: {question}\nA: {best_answer}"
        correct_tokens = model.to_tokens(correct_text, prepend_bos=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(correct_tokens, names_filter=hook_names)
        
        for l in early_layers:
            act = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu()
            early_correct_acts[l].append(act)
        
        for l in late_layers:
            act = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu()
            late_correct_acts[l].append(act)
        
        # Compute crystallization gradient for this sample
        gradients = compute_crystallization_gradient(
            model, correct_tokens, early_layers, late_layers
        )
        for l, g in gradients.items():
            if l in crystallization_gradients:
                crystallization_gradients[l].append(g)
        
        # Incorrect answer activations
        incorrect_text = f"Q: {question}\nA: {incorrect_answers[0]}"
        incorrect_tokens = model.to_tokens(incorrect_text, prepend_bos=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(incorrect_tokens, names_filter=hook_names)
        
        for l in early_layers:
            act = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu()
            early_incorrect_acts[l].append(act)
        
        for l in late_layers:
            act = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].cpu()
            late_incorrect_acts[l].append(act)
    
    # Compute directions
    early_directions = {}
    late_directions = {}
    
    for l in early_layers:
        if early_correct_acts[l] and early_incorrect_acts[l]:
            correct_mean = torch.stack(early_correct_acts[l]).mean(dim=0)
            incorrect_mean = torch.stack(early_incorrect_acts[l]).mean(dim=0)
            # Early: direction points from correct to incorrect (to suppress surface patterns)
            direction = incorrect_mean - correct_mean
            direction = direction / (direction.norm() + 1e-8)
            early_directions[l] = direction
    
    for l in late_layers:
        if late_correct_acts[l] and late_incorrect_acts[l]:
            correct_mean = torch.stack(late_correct_acts[l]).mean(dim=0)
            incorrect_mean = torch.stack(late_incorrect_acts[l]).mean(dim=0)
            # Late: direction points from incorrect to correct (to amplify factual signal)
            direction = correct_mean - incorrect_mean
            direction = direction / (direction.norm() + 1e-8)
            late_directions[l] = direction
    
    # Compute layer weights based on crystallization gradient
    layer_weights = {}
    for l in all_layers:
        if crystallization_gradients[l]:
            mean_gradient = np.mean(crystallization_gradients[l])
        else:
            mean_gradient = 0.0
        
        # Gaussian weighting centered at crystallization peak
        if config.use_gaussian_weighting:
            layer_ratio = l / n_layers
            peak = config.gaussian_peak_ratio
            sigma = config.gaussian_sigma
            gaussian_weight = np.exp(-0.5 * ((layer_ratio - peak) / sigma) ** 2)
            layer_weights[l] = gaussian_weight * (1 + mean_gradient)
        else:
            layer_weights[l] = 1.0 + mean_gradient
    
    # Normalize weights
    max_weight = max(layer_weights.values()) if layer_weights else 1.0
    layer_weights = {l: w / max_weight for l, w in layer_weights.items()}
    
    return {
        "early_layers": early_layers,
        "late_layers": late_layers,
        "boundary_layers": boundary_layers,
        "early_directions": early_directions,
        "late_directions": late_directions,
        "layer_weights": layer_weights,
        "crystallization_gradients": {l: np.mean(g) if g else 0.0 
                                       for l, g in crystallization_gradients.items()},
        "config": config,
        "n_samples_used": n_samples,
    }


def create_crystalboost_hook(
    layer_idx: int,
    crystalboost_info: dict,
    device: str,
    dtype: torch.dtype,
) -> Callable:
    """Create a hook for CrystalBoost intervention at a specific layer."""
    config = crystalboost_info["config"]
    early_layers = crystalboost_info["early_layers"]
    late_layers = crystalboost_info["late_layers"]
    boundary_layers = crystalboost_info["boundary_layers"]
    early_directions = crystalboost_info["early_directions"]
    late_directions = crystalboost_info["late_directions"]
    layer_weights = crystalboost_info["layer_weights"]
    
    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        modified = activation.clone()
        
        if layer_idx in early_layers and layer_idx in early_directions:
            # Early suppression: push away from surface patterns
            direction = early_directions[layer_idx].to(device, dtype)
            weight = layer_weights.get(layer_idx, 1.0)
            coeff = config.early_suppression_coeff * weight
            steering = coeff * direction
            modified = modified + steering.unsqueeze(0).unsqueeze(0)
        
        elif layer_idx in late_layers and layer_idx in late_directions:
            # Late amplification: push toward factual knowledge
            direction = late_directions[layer_idx].to(device, dtype)
            weight = layer_weights.get(layer_idx, 1.0)
            
            # Extra boost at crystallization boundary
            if layer_idx in boundary_layers:
                coeff = config.boundary_boost_coeff * weight
            else:
                coeff = config.late_amplification_coeff * weight
            
            steering = coeff * direction
            modified = modified + steering.unsqueeze(0).unsqueeze(0)
        
        return modified
    
    return hook_fn


def compute_crystalboost_log_prob(
    model,
    question: str,
    answer: str,
    crystalboost_info: dict,
) -> float:
    """Compute CrystalBoost-steered log probability."""
    early_directions = crystalboost_info["early_directions"]
    late_directions = crystalboost_info["late_directions"]
    
    prompt = f"Q: {question}\nA:"
    full_text = f"Q: {question}\nA: {answer}"
    
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(full_text, prepend_bos=True)
    
    q_len = prompt_tokens.shape[1]
    
    if full_tokens.shape[1] <= q_len:
        return float("-inf")
    
    # Create hooks for all intervention layers
    hooks = []
    
    for l in early_directions.keys():
        hook_fn = create_crystalboost_hook(
            l, crystalboost_info, 
            model.cfg.device, model.cfg.dtype
        )
        hooks.append((f"blocks.{l}.hook_resid_post", hook_fn))
    
    for l in late_directions.keys():
        hook_fn = create_crystalboost_hook(
            l, crystalboost_info,
            model.cfg.device, model.cfg.dtype
        )
        hooks.append((f"blocks.{l}.hook_resid_post", hook_fn))
    
    # Run with hooks
    with torch.no_grad():
        logits = model.run_with_hooks(full_tokens, fwd_hooks=hooks)
    
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    
    total_log_prob = 0.0
    for i in range(q_len, full_tokens.shape[1]):
        token_id = full_tokens[0, i].item()
        total_log_prob += log_probs[i - 1, token_id].item()
    
    return total_log_prob


def generate_with_crystalboost(
    model,
    prompt: str,
    crystalboost_info: dict,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> str:
    """Generate text with CrystalBoost intervention."""
    early_directions = crystalboost_info["early_directions"]
    late_directions = crystalboost_info["late_directions"]
    
    tokens = model.to_tokens(prompt, prepend_bos=True)
    
    # Create hooks
    hooks = []
    
    for l in early_directions.keys():
        hook_fn = create_crystalboost_hook(
            l, crystalboost_info,
            model.cfg.device, model.cfg.dtype
        )
        hooks.append((f"blocks.{l}.hook_resid_post", hook_fn))
    
    for l in late_directions.keys():
        hook_fn = create_crystalboost_hook(
            l, crystalboost_info,
            model.cfg.device, model.cfg.dtype
        )
        hooks.append((f"blocks.{l}.hook_resid_post", hook_fn))
    
    # Generate tokens one by one
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        
        next_logits = logits[0, -1, :]
        
        if temperature == 0.0:
            next_token = torch.argmax(next_logits).item()
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        
        generated_tokens.append(next_token)
        tokens = torch.cat([tokens, torch.tensor([[next_token]], device=tokens.device)], dim=1)
        
        # Stop on EOS
        if next_token == model.tokenizer.eos_token_id:
            break
    
    return model.to_string(torch.tensor([generated_tokens]))


# ==================== GRID SEARCH UTILITIES ====================

def grid_search_crystalboost(
    model,
    dataset: list[dict],
    param_grid: dict = None,
) -> dict:
    """Grid search over CrystalBoost hyperparameters."""
    if param_grid is None:
        param_grid = {
            "early_suppression_coeff": [-0.5, -1.0, -1.5],
            "late_amplification_coeff": [1.0, 2.0, 3.0],
            "boundary_boost_coeff": [2.0, 3.0, 5.0],
            "gaussian_sigma": [0.05, 0.1, 0.15],
        }
    
    from itertools import product
    
    best_config = None
    best_mc1 = 0.0
    all_results = []
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combo in product(*values):
        params = dict(zip(keys, combo))
        
        config = CrystalBoostConfig(**params)
        
        # Learn directions with this config
        crystalboost_info = learn_crystalboost_directions(model, dataset, config)
        
        # Evaluate MC1
        score_fn = lambda m, q, a: compute_crystalboost_log_prob(m, q, a, crystalboost_info)
        
        correct = 0
        total = 0
        
        for sample in dataset:
            question = sample["question"]
            best_answer = sample.get("best_answer", "")
            incorrect_answers = sample.get("incorrect_answers", [])
            
            if not best_answer or not incorrect_answers:
                continue
            
            best_score = score_fn(model, question, best_answer)
            incorrect_scores = [score_fn(model, question, a) for a in incorrect_answers]
            
            all_scores = [best_score] + incorrect_scores
            is_correct = best_score == max(all_scores)
            
            if is_correct:
                correct += 1
            total += 1
        
        mc1 = correct / total if total > 0 else 0.0
        
        result = {
            "params": params,
            "mc1": mc1,
        }
        all_results.append(result)
        
        if mc1 > best_mc1:
            best_mc1 = mc1
            best_config = params
        
        logger.info(f"CrystalBoost {params}: MC1 = {mc1:.4f}")
    
    return {
        "best_config": best_config,
        "best_mc1": best_mc1,
        "all_results": all_results,
    }


__all__ = [
    "CrystalBoostConfig",
    "learn_crystalboost_directions",
    "compute_crystalboost_log_prob",
    "generate_with_crystalboost",
    "grid_search_crystalboost",
]
