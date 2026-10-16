"""
Case Study 2: In-Context Learning (ICL) Circuit Discovery

Pipeline (Qwen-only per R9):
1. Load Qwen model
2. Create ICL prompts (few-shot examples)
3. Identify induction heads via attention pattern analysis
4. Run circuit discovery to find ICL-critical components
5. Ablation test to verify circuit importance
6. Analyze copying mechanism
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np

from mechlens.config import SUPPORTED_MODELS
from mechlens.models.model_loader import load_model
from mechlens.types import InterventionTarget, ComponentType

# Analysis
from mechlens.analysis.attention import (
    analyze as analyze_attention,
    find_induction_heads,
    compute_attention_entropy
)
from mechlens.analysis.circuit import discover as discover_circuit
from mechlens.analysis.activation import analyze as analyze_activation

# Intervention
from mechlens.intervention.ablation import ablate
from mechlens.intervention.scaling import scale

logger = logging.getLogger(__name__)


@dataclass
class ICLStudyConfig:
    """Configuration for ICL circuit study."""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    dtype: str = "float16"
    
    # ICL task
    task_type: str = "translation"  # translation, sentiment, arithmetic
    num_shots: int = 4
    
    # Induction head detection
    induction_threshold: float = 0.3
    
    # Circuit discovery
    circuit_method: str = "activation_patching"
    circuit_threshold: float = 0.1
    
    # Ablation
    ablate_induction_heads: bool = True
    ablate_top_k: int = 5
    
    # Output
    output_dir: str = "results/icl_study"


@dataclass
class ICLStudyResult:
    """Results from ICL circuit study."""
    model_name: str
    config: ICLStudyConfig
    
    # Induction heads
    induction_heads: List[Tuple[int, int]] = field(default_factory=list)
    induction_scores: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # Circuit structure
    circuit_nodes: List[Dict[str, Any]] = field(default_factory=list)
    circuit_edges: List[Dict[str, Any]] = field(default_factory=list)
    circuit_faithfulness: float = 0.0
    circuit_completeness: float = 0.0
    
    # Ablation results
    baseline_accuracy: float = 0.0
    ablation_accuracy: float = 0.0
    accuracy_drop: float = 0.0
    
    # Per-head ablation
    per_head_ablation: Dict[str, float] = field(default_factory=dict)
    
    # Copying analysis
    copying_score: float = 0.0
    attention_to_context: Dict[int, float] = field(default_factory=dict)


def _validate_qwen_model(model_name: str):
    """Validate that model is Qwen (per R9)."""
    if "qwen" not in model_name.lower():
        raise ValueError(
            f"ICL circuit study only supports Qwen models (R9). "
            f"Got: {model_name}"
        )


def create_icl_prompts(
    task_type: str = "translation",
    num_shots: int = 4
) -> List[Dict[str, str]]:
    """Create ICL prompts for different tasks."""
    
    if task_type == "translation":
        # Chinese-English translation task
        examples = [
            {"input": "猫", "output": "cat"},
            {"input": "狗", "output": "dog"},
            {"input": "鸟", "output": "bird"},
            {"input": "鱼", "output": "fish"},
            {"input": "马", "output": "horse"},
            {"input": "羊", "output": "sheep"},
        ]
        template = "{input} -> {output}"
        test_input = "牛"
        expected_output = "cow"
        
    elif task_type == "sentiment":
        # Sentiment classification
        examples = [
            {"input": "这个电影太棒了！", "output": "positive"},
            {"input": "服务态度很差。", "output": "negative"},
            {"input": "非常满意这次购物体验！", "output": "positive"},
            {"input": "质量太差了，失望。", "output": "negative"},
            {"input": "推荐给大家！", "output": "positive"},
            {"input": "不会再来了。", "output": "negative"},
        ]
        template = "评论: {input}\n情感: {output}"
        test_input = "产品很好用，下次还买！"
        expected_output = "positive"
        
    elif task_type == "arithmetic":
        # Simple arithmetic
        examples = [
            {"input": "2 + 3", "output": "5"},
            {"input": "7 - 4", "output": "3"},
            {"input": "5 + 1", "output": "6"},
            {"input": "9 - 2", "output": "7"},
            {"input": "3 + 4", "output": "7"},
            {"input": "8 - 5", "output": "3"},
        ]
        template = "{input} = {output}"
        test_input = "6 + 2"
        expected_output = "8"
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Select shots
    selected_examples = examples[:num_shots]
    
    # Build prompt
    prompt_parts = []
    for ex in selected_examples:
        prompt_parts.append(template.format(**ex))
    
    # Add test input
    test_prompt = template.format(input=test_input, output="")
    test_prompt = test_prompt.rstrip()  # Remove trailing space/output
    prompt_parts.append(test_prompt)
    
    full_prompt = "\n".join(prompt_parts)
    
    return [{
        "prompt": full_prompt,
        "test_input": test_input,
        "expected_output": expected_output,
        "task_type": task_type,
        "num_shots": num_shots
    }]


def run_icl_study(config: ICLStudyConfig) -> ICLStudyResult:
    """
    Run the complete ICL circuit study.
    
    Pipeline:
    1. Validate model (Qwen only)
    2. Create ICL prompts
    3. Identify induction heads
    4. Discover circuit
    5. Run ablation tests
    """
    # Validate Qwen model
    _validate_qwen_model(config.model_name)
    
    logger.info(f"Starting ICL study with model {config.model_name}")
    
    result = ICLStudyResult(
        model_name=config.model_name,
        config=config
    )
    
    # Step 1: Load model
    logger.info("Loading model...")
    model = load_model(config.model_name, dtype=config.dtype)
    
    # Step 2: Create ICL prompts
    logger.info(f"Creating {config.task_type} ICL prompts...")
    prompts = create_icl_prompts(
        task_type=config.task_type,
        num_shots=config.num_shots
    )
    prompt_data = prompts[0]
    prompt = prompt_data["prompt"]
    expected = prompt_data["expected_output"]
    
    # Step 3: Identify induction heads
    logger.info("Analyzing attention patterns for induction heads...")
    attn_data = analyze_attention(model, prompt)
    induction_heads = find_induction_heads(
        attn_data,
        threshold=config.induction_threshold
    )
    
    result.induction_heads = induction_heads
    result.induction_scores = {
        (h["layer"], h["head"]): h["score"]
        for h in induction_heads
    }
    
    logger.info(f"Found {len(induction_heads)} induction heads")
    for h in induction_heads[:5]:
        logger.info(f"  Layer {h['layer']}, Head {h['head']}: score={h['score']:.3f}")
    
    # Step 4: Circuit discovery
    logger.info("Discovering ICL circuit...")
    circuit = discover_circuit(
        model=model,
        input_text=prompt,
        target_token_idx=-1,
        method=config.circuit_method,
        threshold=config.circuit_threshold
    )
    
    result.circuit_nodes = [
        {"layer": n.layer, "component": n.component_type.value, "idx": n.component_idx, "importance": n.importance}
        for n in circuit.nodes
    ]
    result.circuit_edges = [
        {"source": (e.source.layer, e.source.component_type.value),
         "target": (e.target.layer, e.target.component_type.value),
         "weight": e.weight}
        for e in circuit.edges
    ]
    result.circuit_faithfulness = circuit.faithfulness
    result.circuit_completeness = circuit.completeness
    
    logger.info(f"Circuit: {len(circuit.nodes)} nodes, {len(circuit.edges)} edges")
    logger.info(f"Faithfulness: {circuit.faithfulness:.3f}, Completeness: {circuit.completeness:.3f}")
    
    # Step 5: Baseline accuracy
    logger.info("Computing baseline accuracy...")
    baseline_correct, baseline_output = _evaluate_icl(model, prompt, expected)
    result.baseline_accuracy = 1.0 if baseline_correct else 0.0
    logger.info(f"Baseline output: '{baseline_output}', correct: {baseline_correct}")
    
    # Step 6: Ablation tests
    if config.ablate_induction_heads and induction_heads:
        logger.info("Running ablation tests on induction heads...")
        
        # Ablate top-k induction heads
        ablation_targets = []
        for h in induction_heads[:config.ablate_top_k]:
            target = InterventionTarget(
                layer=h["layer"],
                component_type=ComponentType.ATTN_HEAD,
                component_idx=h["head"]
            )
            ablation_targets.append(target)
        
        # Run ablation
        ablation_result = ablate(
            model=model,
            input_text=prompt,
            targets=ablation_targets,
            max_new_tokens=10
        )
        
        ablated_correct = expected.lower() in ablation_result.modified_output.lower()
        result.ablation_accuracy = 1.0 if ablated_correct else 0.0
        result.accuracy_drop = result.baseline_accuracy - result.ablation_accuracy
        
        logger.info(f"Ablated output: '{ablation_result.modified_output}'")
        logger.info(f"Accuracy drop: {result.accuracy_drop:.2%}")
        
        # Per-head ablation analysis
        for h in induction_heads[:config.ablate_top_k]:
            single_target = InterventionTarget(
                layer=h["layer"],
                component_type=ComponentType.ATTN_HEAD,
                component_idx=h["head"]
            )
            
            single_result = ablate(model, prompt, [single_target], max_new_tokens=10)
            single_correct = expected.lower() in single_result.modified_output.lower()
            
            key = f"L{h['layer']}H{h['head']}"
            result.per_head_ablation[key] = 0.0 if single_correct else 1.0
    
    # Step 7: Copying analysis
    logger.info("Analyzing copying mechanism...")
    result.copying_score, result.attention_to_context = _analyze_copying(
        model=model,
        attn_data=attn_data,
        prompt=prompt,
        num_shots=config.num_shots
    )
    
    logger.info(f"Copying score: {result.copying_score:.3f}")
    
    # Save results
    _save_icl_results(result, config.output_dir)
    
    return result


def _evaluate_icl(
    model,
    prompt: str,
    expected: str,
    max_new_tokens: int = 10
) -> Tuple[bool, str]:
    """Evaluate ICL task accuracy."""
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # Greedy decoding
            return_type="str"
        )
    
    if isinstance(output, list):
        output = output[0]
    
    # Extract generated part (after prompt)
    generated = output[len(prompt):].strip()
    
    # Check if expected output is in generated text
    is_correct = expected.lower() in generated.lower()
    
    return is_correct, generated


def _analyze_copying(
    model,
    attn_data,
    prompt: str,
    num_shots: int
) -> Tuple[float, Dict[int, float]]:
    """
    Analyze the copying mechanism in ICL.
    
    Measures attention from the prediction position to context examples.
    """
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)
    n_layers = attn_data.patterns.shape[0]
    n_heads = attn_data.patterns.shape[1]
    
    # Last position (prediction) attention to context
    last_pos = n_tokens - 1
    
    # Estimate context token range (rough heuristic: first 80% of tokens are context)
    context_end = int(n_tokens * 0.8)
    
    # Compute attention to context per layer (averaged over heads)
    attention_to_context = {}
    
    for layer in range(n_layers):
        layer_attn = attn_data.patterns[layer]  # [n_heads, seq, seq]
        
        # Attention from last position to context
        attn_to_ctx = layer_attn[:, last_pos, :context_end].mean().item()
        attention_to_context[layer] = attn_to_ctx
    
    # Overall copying score: max attention to context across layers
    copying_score = max(attention_to_context.values()) if attention_to_context else 0.0
    
    return copying_score, attention_to_context


def _save_icl_results(result: ICLStudyResult, output_dir: str):
    """Save ICL study results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable dict
    result_dict = {
        "model_name": result.model_name,
        "config": {
            "model_name": result.config.model_name,
            "dtype": result.config.dtype,
            "task_type": result.config.task_type,
            "num_shots": result.config.num_shots,
            "induction_threshold": result.config.induction_threshold,
            "circuit_method": result.config.circuit_method,
            "ablate_top_k": result.config.ablate_top_k
        },
        "induction_heads": [
            {"layer": h[0], "head": h[1], "score": result.induction_scores.get(h, 0.0)}
            for h in result.induction_heads
        ],
        "circuit": {
            "nodes": result.circuit_nodes,
            "edges": result.circuit_edges,
            "faithfulness": result.circuit_faithfulness,
            "completeness": result.circuit_completeness
        },
        "accuracy": {
            "baseline": result.baseline_accuracy,
            "ablation": result.ablation_accuracy,
            "drop": result.accuracy_drop
        },
        "per_head_ablation": result.per_head_ablation,
        "copying_analysis": {
            "copying_score": result.copying_score,
            "attention_to_context": result.attention_to_context
        }
    }
    
    output_file = output_path / f"icl_study_{result.config.task_type}_{result.model_name.replace('/', '_')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def run_icl_comparison(
    task_types: List[str] = ["translation", "sentiment", "arithmetic"],
    model_name: str = "Qwen/Qwen2.5-0.5B",
    output_dir: str = "results/icl_comparison"
) -> Dict[str, ICLStudyResult]:
    """Compare ICL circuits across different task types."""
    
    _validate_qwen_model(model_name)
    
    results = {}
    
    for task_type in task_types:
        logger.info(f"\n=== Running {task_type} task ===")
        
        config = ICLStudyConfig(
            model_name=model_name,
            task_type=task_type,
            output_dir=output_dir
        )
        
        try:
            result = run_icl_study(config)
            results[task_type] = result
        except Exception as e:
            logger.error(f"Failed to run {task_type}: {e}")
    
    # Save comparison summary
    summary = {}
    for task_type, result in results.items():
        summary[task_type] = {
            "num_induction_heads": len(result.induction_heads),
            "circuit_nodes": len(result.circuit_nodes),
            "faithfulness": result.circuit_faithfulness,
            "baseline_accuracy": result.baseline_accuracy,
            "accuracy_drop": result.accuracy_drop,
            "copying_score": result.copying_score
        }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "icl_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run ICL study on Qwen
    config = ICLStudyConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        task_type="translation",
        num_shots=4
    )
    
    result = run_icl_study(config)
    
    print(f"\n=== ICL Circuit Study Results ===")
    print(f"Model: {result.model_name}")
    print(f"Task: {result.config.task_type}")
    print(f"Induction heads found: {len(result.induction_heads)}")
    print(f"Circuit nodes: {len(result.circuit_nodes)}")
    print(f"Baseline accuracy: {result.baseline_accuracy:.2%}")
    print(f"Accuracy drop after ablation: {result.accuracy_drop:.2%}")
    print(f"Copying score: {result.copying_score:.3f}")
