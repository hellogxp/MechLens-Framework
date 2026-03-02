"""
MechLens Experiments Module

Case studies and cross-model experiment runners.
"""

from mechlens.experiments.hallucination_study import (
    HallucinationStudyConfig,
    HallucinationStudyResult,
    run_hallucination_study,
    run_counterfact_study,
    compare_models as compare_hallucination_models
)

from mechlens.experiments.icl_study import (
    ICLStudyConfig,
    ICLStudyResult,
    run_icl_study,
    run_icl_comparison,
    create_icl_prompts
)

from mechlens.experiments.cross_model import (
    ExperimentConfig,
    ModelExperimentResult,
    CrossModelResult,
    run_cross_model_experiment,
    quick_comparison,
    ALL_MODELS,
    QWEN_MODELS,
    ROME_MEMIT_MODELS
)

__all__ = [
    # Hallucination study
    "HallucinationStudyConfig",
    "HallucinationStudyResult",
    "run_hallucination_study",
    "run_counterfact_study",
    "compare_hallucination_models",
    
    # ICL study
    "ICLStudyConfig",
    "ICLStudyResult",
    "run_icl_study",
    "run_icl_comparison",
    "create_icl_prompts",
    
    # Cross-model
    "ExperimentConfig",
    "ModelExperimentResult",
    "CrossModelResult",
    "run_cross_model_experiment",
    "quick_comparison",
    "ALL_MODELS",
    "QWEN_MODELS",
    "ROME_MEMIT_MODELS"
]
