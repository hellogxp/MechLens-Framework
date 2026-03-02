#!/bin/bash
# ================================================================
# Round 2 GPU Experiments Master Runner
# COLM 2026 Paper: Late Crystallization of Factual Knowledge
# ================================================================
#
# Run on PAI-DSW with A100 40GB GPU
# Total estimated time: ~15-21 GPU hours
#
# Usage:
#   chmod +x experiments/run_round2_all.sh
#   bash experiments/run_round2_all.sh          # Run all experiments
#   bash experiments/run_round2_all.sh --quick  # Quick test (50 samples)
#
# Priority order (run in this order; stop if budget exhausted):
#   2.1 Tuned Lens [CRITICAL]  ~6h  - kills artifact concern
#   2.2 MMLU       [HIGH]      ~3h  - cross-benchmark
#   2.3 Instruct   [HIGH]      ~2h  - generalization
#   2.4 14B Scale   [MEDIUM]   ~4h  - scale validation
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
    echo "=== QUICK MODE: Using 50 samples per experiment ==="
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/results/round2_logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Round 2 GPU Experiments - COLM 2026"
echo "Started: $(date)"
echo "Log directory: $LOG_DIR"
echo "================================================================"

# Ensure dependencies
pip install scipy datasets --quiet 2>/dev/null || true

# ---- 2.1 Tuned Lens Comparison [CRITICAL] ----
echo ""
echo "============================================================"
echo "[2.1] TUNED LENS FEP COMPARISON (CRITICAL)"
echo "Expected: ~6 GPU hours"
echo "============================================================"

if [ "$QUICK_MODE" = true ]; then
    TUNED_ARGS="--models qwen --train-samples 200 --max-eval-samples 50"
else
    TUNED_ARGS="--models qwen llama mistral --train-samples 2000"
fi

python experiments/run_tuned_lens_comparison.py $TUNED_ARGS \
    2>&1 | tee "$LOG_DIR/2.1_tuned_lens_${TIMESTAMP}.log"
echo "[2.1] COMPLETE"

# ---- 2.2 MMLU Cross-Benchmark [HIGH] ----
echo ""
echo "============================================================"
echo "[2.2] MMLU CROSS-BENCHMARK VALIDATION (HIGH)"
echo "Expected: ~3 GPU hours"
echo "============================================================"

if [ "$QUICK_MODE" = true ]; then
    MMLU_ARGS="--models qwen --max-samples 100"
else
    MMLU_ARGS="--models qwen llama mistral"
fi

python experiments/run_mmlu_fep_validation.py $MMLU_ARGS \
    2>&1 | tee "$LOG_DIR/2.2_mmlu_${TIMESTAMP}.log"
echo "[2.2] COMPLETE"

# ---- 2.3 Instruction-Tuned Model [HIGH] ----
echo ""
echo "============================================================"
echo "[2.3] INSTRUCTION-TUNED MODEL PILOT (HIGH)"
echo "Expected: ~2 GPU hours"
echo "============================================================"

if [ "$QUICK_MODE" = true ]; then
    INSTRUCT_ARGS="--max-samples 50 --skip-mc1"
else
    INSTRUCT_ARGS=""
fi

python experiments/run_instruct_model_pilot.py $INSTRUCT_ARGS \
    2>&1 | tee "$LOG_DIR/2.3_instruct_${TIMESTAMP}.log"
echo "[2.3] COMPLETE"

# ---- 2.4 14B Scale Validation [MEDIUM] ----
echo ""
echo "============================================================"
echo "[2.4] 14B SCALE VALIDATION (MEDIUM)"
echo "Expected: ~4 GPU hours"
echo "============================================================"

if [ "$QUICK_MODE" = true ]; then
    SCALE_ARGS="--max-samples 50 --quantize"
else
    SCALE_ARGS="--quantize"  # Use quantization by default for 14B on 40GB
fi

python experiments/run_14b_scale_validation.py $SCALE_ARGS \
    2>&1 | tee "$LOG_DIR/2.4_14b_scale_${TIMESTAMP}.log"
echo "[2.4] COMPLETE"

# ---- Summary ----
echo ""
echo "================================================================"
echo "ALL ROUND 2 EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "================================================================"
echo ""
echo "Results saved in:"
echo "  results/tuned_lens_comparison/"
echo "  results/mmlu_fep/"
echo "  results/instruct_pilot/"
echo "  results/scale_validation/"
echo ""
echo "Logs saved in:"
echo "  $LOG_DIR/"
echo ""
echo "Next: Transfer results to local machine for paper integration."
echo "================================================================"
