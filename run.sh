#!/bin/bash
# MoE Congestion Game — Run Script
# ================================
# GPU requirements:
#   Mixtral-8x7B: 2×A100 80GB (bf16, ~90GB)
#   Mixtral-8x7B-Instruct: same
#   DeepSeek-V2-Lite: 1×A100 80GB
#
# Install:
#   pip install torch transformers datasets matplotlib seaborn accelerate --break-system-packages

set -e

MODEL="${1:-mistralai/Mixtral-8x7B-Instruct-v0.1}"
PHASE="${2:-ABCD}"
OUTPUT_DIR="${3:-.}"

echo "=================================================="
echo "MoE Congestion Game Analysis"
echo "Model: $MODEL"
echo "Phases: $PHASE"
echo "Output: $OUTPUT_DIR"
echo "=================================================="

# Phase A only (fast test, ~20min)
if [ "$PHASE" = "A" ]; then
    python moe_congestion_game.py \
        --model "$MODEL" \
        --phase A \
        --n_per_category 30 \
        --output "$OUTPUT_DIR/moe_results_phase_a.json" \
        --output_dir "$OUTPUT_DIR"
    exit 0
fi

# Full analysis
python moe_congestion_game.py \
    --model "$MODEL" \
    --phase "$PHASE" \
    --n_per_category 50 \
    --n_poa 30 \
    --n_braess 20 \
    --output "$OUTPUT_DIR/moe_congestion_results.json" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Done. Results in $OUTPUT_DIR/"
