#!/bin/bash
# =============================================================================
# Example Script for User Simulator Pipeline
# =============================================================================
#
# This script demonstrates how to run the complete user simulator pipeline
# including both baseline and proposed methods, with LLM-as-Judge evaluation.
#
# Before running:
# 1. Set your API key in config.py or via environment variable:
#    export OPENAI_API_KEY="your-api-key"
#    export OPENAI_BASE_URL="https://api.openai.com/v1"  # or your custom endpoint
#
# 2. Prepare your data files in data/ directory
#
# Usage:
#    chmod +x scripts/run_example.sh
#    ./scripts/run_example.sh
# =============================================================================

set -e  # Exit on error

# Configuration
MODEL="gpt-4o"                    # Model for user simulator
JUDGE_MODEL="gpt-4o"              # Model for LLM-as-Judge
DATA_TYPE="labeled"               # labeled, unlabeled, or unanswerable
INPUT_DATA="data/UserSim_Guard/data_${DATA_TYPE}.jsonl"
OUTPUT_DIR="results"

# Map data type for judge (unanswerable -> una)
if [ "$DATA_TYPE" == "unanswerable" ]; then
    JUDGE_DATA_TYPE="una"
else
    JUDGE_DATA_TYPE="$DATA_TYPE"
fi

# Create output directories
mkdir -p ${OUTPUT_DIR}/prompt
mkdir -p ${OUTPUT_DIR}/response

echo "=============================================="
echo "User Simulator Pipeline - Example Run"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Judge Model: ${JUDGE_MODEL}"
echo "Data Type: ${DATA_TYPE}"
echo "Input: ${INPUT_DATA}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# =============================================================================
# BASELINE PIPELINE
# =============================================================================
echo ""
echo ">>> Step 1: Generate BASE prompts"
python run_pipeline.py generate \
    --prompt_type base \
    --input ${INPUT_DATA} \
    --output ${OUTPUT_DIR}/prompt/base_${DATA_TYPE}_prompts.jsonl \
    --db_base_path data/databases

echo ""
echo ">>> Step 2: Run BASE inference"
python run_pipeline.py infer \
    --input ${OUTPUT_DIR}/prompt/base_${DATA_TYPE}_prompts.jsonl \
    --output ${OUTPUT_DIR}/response/base_${DATA_TYPE}_responses.jsonl \
    --model ${MODEL} \
    --temperature 0

echo ""
echo ">>> Step 3: Generate JUDGE prompts for BASE"
python run_pipeline.py judge \
    --prompts ${OUTPUT_DIR}/prompt/base_${DATA_TYPE}_prompts.jsonl \
    --responses ${OUTPUT_DIR}/response/base_${DATA_TYPE}_responses.jsonl \
    --output ${OUTPUT_DIR}/prompt/judge_base_${DATA_TYPE}_prompts.jsonl \
    --data_type ${JUDGE_DATA_TYPE}

echo ""
echo ">>> Step 4: Run JUDGE inference for BASE"
python run_pipeline.py infer \
    --input ${OUTPUT_DIR}/prompt/judge_base_${DATA_TYPE}_prompts.jsonl \
    --output ${OUTPUT_DIR}/response/judge_base_${DATA_TYPE}_responses.jsonl \
    --model ${JUDGE_MODEL} \
    --temperature 0

echo ""
echo ">>> BASE Results:"
python run_pipeline.py stats \
    --responses ${OUTPUT_DIR}/response/judge_base_${DATA_TYPE}_responses.jsonl \
    --data_type ${JUDGE_DATA_TYPE}

# =============================================================================
# PROPOSED PIPELINE (Step1 + Step2)
# =============================================================================
echo ""
echo "=============================================="
echo ">>> Step 5: Generate STEP1 prompts"
python run_pipeline.py generate \
    --prompt_type step1 \
    --input ${INPUT_DATA} \
    --output ${OUTPUT_DIR}/prompt/step1_${DATA_TYPE}_prompts.jsonl

echo ""
echo ">>> Step 6: Run STEP1 inference"
python run_pipeline.py infer \
    --input ${OUTPUT_DIR}/prompt/step1_${DATA_TYPE}_prompts.jsonl \
    --output ${OUTPUT_DIR}/response/step1_${DATA_TYPE}_responses.jsonl \
    --model ${MODEL} \
    --temperature 0

echo ""
echo ">>> Step 7: Generate STEP2 prompts"
python run_pipeline.py generate \
    --prompt_type step2 \
    --input ${INPUT_DATA} \
    --output ${OUTPUT_DIR}/prompt/step2_${DATA_TYPE}_prompts.jsonl \
    --step1_responses ${OUTPUT_DIR}/response/step1_${DATA_TYPE}_responses.jsonl \
    --db_base_path data/databases

echo ""
echo ">>> Step 8: Run STEP2 inference"
python run_pipeline.py infer \
    --input ${OUTPUT_DIR}/prompt/step2_${DATA_TYPE}_prompts.jsonl \
    --output ${OUTPUT_DIR}/response/step2_${DATA_TYPE}_responses.jsonl \
    --model ${MODEL} \
    --temperature 0

echo ""
echo ">>> Step 9: Generate JUDGE prompts for PROPOSED"
python run_pipeline.py judge \
    --prompts ${OUTPUT_DIR}/prompt/step2_${DATA_TYPE}_prompts.jsonl \
    --responses ${OUTPUT_DIR}/response/step2_${DATA_TYPE}_responses.jsonl \
    --output ${OUTPUT_DIR}/prompt/judge_proposed_${DATA_TYPE}_prompts.jsonl \
    --data_type ${JUDGE_DATA_TYPE}

echo ""
echo ">>> Step 10: Run JUDGE inference for PROPOSED"
python run_pipeline.py infer \
    --input ${OUTPUT_DIR}/prompt/judge_proposed_${DATA_TYPE}_prompts.jsonl \
    --output ${OUTPUT_DIR}/response/judge_proposed_${DATA_TYPE}_responses.jsonl \
    --model ${JUDGE_MODEL} \
    --temperature 0

echo ""
echo ">>> PROPOSED Results:"
python run_pipeline.py stats \
    --responses ${OUTPUT_DIR}/response/judge_proposed_${DATA_TYPE}_responses.jsonl \
    --data_type ${JUDGE_DATA_TYPE}

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================="
echo "PIPELINE COMPLETE!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  Base prompts:     ${OUTPUT_DIR}/prompt/base_${DATA_TYPE}_prompts.jsonl"
echo "  Base responses:   ${OUTPUT_DIR}/response/base_${DATA_TYPE}_responses.jsonl"
echo "  Step1 prompts:    ${OUTPUT_DIR}/prompt/step1_${DATA_TYPE}_prompts.jsonl"
echo "  Step1 responses:  ${OUTPUT_DIR}/response/step1_${DATA_TYPE}_responses.jsonl"
echo "  Step2 prompts:    ${OUTPUT_DIR}/prompt/step2_${DATA_TYPE}_prompts.jsonl"
echo "  Step2 responses:  ${OUTPUT_DIR}/response/step2_${DATA_TYPE}_responses.jsonl"
echo "  Judge results:    ${OUTPUT_DIR}/response/judge_*_responses.jsonl"
echo ""
echo "To view results:"
echo "  python run_pipeline.py stats --responses ${OUTPUT_DIR}/response/judge_base_${DATA_TYPE}_responses.jsonl --data_type ${JUDGE_DATA_TYPE}"
echo "  python run_pipeline.py stats --responses ${OUTPUT_DIR}/response/judge_proposed_${DATA_TYPE}_responses.jsonl --data_type ${JUDGE_DATA_TYPE}"
