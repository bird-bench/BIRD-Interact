#!/bin/bash

# Default settings
DATA_PATH="./data/bird-interact-lite/bird_interact_data.jsonl"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="./outputs/batch_runs/${TIMESTAMP}"
NUM_THREADS=8 # Number of parallel API calls

# --- Agent Models ---
# Define models as an array

AGENT_MODEL_NAMES=("gemini-2.0-flash-001")
DB_PORT=5432
# ## Resume from a previous run
# RESUME=true
# BASE_OUTPUT_DIR="./outputs/batch_runs/20250513_160734"
declare -a PATIENCE_BUDGETS=(6)  # Match the budgets from reference script

# --- User Simulator Model ---
USER_MODEL_NAME="gemini-2.0-flash-001"
USER_SIM_MODE="encoder_decoder" # "vanilla" or "encoder_decoder"

# --- Budget & Turn Limits ---
MAX_TURNS=60  # Match the max turns from reference script

# --- Output & Run Configuration ---

LIMIT=-1 # -1 to run all samples, otherwise limits the number of samples loaded
# RESUME=false

START_INDEX=0 # Start processing from this sample index (0-based)

# --- Data & Output Paths ---
AGENT_OUTPUT_PATH="" # Optional: Base path for agent API call raw outputs
USER_OUTPUT_PATH="" # Optional: Base path for user simulator API call raw outputs
LOG_LEVEL="WARNING"

# --- Parse Command-line Arguments ---
while (( "$#" )); do
  case "$1" in
    --data_path=*) DATA_PATH="${1#*=}"; shift ;;
    --agent_models=*) AGENT_MODELS="${1#*=}"; shift ;;
    --user_model=*) USER_MODEL_NAME="${1#*=}"; shift ;;
    --user_sim_mode=*) USER_SIM_MODE="${1#*=}"; shift ;;
    --budgets=*) PATIENCE_BUDGETS="${1#*=}"; shift ;;
    --max_turns=*) MAX_TURNS="${1#*=}"; shift ;;
    --base_output_dir=*) BASE_OUTPUT_DIR="${1#*=}"; shift ;;
    --num_threads=*) NUM_THREADS="${1#*=}"; shift ;;
    --limit=*) LIMIT="${1#*=}"; shift ;;
    --start_index=*) START_INDEX="${1#*=}"; shift ;;
    --db_port=*) DB_PORT="${1#*=}"; shift ;;
    --agent_output_path=*) AGENT_OUTPUT_PATH="${1#*=}"; shift ;;
    --user_output_path=*) USER_OUTPUT_PATH="${1#*=}"; shift ;;
    --resume) RESUME=true; shift ;;
    --verbose) VERBOSE=true; shift ;;
    --log_level=*) LOG_LEVEL="${1#*=}"; shift ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --data_path=PATH         Path to BIRD-Interact dataset (default: $DATA_PATH)"
      echo "  --agent_models=M1,M2     Comma-separated agent model names (default: ${AGENT_MODEL_NAMES[*]})"
      echo "  --user_model=MODEL       User simulator model name (default: $USER_MODEL_NAME)"
      echo "  --user_sim_mode=MODE     'vanilla' or 'encoder_decoder' (default: $USER_SIM_MODE)"
      echo "  --budgets=B1,B2          Comma-separated patience budgets (default: ${PATIENCE_BUDGETS[*]})"
      echo "  --max_turns=N            Maximum interaction turns per sample (default: $MAX_TURNS)"
      echo "  --base_output_dir=DIR    Base directory for output logs (default: $BASE_OUTPUT_DIR)"
      echo "  --num_threads=N          Number of parallel API threads (default: $NUM_THREADS)"
      echo "  --limit=N                Max samples to process (-1 for all) (default: $LIMIT)"
      echo "  --start_index=N          Start processing from this sample index (default: $START_INDEX)"
      echo "  --db_port=PORT           Database port (default: $DB_PORT)"
      echo "  --agent_output_path=PATH Optional: Base path for agent API call raw outputs"
      echo "  --user_output_path=PATH  Optional: Base path for user simulator API call raw outputs"
      echo "  --resume                 Resume incomplete runs found in output dirs"
      echo "  --verbose                Enable verbose output in main.py"
      echo "  -h, --help               Show this help message"
      exit 0
      ;;
    *) echo "Unknown argument: $1 / Use --help"; exit 1 ;;
  esac
done

# --- Argument Validation ---
if [ "$USER_SIM_MODE" != "vanilla" ] && [ "$USER_SIM_MODE" != "encoder_decoder" ]; then
    echo "Error: --user_sim_mode must be 'vanilla' or 'encoder_decoder'."
    exit 1
fi

# --- Function to Run a Single Experiment Configuration ---
run_single_config() {
    local agent_model=$1
    local patience_budget=$2

    # Replace potential problematic characters in model name for directory path
    local safe_model_name=$(echo "$agent_model" | sed 's/[^a-zA-Z0-9_-]/-/g')
    local exp_output_dir="${BASE_OUTPUT_DIR}/${safe_model_name}_patience_${patience_budget}"
    local output_file="${exp_output_dir}/results.jsonl"
    local log_file="${exp_output_dir}/experiment.log"

    echo "-----------------------------------------------------"
    echo "Running Configuration:"
    echo "  Agent Model:      $agent_model"
    echo "  Patience Budget:  $patience_budget"
    echo "  User Model:       $USER_MODEL_NAME"
    echo "  User Sim Mode:    $USER_SIM_MODE"
    echo "  Output Directory: $exp_output_dir"
    echo "  Output File:      $output_file"
    echo "  Log File:         $log_file"
    echo "  Database:         $DB_PORT"
    echo "-----------------------------------------------------"

    # Check for resume condition
    local resume_flag=""
    if [ "$RESUME" = true ] && [ -f "$output_file" ]; then
        echo "Attempting to resume from existing file: $output_file"
        resume_flag="--resume"
    elif [ -f "$output_file" ]; then
         echo "Warning: Output file already exists, but resume flag is not set. Will overwrite."
         echo "  $output_file"
         rm -f "$output_file" # Or handle differently, e.g., skip or timestamp
    fi

    # Create output directory
    mkdir -p "$exp_output_dir"

    # Build command for batch_run_bird_interact/main.py
    CMD="PYTHONPATH=$(pwd) python batch_run_bird_interact/main.py"
    CMD+=" --data_path "$DATA_PATH""
    CMD+=" --output_path "$output_file""
    CMD+=" --agent_model "$agent_model""
    CMD+=" --user_model "$USER_MODEL_NAME""
    CMD+=" --user_sim_mode "$USER_SIM_MODE""
    CMD+=" --user_patience_budget $patience_budget"
    CMD+=" --max_turns $MAX_TURNS"
    CMD+=" --num_threads $NUM_THREADS"
    CMD+=" --start_index $START_INDEX"
    CMD+=" --log_file "$log_file""
    if [ "$LIMIT" != "-1" ]; then
        CMD+=" --limit $LIMIT"
    fi
    if [ ! -z "$AGENT_OUTPUT_PATH" ]; then
        CMD+=" --agent_output_path "$AGENT_OUTPUT_PATH""
    fi
    if [ ! -z "$USER_OUTPUT_PATH" ]; then
        CMD+=" --user_output_path "$USER_OUTPUT_PATH""
    fi
    CMD+=" --db_port $DB_PORT"

    if [ "$VERBOSE" = true ]; then
        CMD+=" --verbose"
    fi
    if [ -n "$LOG_LEVEL" ]; then
        CMD+=" --log_level $LOG_LEVEL"
    fi 
    if [ -n "$resume_flag" ]; then
        CMD+=" $resume_flag"
    fi
    if [ -n "$START_ACTUAL_INDEX" ]; then
        CMD+=" --start_actual_index $START_ACTUAL_INDEX"
    fi
    # Run the experiment
    echo "Executing command:"
    echo "$CMD"
    echo ""
    eval $CMD
    echo "-----------------------------------------------------"
    echo "Finished Configuration: $agent_model / Patience $patience_budget"
    echo "Results saved in: $output_file"
    echo "Log file: $log_file"
    echo "-----------------------------------------------------"
    echo ""
}

# --- Main Execution Logic ---
echo "Starting batch evaluation runs..."
echo "Base output directory: $BASE_OUTPUT_DIR"

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Run experiments for each combination
for budget in "${PATIENCE_BUDGETS[@]}"; do
    for agent_model in "${AGENT_MODEL_NAMES[@]}"; do
        run_single_config "$agent_model" "$budget"
    done
done

echo "====================================================="
echo "All batch evaluation runs completed."
echo "Results saved under: $BASE_OUTPUT_DIR"
echo "=====================================================" 