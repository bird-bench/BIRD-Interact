#!/bin/bash

# Default settings
DATA_PATH="./data/bird-interact-lite/bird_interact_data.jsonl"
LOG_DIR="./outputs/single_runs/"
MAX_TURNS=100
AGENT_MODEL="gemini-2.0-flash"
AGENT_MODEL_PROVIDER="gemini"
USER_MODEL="gemini-2.0-flash"
USER_MODEL_PROVIDER="gemini"
USER_PATIENCE_BUDGET=6
USE_ENCODER_DECODER=true
#### Debug settings
DEBUG_MODE=false
VERBOSE=true
# DEBUG_NUM=0
DEBUG_SOL_SQL=false
####
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
HUMAN_MODE=false

#### DB settings
DB_PORT=5432

# Parse command-line arguments
while (( "$#" )); do
  case "$1" in
    --data_path=*)
      DATA_PATH="${1#*=}"
      shift
      ;;
    --log_dir=*)
      LOG_DIR="${1#*=}"
      shift
      ;;
    --max_turns=*)
      MAX_TURNS="${1#*=}"
      shift
      ;;
    --agent_model=*)
      AGENT_MODEL="${1#*=}"
      shift
      ;;
    --user_model=*)
      USER_MODEL="${1#*=}"
      shift
      ;;
    --user_patience_budget=*)
      USER_PATIENCE_BUDGET="${1#*=}"
      shift
      ;;
    --encoder-decoder)
      USE_ENCODER_DECODER=true
      shift
      ;;
    --debug)
      DEBUG_MODE=true
      shift
      ;;
    --debug_sol_sql)
      DEBUG_SOL_SQL=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --debug_num=*)
      DEBUG_NUM="${1#*=}"
      shift
      ;;
    --full)
      # Run on the full dataset
      DEBUG_NUM=""
      shift
      ;;
    --human_mode)
      HUMAN_MODE=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --data_path=PATH      Path to BIRD-Interact dataset (default: $DATA_PATH)"
      echo "  --log_dir=DIR         Directory to store logs (default: $LOG_DIR)"
      echo "  --max_turns=N         Maximum interaction turns (default: $MAX_TURNS)"
      echo "  --agent_model=MODEL   LLM for agent (default: $AGENT_MODEL)"
      echo "  --user_model=MODEL    LLM for user simulator (default: $USER_MODEL)"
      echo "  --user_patience_budget=N     User patience budget (default: $USER_PATIENCE_BUDGET)"
      echo "  --encoder-decoder     Use encoder-decoder approach for user simulator"
      echo "  --debug               Enable debug mode"
      echo "  --verbose             Enable verbose output"
      echo "  --debug_num=N         Number of examples to debug (default: $DEBUG_NUM)"
      echo "  --full                Run on full dataset (overrides debug_num)"
      echo "  -h, --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Add timestamp to log directory to avoid overwriting previous results
LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

# Build command
CMD="PYTHONPATH=$(pwd) python experiments/eval_react_bird_interact.py"
CMD+=" --env bird_interact_sql"
CMD+=" --data_path $DATA_PATH"
CMD+=" --log_dir $LOG_DIR"
CMD+=" --max_turns $MAX_TURNS"
CMD+=" --agent_model $AGENT_MODEL"
CMD+=" --user_model $USER_MODEL"
CMD+=" --user_patience_budget $USER_PATIENCE_BUDGET"
CMD+=" --agent_model_provider $AGENT_MODEL_PROVIDER"
CMD+=" --user_model_provider $USER_MODEL_PROVIDER"
CMD+=" --db_port $DB_PORT"

# Add optional flags
if [ "$USE_ENCODER_DECODER" = true ]; then
  CMD+=" --use_encoder_decoder"
fi

if [ "$DEBUG_MODE" = true ]; then
  CMD+=" --debug_mode"
fi

if [ "$VERBOSE" = true ]; then
  CMD+=" --verbose"
fi

if [ -n "$DEBUG_NUM" ]; then
  CMD+=" --debug_num $DEBUG_NUM"
fi

if [ "$DEBUG_SOL_SQL" = true ]; then
  CMD+=" --debug_sol_sql"
fi

if [ "$HUMAN_MODE" = true ]; then
  CMD+=" --human_mode"
fi

# Print the command
echo "Running experiment with command:"
echo "$CMD"
echo ""
echo "Results will be saved to: $LOG_DIR"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Run the experiment
eval $CMD 