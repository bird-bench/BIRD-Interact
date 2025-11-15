import sys
import os
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
import jsonlines
from tqdm import tqdm
import math # For ceil
import re 

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.llm_utils.call_api_batch import collect_response_from_api, load_jsonl
from src.config.db_config import set_global_db_config, reset_global_db_config
from batch_run_bird_interact.sample_status import SampleStatus
from batch_run_bird_interact.action_handler import (
    execute_env_action,
    execute_submit_action,
    _schema_cache,
    load_db_data_if_needed,
    close_db_connection,
    get_db_connection # Needed for schema loading in user sim prompts
)
from batch_run_bird_interact.prompt_utils import (
    build_initial_agent_prompt,
    get_agent_prompt_for_turn,
    parse_agent_response,
    build_user_encoder_prompt, # Assuming encoder/decoder for user sim
    build_user_decoder_prompt,
    parse_encoder_response
)

# Set up logger (consider using a more robust logging setup)
import logging

def setup_logging(verbose: bool = False, log_level: str = "INFO", log_file: str = None):
    """Configure logging with the specified verbosity and level.
    
    Args:
        verbose: Whether to enable verbose logging
        log_level: The logging level to use
        log_file: Path to the log file. If None, will use 'batch_evaluation.log' in current directory
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Set up handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Ensure the log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.FileHandler('batch_evaluation.log'))
    
    # Set up basic logging configuration
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',  # Simplified format
        handlers=handlers,
        force=True
    )
    
    # Set up module loggers with appropriate levels
    if verbose:
        # Verbose mode - show everything
        logging.getLogger('src.llm_utils').setLevel(logging.DEBUG)
        logging.getLogger('batch_run_bird_interact').setLevel(logging.DEBUG)
        logging.getLogger('urllib3').setLevel(logging.DEBUG)
        logging.getLogger('google.auth').setLevel(logging.DEBUG)
    else:
        # Normal mode - only show important information
        logging.getLogger('src.llm_utils').setLevel(logging.WARNING)
        logging.getLogger('batch_run_bird_interact').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('google.auth').setLevel(logging.WARNING)
        # Disable tqdm output when not verbose
        tqdm.monitor_interval = 0

logger = logging.getLogger(__name__)

# Action Costs (from bird-interact-project-rule)
ACTION_COSTS = {
    "ask": 2,
    "submit": 3,
    "execute": 1,
    "get_schema": 1,
    "get_all_column_meanings": 1,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1
}

WRONG_FORMAT_COST = 0.2


# Default fixed budget components (can be overridden by args if needed)
DEFAULT_ENV_INTERACT_BUDGET = 3
DEFAULT_SUBMIT_BUDGET = 3 # This seems low based on cost, maybe adjust?

def calculate_initial_budget(record: Dict, user_patience_budget: int) -> Tuple[float, float]:
    """Calculates the initial total budget for a sample."""
    amb_count = 0
    # Count critical ambiguities in user query
    if "user_query_ambiguity" in record:
        if "critical_ambiguity" in record["user_query_ambiguity"]:
            amb_count += len(record["user_query_ambiguity"]["critical_ambiguity"])
    # Count knowledge ambiguities
    if "knowledge_ambiguity" in record:
        amb_count += len(record["knowledge_ambiguity"])

    amb_resolve_budget = amb_count * ACTION_COSTS["ask"] # Cost of asking about each ambiguity

    # Using default fixed budgets for now, add args later if needed
    total_budget = (
        DEFAULT_ENV_INTERACT_BUDGET +
        DEFAULT_SUBMIT_BUDGET +
        amb_resolve_budget +
        user_patience_budget
    )
    logger.debug(f"Calculated budget for sample {record.get('instance_id', '?')}: "
                 f"Env={DEFAULT_ENV_INTERACT_BUDGET}, Submit={DEFAULT_SUBMIT_BUDGET}, "
                 f"AmbResolve={amb_resolve_budget} ({amb_count} ambiguities), "
                 f"UserPatience={user_patience_budget} -> Total={total_budget}")
    return total_budget, total_budget # Initial remaining budget is the total budget

def update_budget(sample_status: SampleStatus) -> bool:
    """
    Updates the remaining budget based on the last parsed action.
    Returns True if budget is depleted, False otherwise.
    Sets the force_submit flag if budget depletes.
    """
    if not sample_status.parsed_action:
        sample_status.remaining_budget -= WRONG_FORMAT_COST
        return sample_status.remaining_budget <= 0

    action_type = sample_status.parsed_action.split("(")[0] if "(" in sample_status.parsed_action else sample_status.parsed_action
    cost = ACTION_COSTS.get(action_type, WRONG_FORMAT_COST) # Default to 0 cost if action unknown

    if sample_status.force_submit and action_type != "submit":
        # If forced submit, only allow submit action. Don't deduct cost for invalid actions.
        logger.warning(f"Sample {sample_status.idx}: Budget depleted, but action was '{action_type}', not 'submit'. No budget change.")
        # We might want to enforce the next action MUST be submit in the main loop
        return True # Budget is still depleted

    sample_status.remaining_budget -= cost
    logger.debug(f"Sample {sample_status.idx}: Action='{action_type}', Cost={cost}, Remaining Budget={sample_status.remaining_budget:.1f}")

    if sample_status.remaining_budget <= 0 and not sample_status.force_submit:
        logger.warning(f"Sample {sample_status.idx}: Budget depleted. Forcing submit.")
        sample_status.force_submit = True
        return True # Budget depleted

    return sample_status.remaining_budget <= 0

def save_progress(output_path: str, all_statuses: List[SampleStatus]):
    """Saves the current state of all sample statuses to a JSONL file."""
    try:
        with jsonlines.open(output_path, mode='w') as writer:
            for status in all_statuses:
                # Convert dataclass to dict for JSON serialization
                # Be careful with complex objects if they exist in status
                status_dict = status.__dict__
                writer.write(status_dict)
        logger.debug(f"Progress saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save progress to {output_path}: {e}")

def load_progress(output_path: str) -> Optional[List[SampleStatus]]:
    """Loads the sample statuses from a JSONL file.
    
    Args:
        output_path: Path to the JSONL file containing the saved statuses
        
    Returns:
        List of SampleStatus objects if file exists and is valid, None otherwise
    """
    if not os.path.exists(output_path):
        logger.warning(f"No saved progress file found at {output_path}")
        return None
    
    if os.path.getsize(output_path) == 0:
        logger.warning(f"Progress file {output_path} exists but is empty")
        return None
        
    try:
        all_statuses = []
        with jsonlines.open(output_path, mode='r') as reader:
            for item in reader:
                # Basic validation of required fields
                if 'idx' not in item or 'original_data' not in item:
                    logger.warning(f"Skipping invalid status entry: missing required fields")
                    continue
                
                # Ensure original_data is present and contains essential fields
                original_data = item.get('original_data', {})
                if not original_data or 'selected_database' not in original_data:
                    logger.warning(f"Skipping status entry with ID {item.get('idx')}: invalid original_data")
                    continue
                
                # Convert dict back to SampleStatus with proper default values
                try:
                    status = SampleStatus(
                        idx=item['idx'],
                        original_data=original_data,
                        current_prompt=item.get('current_prompt', ''),
                        interaction_history=item.get('interaction_history', []),
                        remaining_budget=item.get('remaining_budget', 0.0),
                        total_budget=item.get('total_budget', 0.0),
                        phase1_completed=item.get('phase1_completed', False),
                        phase2_completed=item.get('phase2_completed', False),
                        task_finished=item.get('task_finished', False),
                        current_turn=item.get('current_turn', 0),
                        current_phase=item.get('current_phase', 1),
                        last_agent_response=item.get('last_agent_response'),
                        parsed_action_object=item.get('parsed_action_object'),
                        parsed_action=item.get('parsed_action'),
                        parsed_thought=item.get('parsed_thought'),
                        last_observation=item.get('last_observation'),
                        last_reward=item.get('last_reward'),
                        last_user_response=item.get('last_user_response'),
                        force_submit=item.get('force_submit', False),
                        successful_phase1_sql=item.get('successful_phase1_sql')
                    )
                    all_statuses.append(status)
                except Exception as e:
                    logger.warning(f"Failed to create SampleStatus from entry with ID {item.get('idx')}: {e}")
                    continue
        
        if not all_statuses:
            logger.warning(f"No valid sample statuses found in {output_path}")
            return None
            
        # Verify all required statuses are loaded
        # You might need to adjust this validation based on your specific requirements
        logger.debug(f"Loaded {len(all_statuses)} sample statuses from {output_path}")
        
        # Ensure statuses are sorted by idx for consistency
        all_statuses.sort(key=lambda s: s.idx)
        
        # Make a backup of the file before proceeding
        backup_path = f"{output_path}.bak"
        try:
            import shutil
            shutil.copy2(output_path, backup_path)
            logger.debug(f"Created backup of progress file at {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup of progress file: {e}")
        
        return all_statuses
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in progress file {output_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load progress from {output_path}: {e}")
        return None

def calculate_metrics(all_statuses: List[SampleStatus]) -> Dict[str, Any]:
    # TODO NOT CORRECT TO CALCULATE REWARD
    """Calculate final metrics for the evaluation run."""
    total_samples = len(all_statuses)
    if total_samples == 0:
        return {
            "total_samples": 0,
            "error": "No samples processed"
        }

    # Initialize counters
    completed_samples = 0
    phase1_completed = 0
    phase2_completed = 0
    total_turns = 0
    total_reward = 0.0
    samples_by_turn_count = {}  # Distribution of turns taken
    samples_by_reward = {}      # Distribution of rewards

    for status in all_statuses:
        # Count completions
        if status.task_finished:
            completed_samples += 1
            if status.phase1_completed:
                phase1_completed += 1
            if status.phase2_completed:
                phase2_completed += 1

        # Count turns (only for completed samples or samples that reached max turns)
        turns_taken = status.current_turn
        total_turns += turns_taken
        samples_by_turn_count[turns_taken] = samples_by_turn_count.get(turns_taken, 0) + 1

        # Track rewards
        if status.task_finished:
            if status.last_reward == None:
                status.last_reward = 0.0
            total_reward += status.last_reward
            reward_bucket = int(status.last_reward)  # Round down to nearest integer
            samples_by_reward[reward_bucket] = samples_by_reward.get(reward_bucket, 0) + 1

    # Calculate averages and rates
    completion_rate = completed_samples / total_samples if total_samples > 0 else 0
    phase1_rate = phase1_completed / total_samples if total_samples > 0 else 0
    phase2_rate = phase2_completed / total_samples if total_samples > 0 else 0
    avg_turns = total_turns / total_samples if total_samples > 0 else 0
    avg_reward = total_reward / completed_samples if completed_samples > 0 else 0

    return {
        "total_samples": total_samples,
        "completion_rate": completion_rate,
        "phase1_completion_rate": phase1_rate,
        "phase2_completion_rate": phase2_rate,
        "avg_turns": avg_turns,
        "avg_reward": avg_reward,
        "total_reward": total_reward,
        "turn_distribution": dict(sorted(samples_by_turn_count.items())),
        "reward_distribution": dict(sorted(samples_by_reward.items())),
        "completed_samples": completed_samples,
        "phase1_completed": phase1_completed,
        "phase2_completed": phase2_completed
    }

def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print metrics in a readable format."""
    print("\n=== Evaluation Metrics ===")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Completion Rate: {metrics['completion_rate']:.1%}")
    print(f"Phase 1 Completion Rate: {metrics['phase1_completion_rate']:.1%}")
    print(f"Phase 2 Completion Rate: {metrics['phase2_completion_rate']:.1%}")
    print(f"Average Turns: {metrics['avg_turns']:.1f}")
    print(f"Average Reward: {metrics['avg_reward']:.1f}")
    print(f"Total Reward: {metrics['total_reward']:.1f}")
    
    print("\nTurn Distribution:")
    for turns, count in metrics['turn_distribution'].items():
        print(f"  {turns} turns: {count} samples")
    
    print("\nReward Distribution:")
    for reward, count in metrics['reward_distribution'].items():
        print(f"  {reward} points: {count} samples")

def run_batch_evaluation(args):
    """
    Main function to run the batch evaluation.
    """
    # Disable tqdm output if not verbose
    if not args.verbose:
        tqdm.monitor_interval = 0
        tqdm_kwargs = {'disable': True}
    else:
        tqdm_kwargs = {}
    
    # Log only important startup information
    logger.warning("=== Starting Batch Evaluation ===")
    logger.warning(f"Configuration:")
    logger.warning(f"- Agent Model: {args.agent_model}")
    logger.warning(f"- User Model: {args.user_model} (mode: {args.user_sim_mode})")
    logger.warning(f"- Processing {args.limit if args.limit else 'all'} samples from index {args.start_index} (actual index: {args.start_actual_index})")
    logger.warning(f"- Max turns: {args.max_turns}, Threads: {args.num_threads}")
    logger.warning(f"- User patience budget: {args.user_patience_budget}")
    logger.warning(f"- Log level: {args.log_level}")
    if args.resume:
        logger.warning("Resuming from previous run")

    # Determine base path for data loading
    data_path_base = os.path.dirname(args.data_path)

    # Check if resuming from existing run
    all_statuses = None
    resume_turn = 0
    if args.resume and os.path.exists(args.output_path):
        logger.warning(f"Attempting to resume from: {args.output_path}")
        all_statuses = load_progress(args.output_path)
        if all_statuses:
            logger.warning(f"Successfully loaded {len(all_statuses)} statuses from previous run")
            completed_turns = [s.current_turn for s in all_statuses]
            if completed_turns:
                resume_turn = max(completed_turns)
                logger.warning(f"Resuming from turn {resume_turn}")
                
                # Generate current metrics for the loaded state
                logger.warning("Generating metrics for current state...")
                current_metrics = calculate_metrics(all_statuses)
                print_metrics(current_metrics)
                
                # Save current metrics
                resume_metrics_path = f"{os.path.splitext(args.output_path)[0]}_resume_metrics.json"
                with open(resume_metrics_path, 'w') as f:
                    json.dump(current_metrics, f, indent=2)
                logger.warning(f"Resume metrics saved to: {resume_metrics_path}")
        else:
            logger.warning("Failed to load progress data. Starting from beginning.")

    # If not resuming or failed to resume, initialize from scratch
    if all_statuses is None:
        # 1. Load Data
        logger.debug(f"Loading data from: {args.data_path}")
        all_samples_data = load_jsonl(args.data_path)

        # Apply limit and start_index
        start_index = args.start_index if args.start_index >= 0 else 0
        end_index = len(all_samples_data)
        if args.limit is not None and args.limit > 0:
            end_index = min(start_index + args.limit, len(all_samples_data))

        if start_index >= len(all_samples_data):
            logger.error(f"Start index {start_index} is out of bounds for data length {len(all_samples_data)}.")
            return

        samples_to_process_data = all_samples_data[start_index:end_index]
        logger.debug(f"Processing {len(samples_to_process_data)} samples (index {start_index} to {end_index-1}).")

        # 2. Initialize Sample Statuses
        all_statuses = []
        for i, record in enumerate(samples_to_process_data):
            actual_idx = start_index + i + args.start_actual_index
            total_budget, remaining_budget = calculate_initial_budget(record, args.user_patience_budget)
            status = SampleStatus(
                idx=actual_idx,
                original_data=record,
                remaining_budget=remaining_budget,
                total_budget=total_budget
            )
            # Build and store the initial prompt (Turn 0)
            initial_prompt = build_initial_agent_prompt(status, {"total_budget": total_budget, "remaining_budget": remaining_budget})
            # Initial prompts are not directly sent to API here, but stored in status
            # The first API call will use get_agent_prompt_for_turn which returns this initial prompt
            all_statuses.append(status)

        # Ensure output directory exists before saving
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Initial save (contains original data and initial budget)
        save_progress(args.output_path, all_statuses)
        resume_turn = 0

    # 3. Main Processing Loop (Turn-based)
    current_turn = resume_turn
    # Check if there are any active samples
    active_samples_exist = any(not s.task_finished for s in all_statuses)
    
    if not active_samples_exist:
        logger.warning("No active samples remaining. All samples are already complete.")
        # Calculate and report metrics for the completed run
        metrics = calculate_metrics(all_statuses)
        print_metrics(metrics)
        
        # Save metrics to a separate file
        metrics_path = f"{os.path.splitext(args.output_path)[0]}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.warning(f"Metrics saved to: {metrics_path}")
        return

    logger.debug(f"Starting processing from turn {current_turn + 1}")
    while current_turn < args.max_turns and active_samples_exist:
        logger.warning(f"--- Starting Turn {current_turn + 1} ---")
        turn_start_time = time.time()

        # 3.1 Identify active samples
        active_statuses = [s for s in all_statuses if not s.task_finished]
        if not active_statuses:
            logger.warning("No active samples remaining. Finishing evaluation.")
            active_samples_exist = False
            break
        logger.warning(f"Active samples this turn: {len(active_statuses)}")

        # --- Agent Turn ---
        # 3.2 Prepare agent prompts
        agent_prompts = []
        agent_api_data = []
        prompt_map = {} # Map api_data index back to sample status index

        for i, status in enumerate(active_statuses):
            status.current_turn = current_turn + 1 # Update turn number
            full_prompt = get_agent_prompt_for_turn(status)
            agent_prompts.append(full_prompt)
            # Create data structure for call_api_batch
            api_data_item = {"prompt": full_prompt, "id": status.idx} # Use sample idx as id
            agent_api_data.append(api_data_item)
            prompt_map[i] = status.idx # api_data index -> original status index

        # 3.3 Batch call agent API
        logger.debug(f"Sending {len(agent_prompts)} prompts to agent model: {args.agent_model}")
        agent_api_start_time = time.time()
        # Decide output path for raw agent responses
        raw_agent_output = args.agent_output_path if args.agent_output_path else f"{args.output_path}.agent_raw_turn_{current_turn+1}.jsonl"
        # Assuming collect_response_from_api is adapted to write results keyed by the 'id' provided
        collect_response_from_api(
            prompt_list=[item['prompt'] for item in agent_api_data], # Explicitly pass prompts
            model_name=args.agent_model,
            data_list=agent_api_data, # Pass data with prompts and IDs
            output_path=raw_agent_output,
            num_threads=args.num_threads,
            stop=["Observation:", "\nObservation:", "Observation: "]
            # start_index=0 # Process all active prompts
        )
        agent_api_duration = time.time() - agent_api_start_time
        logger.debug(f"Agent API calls completed in {agent_api_duration:.2f} seconds.")

        # 3.4 Load and Parse agent responses
        # Load results from the temporary file created by collect_response_from_api
        agent_results = {}
        try:
            with jsonlines.open(raw_agent_output, mode='r') as reader:
                 for item in reader:
                     agent_results[item['id']] = item # Store by sample idx
        except FileNotFoundError:
            logger.error(f"Agent raw output file not found: {raw_agent_output}. Cannot proceed this turn.")
            # Handle this error - maybe retry or mark samples as failed? For now, break.
            # Set all status to end and count as failed
            for status in active_statuses:
                status.task_finished = True
                status.last_observation = "Error: No response from agent model."
                status.add_turn_log(status.parsed_thought, status.parsed_action_object, status.parsed_action, status.last_observation, 0.0, {"remaining_budget": status.remaining_budget, "total_budget": status.total_budget})
            break
        except Exception as e:
            logger.error(f"Error reading agent raw output file {raw_agent_output}: {e}")
            break

        logger.debug(f"Parsing {len(agent_results)} agent responses.")
        actions_to_process = {'Environment': [], 'User_ask': [], 'User_submit': []}

        for status in active_statuses:
            if status.idx not in agent_results:
                logger.warning(f"Sample {status.idx}: No response found from agent API call. Marking as finished.")
                status.task_finished = True
                status.last_observation = "Error: No response from agent model."
                continue # Skip processing for this sample

            result_data = agent_results[status.idx]
            status.last_agent_response = result_data.get("response", "Error: Missing response field")
            # TODO: Store token usage if needed: status.token_usage.update(...)

            # Parse thought, object, action
            thought, obj, action = parse_agent_response(status.last_agent_response)
            status.parsed_thought = thought
            status.parsed_action_object = obj
            status.parsed_action = action

            # Validate action based on force_submit flag
            if status.force_submit and not action.startswith("submit("):
                logger.warning(f"Sample {status.idx}: Budget depleted, agent action was '{action}'. Overriding to force submit.")
                # How to get a valid SQL for forced submit? Difficult.
                # Option 1: Use last successful SQL? Risky.
                # Option 2: Submit a dummy/error SQL?
                # Option 3: Ask agent for one final attempt with a special prompt? (Adds complexity)
                # For now, let's mark it as an error observation and finish.
                status.last_observation = "[SYSTEM NOTE: Budget depleted. Agent failed to submit. Task failed.]"
                status.task_finished = True
                # Log the turn even if failed
                status.add_turn_log(thought, obj, action, status.last_observation, 0.0, {"remaining_budget": status.remaining_budget, "total_budget": status.total_budget})
                continue

            # 3.5 Update Budget *before* executing action
            budget_depleted = update_budget(status)
            budget_info_after_action = {"remaining_budget": status.remaining_budget, "total_budget": status.total_budget, "force_submit": status.force_submit}

    
            # 3.6 Group actions
            if obj == "Environment":
                actions_to_process['Environment'].append(status)
            elif obj == "User" and action.startswith("ask("):
                actions_to_process['User_ask'].append(status)
            elif obj == "User" and action.startswith("submit("):
                 # Extract SQL for submission
                sql_match = re.search(r'submit\((.*)\)', action, re.DOTALL)
                if sql_match:
                    sql_to_submit = sql_match.group(1).strip().strip("'\"")
                    actions_to_process['User_submit'].append((status, sql_to_submit))
                else:
                    logger.warning(f"Sample {status.idx}: Could not parse SQL from submit action: {action}. Treating as error.")
                    status.last_observation = "Error: Invalid submit action format."
                    # Log the turn with error
                    status.add_turn_log(thought, obj, action, status.last_observation, 0.0, budget_info_after_action)

            else:
                logger.warning(f"Sample {status.idx}: Unknown action object/type: Object='{obj}', Action='{action}'")
                status.last_observation = f"Error: Unknown action object or type. The format should be in <thought>...</thought><interaction_object>...</interaction_object><action>...</action>. And the input(s) of action content is/are stricted to STRING ENCLOSED BY SINGLE PAIR OF QUOTES OR \"\"\"YOUR ACTION HERE\"\"\"."
                # Log the turn with error
                status.add_turn_log(thought, obj, action, status.last_observation, 0.0, budget_info_after_action)

        # --- Execute Actions Sequentially ---
        # 3.7 Process Environment Actions
        logger.debug(f"Executing {len(actions_to_process['Environment'])} Environment actions...")
        for status in tqdm(actions_to_process['Environment'], desc="Env Actions", **tqdm_kwargs):
             # Retrieve thought/obj/action stored previously
             thought = status.parsed_thought # Re-use variables for logging
             obj = status.parsed_action_object
             action = status.parsed_action
             logger.debug(f"Sample {status.idx}: Attempting Env Action: {action}")
             budget_info = {"remaining_budget": status.remaining_budget, "total_budget": status.total_budget, "force_submit": status.force_submit}

             observation, success = execute_env_action(status.parsed_action, status, data_path_base)
             observation = observation if isinstance(observation, str) else str(observation)
             status.last_observation = observation if isinstance(observation, str) else str(observation)
             # Add budget info to observation for next agent prompt
             budget_prompt = f"\n\n[SYSTEM NOTE: Remaining budget: {status.remaining_budget:.1f}/{status.total_budget:.1f}]"
             if status.force_submit:
                 budget_prompt += f" Budget depleted. You MUST submit next."
             status.last_observation += budget_prompt

             # Log this part of the turn
             status.add_turn_log(thought, obj, action, status.last_observation, 0.0, budget_info) # Reward is 0 for env actions
             logger.debug(f"Sample {status.idx}: Env Action '{action}' -> Obs: {observation[:100]}...")


        # 3.8 Prepare and Batch Call User Simulator (for 'ask' actions)
        user_ask_statuses = actions_to_process['User_ask']
        if user_ask_statuses:
            logger.debug(f"Processing {len(user_ask_statuses)} User 'ask' actions (mode: {args.user_sim_mode})...")
            user_api_prompts = []
            user_api_data = []
            user_prompt_map = {}
            processed_user_responses = {} # Store final responses {status_idx: response_str}

            # --- Prepare questions --- #
            valid_ask_statuses = []
            questions = {}
            for status in user_ask_statuses:
                question_match = re.search(r'ask\((.*)\)', status.parsed_action, re.DOTALL)
                if question_match:
                    question = question_match.group(1).strip().strip("'\"")
                    questions[status.idx] = question
                    valid_ask_statuses.append(status)
                else:
                    logger.warning(f"Sample {status.idx}: Could not parse question from ask action: {status.parsed_action}")
                    status.last_observation = "Error: Invalid ask action format."
                    budget_info = {"remaining_budget": status.remaining_budget, "total_budget": status.total_budget, "force_submit": status.force_submit}
                    status.add_turn_log(status.parsed_action_object, status.parsed_action_object, status.parsed_action, status.last_observation, 0.0, budget_info)

            # --- Execute based on mode --- #
            if valid_ask_statuses:
                if args.user_sim_mode == 'vanilla':
                    # --- Vanilla Mode --- #
                    logger.debug(f"Building {len(valid_ask_statuses)} prompts for vanilla user sim...")
                    for i, status in enumerate(valid_ask_statuses):
                        question = questions[status.idx]
                        # TODO: Implement `build_vanilla_user_prompt` in prompt_utils.py
                        # For now, using a placeholder simple prompt
                        user_sim_prompt = f"The agent asked: {question}. How would you respond simply?"
                        user_api_prompts.append(user_sim_prompt)
                        api_data_item = {"prompt": user_sim_prompt, "id": status.idx}
                        user_api_data.append(api_data_item)
                        user_prompt_map[i] = status.idx

                    if user_api_data:
                        logger.debug(f"Sending {len(user_api_data)} prompts to user model (vanilla): {args.user_model}")
                        user_api_start_time = time.time()
                        raw_user_output = args.user_output_path if args.user_output_path else f"{args.output_path}.user_vanilla_raw_turn_{current_turn+1}.jsonl"

                        collect_response_from_api(
                            prompt_list=[item['prompt'] for item in user_api_data],
                            model_name=args.user_model,
                            data_list=user_api_data,
                            output_path=raw_user_output,
                            num_threads=args.user_num_threads
                        )
                        user_api_duration = time.time() - user_api_start_time
                        logger.debug(f"Vanilla User API calls completed in {user_api_duration:.2f} seconds.")

                        # Load results
                        try:
                            with jsonlines.open(raw_user_output, mode='r') as reader:
                                for item in reader:
                                    processed_user_responses[item['id']] = item.get("response", "Error: Missing response field")
                        except Exception as e:
                            logger.error(f"Error reading vanilla user output file {raw_user_output}: {e}")
                            # Mark samples with errors
                            for status in valid_ask_statuses:
                                if status.idx not in processed_user_responses:
                                    processed_user_responses[status.idx] = f"Error reading vanilla user response file: {e}"

                elif args.user_sim_mode == 'encoder_decoder':
                    # --- Encoder/Decoder Mode --- #
                    # Step 1: Call Encoder
                    logger.debug(f"Building {len(valid_ask_statuses)} prompts for user encoder...")
                    encoder_api_data = []
                    encoder_prompt_map = {}
                    for i, status in enumerate(valid_ask_statuses):
                        question = questions[status.idx]
                        db_name = status.original_data['selected_database']
                        load_db_data_if_needed(db_name, data_path_base) # Ensure schema is loaded
                        db_schema = _schema_cache.get(db_name, "Schema not available")
                        encoder_prompt = build_user_encoder_prompt(question, status, db_schema)
                        api_data_item = {"prompt": encoder_prompt, "id": status.idx}
                        encoder_api_data.append(api_data_item)
                        encoder_prompt_map[i] = status.idx

                    encoder_results = {}
                    if encoder_api_data:
                        # Use user_model for encoder by default, add --encoder_model arg if needed later
                        encoder_model = args.user_model
                        logger.debug(f"Sending {len(encoder_api_data)} prompts to user encoder model: {encoder_model}")
                        encoder_api_start_time = time.time()
                        raw_encoder_output = args.user_output_path if args.user_output_path else f"{args.output_path}.user_encoder_raw_turn_{current_turn+1}.jsonl"

                        collect_response_from_api(
                            prompt_list=[item['prompt'] for item in encoder_api_data],
                            model_name=encoder_model,
                            data_list=encoder_api_data,
                            output_path=raw_encoder_output,
                            num_threads=args.user_num_threads
                        )
                        encoder_api_duration = time.time() - encoder_api_start_time
                        logger.debug(f"User Encoder API calls completed in {encoder_api_duration:.2f} seconds.")

                        # Load and parse encoder results
                        try:
                            with jsonlines.open(raw_encoder_output, mode='r') as reader:
                                for item in reader:
                                    raw_response = item.get("response", "")
                                    parsed_action = parse_encoder_response(raw_response)
                                    encoder_results[item['id']] = parsed_action # Store {status_idx: parsed_action_label}
                        except Exception as e:
                            logger.error(f"Error reading/parsing encoder output file {raw_encoder_output}: {e}")
                            # Mark samples with errors - they won't proceed to decoder
                            for status in valid_ask_statuses:
                                if status.idx not in encoder_results:
                                    processed_user_responses[status.idx] = f"Error processing encoder response: {e}"

                    # Step 2: Prepare and Call Decoder
                    decoder_api_data = []
                    decoder_prompt_map = {}
                    logger.debug(f"Building prompts for user decoder...")
                    for i, status in enumerate(valid_ask_statuses):
                        if status.idx in encoder_results:
                            question = questions[status.idx]
                            encoder_action = encoder_results[status.idx]
                            status.last_encoder_action = encoder_action # Store for logging/debug

                            db_name = status.original_data['selected_database']
                            load_db_data_if_needed(db_name, data_path_base) # Ensure schema is loaded
                            db_schema = _schema_cache.get(db_name, "Schema not available")
                            decoder_prompt = build_user_decoder_prompt(question, encoder_action, status, db_schema)
                            api_data_item = {"prompt": decoder_prompt, "id": status.idx}
                            decoder_api_data.append(api_data_item)
                            decoder_prompt_map[i] = status.idx
                        # else: Samples that failed encoder step are skipped

                    if decoder_api_data:
                        decoder_model = args.user_model
                        logger.debug(f"Sending {len(decoder_api_data)} prompts to user decoder model: {decoder_model}")
                        decoder_api_start_time = time.time()
                        raw_decoder_output = args.user_output_path if args.user_output_path else f"{args.output_path}.user_decoder_raw_turn_{current_turn+1}.jsonl"

                        collect_response_from_api(
                            prompt_list=[item['prompt'] for item in decoder_api_data],
                            model_name=decoder_model,
                            data_list=decoder_api_data,
                            output_path=raw_decoder_output,
                            num_threads=args.user_num_threads
                        )
                        decoder_api_duration = time.time() - decoder_api_start_time
                        logger.debug(f"User Decoder API calls completed in {decoder_api_duration:.2f} seconds.")

                        # Load decoder results
                        try:
                            with jsonlines.open(raw_decoder_output, mode='r') as reader:
                                for item in reader:
                                     # Ensure we only overwrite if decoder was successful
                                     if item['id'] not in processed_user_responses:
                                         processed_user_responses[item['id']] = item.get("response", "Error: Missing decoder response field")
                        except Exception as e:
                            logger.error(f"Error reading decoder output file {raw_decoder_output}: {e}")
                            # Mark samples that expected a decoder response but didn't get one
                            for status in valid_ask_statuses:
                                if status.idx in decoder_prompt_map and status.idx not in processed_user_responses:
                                    processed_user_responses[status.idx] = f"Error reading decoder response file: {e}"

            # 3.9 Process final user responses (common for both modes)
            logger.debug(f"Processing {len(processed_user_responses)} final user responses for 'ask' actions.")
            for status in valid_ask_statuses:
                 if status.idx in processed_user_responses:
                     user_response = processed_user_responses[status.idx]
                     # Ensure user_response is a string by converting if it's a dictionary
                     status.last_observation = user_response if isinstance(user_response, str) else str(user_response)
                     status.last_user_response = user_response # Store separately if needed
                 else:
                     # This case should ideally be handled by error logging during API calls/file reads
                     status.last_observation = status.last_observation or "Error: User response generation failed."
                     status.last_observation = status.last_observation if isinstance(status.last_observation, str) else str(status.last_observation)

                 # Add budget info to observation
                 budget_prompt = f"\n\n[SYSTEM NOTE: Remaining budget: {status.remaining_budget:.1f}/{status.total_budget:.1f}]"
                 if status.force_submit:
                    budget_prompt += f" Budget depleted. You MUST submit next."
                 status.last_observation += budget_prompt

                 # Log this part of the turn
                 budget_info = {"remaining_budget": status.remaining_budget, "total_budget": status.total_budget, "force_submit": status.force_submit}
                 # Ensure thought, object, action are retrieved from status for logging
                 thought = status.last_agent_response.split("</thought>")[0].split("<thought>")[-1] if status.last_agent_response and "<thought>" in status.last_agent_response else ""
                 obj = status.parsed_action_object
                 action = status.parsed_action
                 status.add_turn_log(thought, obj, action, status.last_observation, 0.0, budget_info)
                 logger.debug(f"Sample {status.idx}: User Action '{status.parsed_action}' -> Final Obs: {status.last_observation[:100]}...")


        # 3.10 Process Submit Actions
        submit_actions = actions_to_process['User_submit']
        if submit_actions:
            logger.debug(f"Executing {len(submit_actions)} Submit actions...")
            for status, sql_to_submit in tqdm(submit_actions, desc="Submit Actions", **tqdm_kwargs):
                 # Retrieve thought/obj/action stored previously
                 thought = status.parsed_thought
                 obj = status.parsed_action_object
                 action = status.parsed_action # The original submit(sql) action string
                 budget_info = {"remaining_budget": status.remaining_budget, "total_budget": status.total_budget, "force_submit": status.force_submit}


                 observation, reward, p1_comp, p2_comp, finished = execute_submit_action(sql_to_submit, status, data_path_base)
                 observation = observation if isinstance(observation, str) else str(observation)
                 status.last_observation = observation if isinstance(observation, str) else str(observation)
                 status.phase1_completed = p1_comp
                 status.phase2_completed = p2_comp
                 if reward == None:
                     reward = 0.0
                 status.last_reward = reward # Store reward received in this turn

                 # **Crucially: If this was a forced submission, the task ends NOW.**
                 if status.force_submit: # Check if force_submit was true *before* this action
                     logger.warning(f"Sample {status.idx}: Task ending due to forced submission attempt.")
                     status.task_finished = True
                 else:
                     # Otherwise, task finishes only if test case passed and indicated completion
                     status.task_finished = finished

                 # Handle phase transition (only relevant if task didn't finish)
                 if not status.task_finished and status.phase1_completed and not status.phase2_completed and status.current_phase == 1:
                      logger.warning(f"Sample {status.idx}: Phase 1 completed, moving to Phase 2.")
                      status.current_phase = 2
                      # Update observation to include the follow-up query for the agent
                      follow_up_query = status.original_data.get("follow_up", {}).get("query")
                      if follow_up_query:
                          status.last_observation += f"\n\nNow, here's a follow-up question: {follow_up_query}"
                      else:
                          # Should not happen if task_finished is False, but handle defensively
                          logger.warning(f"Sample {status.idx}: Phase 1 completed but no follow-up query found. Marking as finished.")
                          status.task_finished = True

                 # Add budget info to observation
                 budget_prompt = f"\n\n[SYSTEM NOTE: Remaining budget: {status.remaining_budget:.1f}/{status.total_budget:.1f}]"
                 if status.force_submit and not status.task_finished: # Should not happen if submit was successful
                      budget_prompt += f" Budget depleted. You MUST submit next."
                 elif status.task_finished:
                      budget_prompt = f"\n\n[SYSTEM NOTE: Task finished. Total reward: {status.last_reward} points.]" # TODO: Calculate total reward

                 status.last_observation += budget_prompt

                 # Log the turn
                 status.add_turn_log(thought, obj, action, status.last_observation, status.last_reward, budget_info)
                 logger.debug(f"Sample {status.idx}: Submit Action -> Obs: {observation[:100]}..., Reward: {reward}, Finished: {finished}")


        # --- End of Turn ---
        # 3.11 Save progress after processing all actions for the turn
        save_progress(args.output_path, all_statuses)

        turn_duration = time.time() - turn_start_time
        logger.warning(f"--- Turn {current_turn + 1} completed in {turn_duration:.2f} seconds ---")
        logger.warning(f"=== Turn {current_turn + 1} Summary ===")
        logger.warning(f"Active samples: {len(active_statuses)}")
        logger.warning(f"Completed samples: {sum(1 for s in all_statuses if s.task_finished)}")
        logger.warning(f"Turn duration: {turn_duration:.2f} seconds")
        current_turn += 1
        # active_samples_exist is updated at the start of the loop

    # --- End of Evaluation Loop ---
    logger.warning("=== Evaluation Complete ===")

    # 4. Calculate and Report Metrics
    logger.warning("Calculating final metrics...")
    metrics = calculate_metrics(all_statuses)
    print_metrics(metrics)
    
    # Save metrics to a separate file
    metrics_path = f"{os.path.splitext(args.output_path)[0]}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.warning(f"Metrics saved to: {metrics_path}")

    # 5. Final Save and Cleanup
    logger.warning("Performing final save...")
    save_progress(args.output_path, all_statuses) # Final save

    logger.warning("Closing database connections...")
    # Need to know which dbs were connected
    connected_dbs = set(s.original_data['selected_database'] for s in all_statuses)
    for db_name in connected_dbs:
        close_db_connection(db_name)

    logger.warning(f"Evaluation complete. Results saved to: {args.output_path}")
    # TODO: Calculate and report final metrics (completion rates, avg turns, etc.)


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation for Bird-Interact environment using parallel API calls.')

    # Input/Output
    parser.add_argument('--data_path', type=str, required=True, help='Path to the Bird-Interact data file (e.g., bird_interact_data.jsonl)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the results and logs (JSONL format)')
    parser.add_argument('--agent_output_path', type=str, help='Optional: Base path for agent API call raw outputs (turn number will be appended)')
    parser.add_argument('--user_output_path', type=str, help='Optional: Base path for user simulator API call raw outputs (turn number will be appended)')

    # Model Configuration
    parser.add_argument('--agent_model', type=str, default='gemini-2.0-flash-001', help='Model name for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-1.5-flash', help='Model name for the user simulator (used for vanilla or decoder)')
    parser.add_argument('--user_sim_mode', type=str, choices=['vanilla', 'encoder_decoder'], default='encoder_decoder', help='User simulator mode')

    # Processing Configuration
    parser.add_argument('--user_num_threads', type=int, default=10, help='Number of threads for parallel API calls')
    parser.add_argument('--num_threads', type=int, default=8, help='Number of threads for parallel API calls')
    parser.add_argument('--max_turns', type=int, default=20, help='Maximum number of interaction turns per sample')
    parser.add_argument('--start_index', type=int, default=0, help='Index of the first sample to process (0-based)')
    parser.add_argument('--start_actual_index', type=int, default=0, help='Start actual index to be assigned to the first sample')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file if it exists')

    # Environment/Budget Configuration
    parser.add_argument('--db_host', type=str, default='bird_interact_postgresql', help='Database host to connect to. If you use docker env to run the code, should set it as `bird_interact_postgresql`. If running in local, should set it as `localhost`.')
    parser.add_argument('--db_port', type=int, default=5432, help='Database port to connect to')
    parser.add_argument('--user_patience_budget', type=int, default=10, help='Initial user patience budget component')

    # Logging Configuration
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (DEBUG level for key modules)')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='WARNING', help='Set the logging level (default: INFO)')
    parser.add_argument('--log_file', type=str, help='Path to log file (default: batch_evaluation.log in current directory)')
    parser.add_argument('--user_sim_prompt_version', type=str, default='v2', choices=['v1', 'v2'], help='User simulator prompt version. v1: preliminary prompt used in our early experiments (used in our bird-interact-lite experiments results). v2 (recommended): more robust prompt for the experiment (used in our bird-interact-full experiments results). You cal also use v2 for bird-interact-lite experiments.')
    
    args = parser.parse_args()

    # Set up logging with the specified configuration
    setup_logging(verbose=args.verbose, log_level=args.log_level, log_file=args.log_file)
    
    # Add a test log message right after setup
    logger = logging.getLogger(__name__)
    logger.warning("=== Starting batch evaluation with configuration ===")
    logger.warning(f"Agent Model: {args.agent_model}")
    logger.warning(f"User Model: {args.user_model} (mode: {args.user_sim_mode})")
    logger.warning(f"Processing {args.limit if args.limit else 'all'} samples starting from index {args.start_index} (actual index: {args.start_actual_index})")
    logger.warning(f"Max turns: {args.max_turns}, Threads: {args.num_threads}")
    logger.warning(f"User patience budget: {args.user_patience_budget}")
    logger.warning(f"Log level: {args.log_level}")
    if args.verbose:
        logger.debug("Verbose logging enabled - will show detailed API and module logs")

    # Set database configuration
    set_global_db_config(host=args.db_host, port=args.db_port)

    try:
        run_batch_evaluation(args)
    finally:
        # Ensure DB config is reset even if errors occur
        reset_global_db_config()
        logger.warning("Global DB config reset.")

if __name__ == '__main__':
    main() 