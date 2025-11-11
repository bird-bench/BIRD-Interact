import sys
import os
import jsonlines  # Add this import

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse, re
from src.envs import (
    BirdInteractSqlEnv, ACTION_EXEC
)
from src.envs.user_simulator.us_env_bird_interact import UserSimulatorBirdInteractEnv
from src.envs.bird_interact_env.token_counter import token_counter
from tqdm import tqdm
from typing import Dict, List, Tuple
from experiments.utils import TemplateReActUserBirdInteract
from src.llm_utils.llm_provider import LLMProvider
from src.llm_utils.human_agent import HumanAgent
from src.llm_utils.llm_agent import LLMAgent
from src.config.db_config import set_global_db_config, reset_global_db_config
from tqdm import tqdm


SETTING_MAP = {
    "bird_interact_sql": "PostgreSQL Database",
}


parser = argparse.ArgumentParser(description='ReAct evaluation for BIRD-Interact environment')
parser.add_argument('--data_path', type=str, help='path to dataset to evaluate on')
parser.add_argument('--env', choices=['bird_interact_sql'], help='BIRD-Interact environment to run eval on')
parser.add_argument('--log_dir', type=str, help='folder to save experiment run log file to')
parser.add_argument('--max_turns', type=int, help='max number of interaction turns')
parser.add_argument('--verbose', action='store_true', help="print out logs")
parser.add_argument('--agent_model', type=str, help='model to use for generation', default="gpt-3.5-turbo")
parser.add_argument('--user_model', type=str, help='model to use for user simulator', default="gpt-3.5-turbo")
parser.add_argument('--user_model_provider', type=str, default='openai', help='LLM provider to use')
parser.add_argument('--agent_model_provider', type=str, default='openai', help='LLM provider to use')
parser.add_argument('--kb_path', type=str, help='path to knowledge base to use for sql_amb environment')
parser.add_argument('--env_interact_budget', type=int, help='budget for environment interactions', default=3)
parser.add_argument('--submit_budget', type=int, help='budget for SQL submissions', default=3)
parser.add_argument('--user_patience_budget', type=int, help='user patience budget', default=6)
parser.add_argument('--use_encoder_decoder', action='store_true', help='whether to use encoder-decoder approach for user simulator')
parser.add_argument('--debug_mode', action='store_true', help='print debug information for encoder-decoder')
parser.add_argument('--debug_num', type=int, help='number of examples to debug', default=None)
parser.add_argument('--debug_sol_sql', action='store_true', help='debug with gold SQL query')
parser.add_argument('--human_mode', action='store_true', help='Use human agent instead of LLM')
parser.add_argument('--resume', action='store_true', help='Resume from previous run')
parser.add_argument('--db_host', type=str, help='Database host to connect to', default='bird_interact_postgresql')
parser.add_argument('--db_port', type=int, help='Database port to connect to', default=5432)
args = parser.parse_args()

# Initialize LLM provider
agent_llm_provider = LLMProvider(
    provider=args.agent_model_provider,
    model_id=args.agent_model,
    token_counter=token_counter
)
user_llm_provider = LLMProvider(
    provider=args.user_model_provider,
    model_id=args.user_model,
    token_counter=token_counter
)
print(f"Agent model: {args.agent_model_provider.capitalize()} {args.agent_model}")
print(f"User model: {args.user_model_provider.capitalize()} {args.user_model}")

class BudgetTracker:
    """Class to track and manage the agent cost budget."""
    
    def __init__(self, 
                 env_interact_budget=3, 
                 submit_budget=3, 
                 amb_resolve_budget=0,  # Will be calculated based on ambiguities
                 user_patience_budget=10):
        self.env_interact_budget = env_interact_budget
        self.submit_budget = submit_budget
        self.amb_resolve_budget = amb_resolve_budget
        self.user_patience_budget = user_patience_budget
        
        # Calculate total budget
        self.total_budget = (
            self.env_interact_budget + 
            self.submit_budget + 
            self.amb_resolve_budget + 
            self.user_patience_budget
        )
        
        # Initialize trackers
        self.remaining_budget = self.total_budget
        self.env_interact_used = 0
        self.submit_used = 0
        self.amb_resolve_used = 0
        self.user_patience_used = 0
        self.force_submit = False  # Flag to indicate that next action must be a submission
        
        # Cost mapping
        self.action_costs = {
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
    
    def calculate_amb_resolve_budget(self, amb_data):
        """Calculate ambiguity resolution budget based on ambiguities in data"""
        amb_count = 0
        
        # Count critical ambiguities in user query
        if "user_query_ambiguity" in amb_data:
            if "critical_ambiguity" in amb_data["user_query_ambiguity"]:
                amb_count += len(amb_data["user_query_ambiguity"]["critical_ambiguity"])
        
        # Count knowledge ambiguities
        if "knowledge_ambiguity" in amb_data:
            amb_count += len(amb_data["knowledge_ambiguity"])
        
        # Each ambiguity costs 2 budget units
        self.amb_resolve_budget = amb_count * 2
        
        # Recalculate total budget
        self.total_budget = (
            self.env_interact_budget + 
            self.submit_budget + 
            self.amb_resolve_budget + 
            self.user_patience_budget
        )

        if args.verbose:
            print(f"""
                Total budget: {self.total_budget}
                Remaining budget: {self.remaining_budget}
                Env interact budget: {self.env_interact_budget}
                Submit budget: {self.submit_budget}
                Amb resolve budget: {self.amb_resolve_budget}
                User patience budget: {self.user_patience_budget}
                """)
        
        self.remaining_budget = self.total_budget
        
        return self.amb_resolve_budget
    
    def update_budget(self, action):
        """Update budget based on action taken"""
        action_type = action.split("(")[0] if "(" in action else action
        cost = self.action_costs.get(action_type, 0)
        
        # If we're in force_submit mode, only allow submit actions
        if self.force_submit:
            if action_type != "submit":
                return self.remaining_budget  # Don't update budget, just return current value
            # For submit actions in force_submit mode, allow it regardless of remaining budget
            self.submit_used += cost
            self.remaining_budget -= cost
            return self.remaining_budget
            
        # Track budget usage by category
        if action_type in ["execute", "get_schema", "get_all_column_meanings", 
                          "get_column_meaning", "get_all_external_knowledge_names",
                          "get_knowledge_definition", "get_all_knowledge_definitions"]:
            self.env_interact_used += cost
        elif action_type == "submit":
            self.submit_used += cost
        elif action_type == "ask":
            self.user_patience_used += cost
            
        # Deduct from remaining budget
        self.remaining_budget -= cost

        return self.remaining_budget
    
    def get_budget_info(self):
        """Get current budget information"""
        return {
            "total_budget": self.total_budget,
            "remaining_budget": self.remaining_budget
        }
    
    def is_depleted(self):
        """Check if budget is depleted"""
        return self.remaining_budget <= 0

class ExperimentWrapper():
    def __init__(self, args):
        self.args = args

        # Set environments (No logging for env)
        self.env = None
        self.user_env = None
        if args.env == 'bird_interact_sql':
            self.env = BirdInteractSqlEnv(data_path=args.data_path, kb_path=args.kb_path, db_port=args.db_port)
        else:
            raise ValueError(f'Environment {args.env} not recognized')
        
        # Initialize agent (either human or LLM)
        if args.human_mode:
            self.agent = HumanAgent(verbose=True)
            print("Running in human agent mode")
        else:
            self.agent = LLMAgent(
                llm_provider=agent_llm_provider,
                model_id=args.agent_model
            )
            print(f"Running with {args.agent_model_provider.capitalize()} {args.agent_model}")
        
        self.user_env = UserSimulatorBirdInteractEnv(
            llm_provider=user_llm_provider,
            model_id=args.user_model,
            use_encoder_decoder=args.use_encoder_decoder,
            debug_mode=args.debug_mode
        )
        
        # Initialize budget tracker
        self.budget_tracker = BudgetTracker(
            env_interact_budget=args.env_interact_budget,
            submit_budget=args.submit_budget,
            user_patience_budget=args.user_patience_budget
        )
            
        # Connect SQL environment to user environment
        if args.env == 'bird_interact_sql':
            self.user_env.set_sql_env(self.env)
        
        # Define log file name and path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        
        # Include debug_num in log filename if provided
        # debug_suffix = f"_debug{args.debug_num}" if args.debug_num is not None else ""
        # log_file_name = f"{args.env}_react_{args.max_turns}_turns_user_patience_{args.user_patience_budget}{debug_suffix}.jsonl"
        log_file_name = f"{args.env}_react_{args.max_turns}_turns_user_patience_{args.user_patience_budget}.jsonl"
        self.log_path = os.path.join(args.log_dir, log_file_name)
        print(f"Log file: {self.log_path}")
        
        # Initialize metrics
        self.metrics = {
            "num_examples": 0,
            "num_solved": 0,
            "phase1_complete": 0,
            "phase2_complete": 0,
            "total_tokens": 0,
            "token_breakdown": {
                "system_input": 0,
                "system_output": 0,
                "user_simulator_input": 0,
                "user_simulator_output": 0
            }
        }
        
        # Load existing results if resuming
        self.completed_indices = set()
        if args.resume and os.path.exists(self.log_path):
            print(f"Resuming from {self.log_path}")
            with jsonlines.open(self.log_path, mode='r') as reader:
                for record in reader:
                    if "idx" in record:
                        self.completed_indices.add(record["idx"])
                        # Update metrics
                        self.metrics["num_examples"] += 1
                        if record.get("reward", 0) > 0:
                            self.metrics["num_solved"] += 1
                        if record.get("phase1_completed", False):
                            self.metrics["phase1_complete"] += 1
                        if record.get("phase2_completed", False):
                            self.metrics["phase2_complete"] += 1
                        if "token_usage" in record:
                            self.metrics["total_tokens"] += sum(record["token_usage"].values())
                            for k, v in record["token_usage"].items():
                                self.metrics["token_breakdown"][k] += v
            print(f"Loaded {len(self.completed_indices)} completed examples")
        
        # Open log file for writing
        self.log_writer = jsonlines.open(self.log_path, mode='a')
        
        # Initialize prompt template
        self.template = TemplateReActUserBirdInteract(args.env, SETTING_MAP[args.env])

    def _parse_action_bird_interact(self, action_str: str) -> Tuple[str, bool]:
        """
        Parse the action string into a format that can be executed by the environment.
        Returns (parsed_action, is_valid)
        """
        # Extract parameters from the action
        if action_str.startswith("execute("):
            # Handle SQL commands
            match = re.search(r'execute\((.*)\)', action_str, re.DOTALL)
            if match:
                sql = match.group(1).strip().strip("'\"")
                return f"execute({sql})", True
            return "", False
        elif action_str.startswith("get_schema("):
            return "get_schema()", True
        elif action_str.startswith("get_all_column_meanings("):
            return "get_all_column_meanings()", True
        elif action_str.startswith("get_column_meaning("):
            match = re.search(r'get_column_meaning\((.*)\)', action_str)
            if match:
                params = match.group(1).strip()
                return f"get_column_meaning({params})", True
            return "", False
        elif action_str.startswith("get_all_external_knowledge_names("):
            return "get_all_external_knowledge_names()", True
        elif action_str.startswith("get_knowledge_definition("):
            match = re.search(r'get_knowledge_definition\((.*)\)', action_str)
            if match:
                param = match.group(1).strip()
                return f"get_knowledge_definition({param})", True
            return "", False
        elif action_str.startswith("get_all_knowledge_definitions("):
            return "get_all_knowledge_definitions()", True
        elif action_str.startswith("ask("):
            match = re.search(r'ask\((.*)\)', action_str, re.DOTALL)
            if match:
                question = match.group(1).strip()
                return f"ask({question})", True
            return "", False
        elif action_str.startswith("submit("):
            match = re.search(r'submit\((.*)\)', action_str, re.DOTALL)
            if match:
                sql = match.group(1).strip()
                return f"submit({sql})", True
            return "", False
        else:
            return action_str, False

    def parse_agent_response(self, response: str) -> Tuple[str, str, str]:
        """
        Parse the agent's response into thought, interaction object, and action.
        
        Args:
            response: The agent's response string
            
        Returns:
            Tuple[str, str, str]: (thought, interaction_object, action)
        """
        # Extract thought
        thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = response.split("\n")[0]  # Fallback to first line
            
        # Extract interaction object
        object_match = re.search(r'<interaction_object>(.*?)</interaction_object>', response, re.DOTALL)
        if object_match:
            interaction_object = object_match.group(1).strip()
        else:
            # Improved detection of interaction object from context
            if "ask(" in response or "submit(" in response:
                interaction_object = "User"
            else:
                interaction_object = "Environment"  # Default to Environment
            
        # Extract action
        action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
        else:
            # Try to extract action from response
            for line in response.split("\n"):
                if line.strip().startswith(("execute(", "get_schema(", "get_all_column_meanings(", 
                                          "get_column_meaning(", "get_all_external_knowledge_names(",
                                          "get_knowledge_definition(", "get_all_knowledge_definitions(",
                                          "ask(", "submit(")):
                    action = line.strip()
                    break
            else:
                action = response.split("\n")[-1]  # Fallback to last line
            
        return thought, interaction_object, action

    def _update_metrics(self, record):
        """Update metrics with a new record."""
        self.metrics["num_examples"] += 1
        if record.get("reward", 0) > 0:
            self.metrics["num_solved"] += 1
        if record.get("phase1_completed", False):
            self.metrics["phase1_complete"] += 1
        if record.get("phase2_completed", False):
            self.metrics["phase2_complete"] += 1
        if "token_usage" in record:
            self.metrics["total_tokens"] += sum(record["token_usage"].values())
            for k, v in record["token_usage"].items():
                self.metrics["token_breakdown"][k] += v
    
    def _print_metrics(self):
        """Print current metrics."""
        num_examples = self.metrics["num_examples"]
        if num_examples == 0:
            return
            
        print("\nCurrent Metrics:")
        print(f"Processed {num_examples} examples")
        print(f"Solved {self.metrics['num_solved']}/{num_examples} examples ({self.metrics['num_solved']/num_examples:.2f})")
        print(f"Phase 1 completion: {self.metrics['phase1_complete']}/{num_examples} examples ({self.metrics['phase1_complete']/num_examples:.2f})")
        print(f"Phase 2 completion: {self.metrics['phase2_complete']}/{num_examples} examples ({self.metrics['phase2_complete']/num_examples:.2f})")
        print(f"Total tokens: {self.metrics['total_tokens']}")
        print(f"Average tokens per example: {self.metrics['total_tokens']/num_examples:.2f}")
        print(f"Token breakdown: {self.metrics['token_breakdown']}")
        print()

    def run_expr(self):
        try:
            # Get the range of examples to process
            total_examples = len(self.env.data_loader)
            if self.args.debug_num is not None:
                num_examples = min(self.args.debug_num, total_examples)
                print(f"Debug mode: Processing only {num_examples} examples out of {total_examples}")
            else:
                num_examples = total_examples
            
            # Evaluation loop
            for idx in tqdm(range(num_examples), disable=self.args.verbose, desc="Processing examples"):
                try:
                    # Skip if already completed and resuming
                    if idx in self.completed_indices:
                        continue
                    
                    turn_history = {
                        "idx": idx, 
                        "thoughts": [], 
                        "interactions": [], 
                        "actions": [],
                        "observations": [], 
                        "rewards": [],
                        "valid_action": [],
                        "remaining_budget": [],
                        "encode_decode_info": []  # New field to store encoder-decoder info
                    }
                    
                    # Reset token counter for this example
                    token_counter.reset()
                    
                    # Reset environments
                    observation = self.env.reset(idx)
                    
                    # Connect SQL environment to user environment
                    self.user_env.set_sql_env(self.env)
                    
                    # Initialize user simulator with current example
                    record = self.env.data_loader.get(idx)
                    
                    # Calculate ambiguity budget based on this record
                    self.budget_tracker.calculate_amb_resolve_budget(record)
                    user_query_ambiguity = record.get("user_query_ambiguity", {})
                    knowledge_ambiguity = record.get("knowledge_ambiguity", [])
                    clarification_json = {
                        "user_query_ambiguity": user_query_ambiguity,
                        "knowledge_ambiguity": knowledge_ambiguity
                    }
                    self.user_env.reset(
                        ambiguous_query=record["amb_user_query"],
                        clarification_json=clarification_json,
                        reference_sql=record["sol_sql"],
                        follow_up_query=record.get("follow_up", {}).get("query") if "follow_up" in record else None,
                        clear_query=record.get("query"),  # Add clear query if available
                        follow_up_reference_sql=record.get("follow_up", {}).get("sol_sql") if "follow_up" in record else None
                    )
                    
                    # Initialize prompt with observation
                    system_prompt = self.template.get_init_msg()
                    prompt = system_prompt + self.template.get_demos() + self.template.get_query_msg(self.user_env.ambiguous_query)

                    # Add budget information to the prompt
                    budget_info = self.budget_tracker.get_budget_info()
                    budget_prompt = f"\n\n  [SYSTEM NOTE: You have a total action budget of {budget_info['total_budget']} units:\n"
                    budget_prompt += "Each action you take will consume budget. Once your budget is depleted, you must submit your final SQL solution.\n]"
                    
                    prompt += budget_prompt

                    if self.args.verbose:
                        print(f"Input observation: {observation}")
                        print("-" * 50)
                    
                    # Set up logging for this example
                    record_log = {}
                    record_log["idx"] = idx
                    record_log["observation"] = observation
                    record_log["selected_database"] = self.env.data_loader.get(idx)["selected_database"]
                    record_log["interaction_history"] = []
                    
                    # Maximum turns
                    reward = 0
                    valid_action = True
                    done = False
                    
                    # Interaction loop
                    for turn_idx in range(self.args.max_turns):
                        if self.args.verbose:
                            print(f"Turn {turn_idx + 1}")
                        
                        # Generate agent response (either from human or LLM)
                        model_input = prompt
                        if self.args.debug_sol_sql:
                            if not self.user_env.phase1_completed:
                                sol_sql = "\n".join(record['sol_sql']) if isinstance(record['sol_sql'], list) else record['sol_sql']
                                completion = "<object>user</object><action>submit(" + sol_sql + ")</action>"
                            else:
                                sol_sql = "\n".join(record['follow_up']['sol_sql']) if isinstance(record['follow_up']['sol_sql'], list) else record['follow_up']['sol_sql']
                                completion = "<object>user</object><action>ask('Hi! To better answer your question, could you tell me what do you mean of AOI?')</action>"
                        else:
                            completion = self.agent.get_response(model_input, system_prompt)
                        
                        response = completion
                        
                        if self.args.verbose:
                            print(f"Response: {response}")
                        
                        # Track token usage for agent interaction
                        token_counter.add_system_input(model_input)
                        token_counter.add_system_output(response)

                        # Parse response
                        thought, interaction_object, action = self.parse_agent_response(response)
                        
                        if self.args.verbose:
                            print(f"Turn {turn_idx + 1}:")
                            print(f"Thought: {thought}")
                            print(f"Interaction: {interaction_object}")
                            print(f"Action: {action}")

                        encode_decode_info = {"turn": turn_idx + 1}
                        
                        if interaction_object == "Environment":
                            # Parse action + execute in environment
                            action_parsed, is_valid = self._parse_action_bird_interact(action)
                            if not is_valid:
                                observation = f"Error: Invalid action format for Environment: {action}"
                                valid_action = False
                            else:
                                # Count tokens for system input when interacting with environment
                                token_counter.add_system_input(action_parsed)
                                
                                # Update budget
                                self.budget_tracker.update_budget(action)
                                
                                # Execute action
                                observation, reward, done, info = self.env.step(action_parsed)
                                valid_action = info[ACTION_EXEC]
                                
                                # Count tokens for system output (environment's response)
                                token_counter.add_system_output(observation)
                                
                                # Add budget information to observation
                                budget_info = self.budget_tracker.get_budget_info()
                                if self.budget_tracker.is_depleted() and not self.budget_tracker.force_submit:
                                    self.budget_tracker.force_submit = True
                                    observation += f"\n\n[SYSTEM NOTE: Your action budget is depleted ({budget_info['remaining_budget']}/{budget_info['total_budget']}). You must submit your final SQL query in the next turn.]"
                                elif self.budget_tracker.force_submit:
                                    # If this is not a submit action after force_submit was set
                                    if not action.startswith("submit("):
                                        observation += f"\n\n[SYSTEM NOTE: Your action budget is depleted. You MUST submit your final SQL query now.]"
                                    else:
                                        observation += f"\n\n[SYSTEM NOTE: Forced submission has been submitted. The task is done.]"
                                        done = True
                                else:
                                    observation += f"\n\n[SYSTEM NOTE: Remaining action budget: {budget_info['remaining_budget']}/{budget_info['total_budget']}]"
                                
                        elif interaction_object == "User":
                            # Parse action + send to user simulator
                            action_parsed, is_valid = self._parse_action_bird_interact(action)
                            if not is_valid:
                                observation = f"Error: Invalid action format for User: {action}"
                                valid_action = False
                            else:
                                # Update budget
                                self.budget_tracker.update_budget(action)
                                
                                # Store encoder-decoder info if asking a question with encoder-decoder approach
                                if action_parsed.startswith("ask(") and self.args.use_encoder_decoder:
                                    # Execute user step (this will internally use decode_response)
                                    response, reward, done = self.user_env.step(action_parsed)
                                    encode_decode_info["encoded_action"] = self.user_env.info.get("encoded_action", "")
                                    
                                    # Clean the response by removing system notes
                                    if "\n\n[SYSTEM NOTE" in response:
                                        clean_response = response.split("\n\n[SYSTEM NOTE")[0]
                                    else:
                                        clean_response = response
                                    
                                    # Add budget info to observation
                                    budget_info = self.budget_tracker.get_budget_info()
                                    if self.budget_tracker.is_depleted() and not self.budget_tracker.force_submit:
                                        if not action.startswith("submit("):
                                            self.budget_tracker.force_submit = True
                                            observation = f"{clean_response}\n\n[SYSTEM NOTE: Your action budget is depleted ({budget_info['remaining_budget']}/{budget_info['total_budget']}). You must submit your final SQL query in the next turn.]"
                                        else:
                                            observation = f"{clean_response}\n\n[SYSTEM NOTE: The agent has already submitted the final SQL query. The task is done.]"
                                            done = True
                                    elif self.budget_tracker.force_submit:
                                        # If this is not a submit action after force_submit was set
                                        if not action.startswith("submit("):
                                            observation = f"{clean_response}\n\n[SYSTEM NOTE: Your action budget is depleted. You MUST submit your final SQL query now.]"
                                        else:
                                            observation = f"{clean_response}\n\n[SYSTEM NOTE: Forced submission has been submitted. The task is done.]"
                                            done = True
                                    else:
                                        observation = f"{clean_response}\n\n[SYSTEM NOTE: Remaining action budget: {budget_info['remaining_budget']}/{budget_info['total_budget']}]"
                                else:
                                    # Execute regular step
                                    response, reward, done = self.user_env.step(action_parsed)
                                    
                                    # Clean the response by removing system notes
                                    if "\n\n[SYSTEM NOTE" in response:
                                        clean_response = response.split("\n\n[SYSTEM NOTE")[0]
                                    else:
                                        clean_response = response
                                    
                                    # Add budget info to observation
                                    budget_info = self.budget_tracker.get_budget_info()
                                    if self.budget_tracker.is_depleted() and not self.budget_tracker.force_submit:
                                        if not action.startswith("submit("):
                                            self.budget_tracker.force_submit = True
                                            observation = f"{clean_response}\n\n[SYSTEM NOTE: Your action budget is depleted ({budget_info['remaining_budget']}/{budget_info['total_budget']}). You must submit your final SQL query in the next turn.]"
                                        else:
                                            observation = f"{clean_response}\n\n[SYSTEM NOTE: The agent has already submitted the final SQL query. The task is done.]"
                                            done = True
                                    elif self.budget_tracker.force_submit:
                                        # If this is not a submit action after force_submit was set
                                        if not action.startswith("submit("):
                                            observation = f"{clean_response}\n\n[SYSTEM NOTE: Your action budget is depleted. You MUST submit your final SQL query now.]"
                                        else:
                                            observation = f"{clean_response}\n\n[SYSTEM NOTE: Forced submission has been submitted. The task is done.]"
                                            done = True
                                    else:
                                        observation = f"{clean_response}\n\n[SYSTEM NOTE: Remaining action budget: {budget_info['remaining_budget']}/{budget_info['total_budget']}]"
                        else:
                            observation = f"Error: Invalid interaction object: {interaction_object}"
                            valid_action = False

                        # Save encode_decode_info for this turn
                        turn_history["encode_decode_info"].append(encode_decode_info)
                        
                        # Update Prompt with latest turn information
                        turn_info = f"""<thought>{thought}</thought>
<interaction_object>{interaction_object}</interaction_object>
<action>{action}</action>

Observation: {observation}

"""
                        prompt += turn_info
                        
                        if self.args.verbose:
                            print(f"Observation: {observation}")
                            print(f"Reward: {reward}")
                            print(f"Valid action: {valid_action}")
                            print("-" * 50)
                        
                        # Logging
                        turn_history["thoughts"].append(thought)
                        turn_history["interactions"].append(interaction_object)
                        turn_history["actions"].append(action)
                        turn_history["observations"].append(str(observation))
                        turn_history["rewards"].append(reward)
                        turn_history["valid_action"].append(valid_action)
                        turn_history["remaining_budget"].append(self.budget_tracker.get_budget_info())
                        
                        # Add to interaction history
                        record_log["interaction_history"].append({
                            "turn": turn_idx + 1,
                            "thought": thought,
                            "interaction_object": interaction_object,
                            "action": action,
                            "observation": str(observation),
                            "reward": reward,
                            "valid_action": valid_action,
                            "budget": self.budget_tracker.get_budget_info(),
                            "encode_decode_info": encode_decode_info if "encoded_action" in encode_decode_info else {}
                        })
                        
                        # Break the loop if we're done, but only after logging the current turn
                        if done:
                            break
                    
                    # End of example evaluation
                    if self.args.verbose:
                        print(f"Final reward: {reward}")
                        print(f"Done: {done}")
                        print(f"Token usage: {token_counter.summary()}")
                        print("=" * 50)
                    
                    # Get phase completion directly from user simulator
                    phase1_completed = self.user_env.phase1_completed
                    phase2_completed = self.user_env.phase2_completed
                    
                    # Calculate reward based solely on phase completion
                    total_reward = 0
                    if phase1_completed:
                        total_reward += self.user_env.phase_rewards[1]
                    if phase2_completed:
                        total_reward += self.user_env.phase_rewards[2]
                    
                    # Get final budget information
                    final_budget_info = self.budget_tracker.get_budget_info()
                    
                    # Log results
                    record_log["reward"] = total_reward  # Based on phase completion
                    record_log["done"] = done
                    record_log["token_usage"] = token_counter.get_counts()
                    record_log["phase1_completed"] = phase1_completed
                    record_log["phase2_completed"] = phase2_completed
                    record_log["total_reward"] = total_reward
                    record_log["budget"] = final_budget_info
                    record_log["error"] = None  # No error occurred
                    
                    # Write to JSONL file
                    self.log_writer.write(record_log)
                    
                    # Update metrics
                    self._update_metrics(record_log)
                    
                    # Print current metrics periodically
                    if (idx + 1) % 5 == 0:  # Print every 5 examples
                        self._print_metrics()
                    
                    # Reset budget tracker for next example
                    self.budget_tracker = BudgetTracker(
                        env_interact_budget=self.args.env_interact_budget,
                        submit_budget=self.args.submit_budget,
                        user_patience_budget=self.args.user_patience_budget
                    )

                except Exception as e:
                    # Log the error and continue with next sample
                    error_msg = f"Error processing sample {idx}: {str(e)}"
                    print(error_msg)
                    import traceback
                    error_traceback = traceback.format_exc()
                    print(error_traceback)
                    
                    # Create error log entry
                    error_log = {
                        "idx": idx,
                        "error": error_msg,
                        "error_traceback": error_traceback,
                        "done": False,
                        "reward": 0,
                        "phase1_completed": False,
                        "phase2_completed": False,
                        "total_reward": 0,
                        "token_usage": token_counter.get_counts() if 'token_counter' in locals() else {},
                        "budget": self.budget_tracker.get_budget_info() if 'self.budget_tracker' in locals() else None
                    }
                    
                    # Write error log to file
                    self.log_writer.write(error_log)
                    
                    # Update metrics with failed example
                    self.metrics["num_examples"] += 1
                    if "token_usage" in error_log:
                        self.metrics["total_tokens"] += sum(error_log["token_usage"].values())
                        for k, v in error_log["token_usage"].items():
                            self.metrics["token_breakdown"][k] += v
                    
                    # Print current metrics after error
                    self._print_metrics()
                    
                    # Reset budget tracker for next example
                    self.budget_tracker = BudgetTracker(
                        env_interact_budget=self.args.env_interact_budget,
                        submit_budget=self.args.submit_budget,
                        user_patience_budget=self.args.user_patience_budget
                    )
                    continue

            # Close log writer
            self.log_writer.close()
            
            # Print final metrics
            self._print_metrics()

            print("File saved to ", self.log_path)
            
            return self.metrics
            
        except Exception as e:
            print(f"Error in experiment: {e}")
            import traceback
            traceback.print_exc()
            # Make sure to close the log writer even if there's an error
            if hasattr(self, 'log_writer'):
                self.log_writer.close()
            return None

if __name__ == '__main__':
    # Set global database configuration based on command line arguments
    # For local development, use localhost, otherwise use container hostname
    # host = 'localhost' if using local environment else 'bird_interact_postgresql'
    set_global_db_config(host=args.db_host, port=args.db_port)
    
    try:
        expr_wrapper = ExperimentWrapper(args)
        expr_wrapper.run_expr()
    finally:
        # Reset global configuration after experiment
        reset_global_db_config()