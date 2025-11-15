import sys
import os
import re
import json
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import necessary prompt templates and parsers
from experiments.utils.prompts import TemplateReActUserBirdInteract
from src.envs.user_simulator.prompts import USER_SIMULATOR_ENCODER, USER_SIMULATOR_DECODER
from src.envs.user_simulator.sql_parser import segment_sql # Assuming this can be imported

# Constants
SETTING_MAP = {
    "bird_interact_sql": "PostgreSQL Database"
}

# Initialize template (assuming 'bird_interact_sql' is the env type)
# This might need to be initialized differently if handling multiple env types
react_template = TemplateReActUserBirdInteract("bird_interact_sql", SETTING_MAP["bird_interact_sql"])

# Cache for segmented SQL
_segment_sql_cache: Dict[str, Any] = {}

# --- Agent Prompt --- #

def build_initial_agent_prompt(sample_status: 'SampleStatus', budget_info: Dict) -> str:
    """Builds the initial prompt for the agent for turn 0."""
    system_prompt = react_template.get_init_msg()
    demos = react_template.get_demos()
    query = sample_status.original_data["amb_user_query"]
    query_msg = react_template.get_query_msg(query)

    budget_prompt = f"\n\n[SYSTEM NOTE: You have a total action budget of {budget_info['total_budget']:.1f} units. Each action consumes budget. If the budget runs out, you must submit.]"

    initial_prompt = system_prompt + demos + query_msg + budget_prompt
    sample_status.current_prompt = initial_prompt # Store the base prompt
    return initial_prompt

def get_agent_prompt_for_turn(sample_status: 'SampleStatus') -> str:
    """Constructs the full agent prompt for the current turn based on history."""
    # The base prompt (query + budget info) should be in sample_status.current_prompt
    # The interaction history is used to build the rest
    return sample_status.get_full_interaction_prompt()

def parse_agent_response(response: str) -> Tuple[str, str, str]:
    """
    Parse the agent's response into thought, interaction object, and action.
    (Copied/adapted from experiments/eval_react_bird_interact.py)
    """
    thought = ""
    interaction_object = ""
    action = ""

    # Extract thought
    thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        # Fallback: Try to find a line starting with Thought:
        lines = response.split('\n')
        for line in lines:
            if line.strip().lower().startswith("thought:"):
                thought = line.split(":", 1)[1].strip()
                break
        if not thought:
            thought = lines[0] if lines else "" # Fallback to first line

    # Extract interaction object
    object_match = re.search(r'<interaction_object>(.*?)</interaction_object>', response, re.DOTALL)
    if object_match:
        interaction_object = object_match.group(1).strip()
    else:
        # Fallback: Infer from action keywords
        if any(kw in response for kw in ["ask(", "submit("]):
            interaction_object = "User"
        elif any(kw in response for kw in ["execute(", "get_schema(", "get_column_meaning(", "get_knowledge_definition("]):
             interaction_object = "Environment"
        else:
             # Default or further inference needed
             interaction_object = "Environment" # Default assumption

    # Extract action
    action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
    else:
        # Fallback: Find the line likely containing the action
        lines = response.split('\n')
        for line in reversed(lines):
            line_stripped = line.strip()
            if line_stripped.startswith(("execute(", "get_schema(", "get_all_column_meanings(",
                                       "get_column_meaning(", "get_all_external_knowledge_names(",
                                       "get_knowledge_definition(", "get_all_knowledge_definitions(",
                                       "ask(", "submit(")):
                action = line_stripped
                break
        if not action:
             action = lines[-1].strip() if lines else "" # Fallback to last line

    # Basic validation/cleanup
    if interaction_object not in ["User", "Environment"]:
        # Attempt to correct based on action
        if action.startswith("ask(") or action.startswith("submit("):
            interaction_object = "User"
        else:
            interaction_object = "Environment"

    return thought, interaction_object, action

# --- User Simulator Prompts (Encoder/Decoder) --- #

def _get_sql_segments(sql: Union[str, List[str]]) -> str:
    """Segments SQL and caches the result."""
    sql_key = sql if isinstance(sql, str) else "\n===\n".join(sql)
    if sql_key in _segment_sql_cache:
        return _segment_sql_cache[sql_key]

    sql_list = [sql] if isinstance(sql, str) else sql
    sql_segments = ""
    for i, s in enumerate(sql_list):
        if i > 0:
            sql_segments += "\n===\n"
        try:
            for clause, text in segment_sql(s):
                sql_segments += clause + ":\n" + text + "\n\n"
        except Exception as e:
            # Fallback for segmentation error
            sql_segments += f"QUERY:\n{s}\n\n" # Treat whole query as one segment
            print(f"Warning: SQL segmentation failed: {e}. Using full query.")

    _segment_sql_cache[sql_key] = sql_segments.strip()
    return _segment_sql_cache[sql_key]

def build_user_encoder_prompt(question: str, sample_status: 'SampleStatus', db_schema: str, user_sim_prompt_version: str = 'v2') -> str:
    """Builds the prompt for the User Simulator Encoder."""
    prompt = USER_SIMULATOR_ENCODER[user_sim_prompt_version].replace('[[clarification_Q]]', question)

    record = sample_status.original_data
    user_query_ambiguity = record.get('user_query_ambiguity', {})
    knowledge_ambiguity = record.get('knowledge_ambiguity', [])

    ambiguity_json = {
        'user_query_ambiguity': user_query_ambiguity,
        'knowledge_ambiguity': knowledge_ambiguity
    }

    # Use phase-specific reference SQL and ambiguity info
    if sample_status.current_phase == 1:
        prompt = prompt.replace('[[amb_json]]', json.dumps(ambiguity_json, indent=2))
        reference_sql = record.get("sol_sql", "")
    else: # Phase 2
        prompt = prompt.replace('[[amb_json]]', json.dumps({}, indent=2)) # No ambiguity in phase 2
        reference_sql = record.get("follow_up", {}).get("sol_sql", "")

    sql_segments = _get_sql_segments(reference_sql)
    prompt = prompt.replace('[[SQL_Glot]]', sql_segments)
    prompt = prompt.replace('[[DB_schema]]', db_schema)

    return prompt

def build_user_decoder_prompt(question: str, encoded_action: str, sample_status: 'SampleStatus', db_schema: str, user_sim_prompt_version: str = 'v2') -> str:
    """Builds the prompt for the User Simulator Decoder."""
    prompt = USER_SIMULATOR_DECODER[user_sim_prompt_version].replace('[[clarification_Q]]', question)
    prompt = prompt.replace('[[Action]]', encoded_action)

    record = sample_status.original_data
    user_query_ambiguity = record.get('user_query_ambiguity', {})
    knowledge_ambiguity = record.get('knowledge_ambiguity', [])
    clear_query = record.get("query", record.get("amb_user_query")) # Fallback to amb query if clear not present

    ambiguity_json = {
        'user_query_ambiguity': user_query_ambiguity,
        'knowledge_ambiguity': knowledge_ambiguity
    }

    # Use phase-specific reference SQL, clear query, and ambiguity info
    if sample_status.current_phase == 1:
        prompt = prompt.replace('[[amb_json]]', json.dumps(ambiguity_json, indent=2))
        reference_sql = record.get("sol_sql", "")
        prompt = prompt.replace('[[clear_query]]', clear_query)
    else: # Phase 2
        prompt = prompt.replace('[[amb_json]]', json.dumps({}, indent=2)) # No ambiguity in phase 2
        reference_sql = record.get("follow_up", {}).get("sol_sql", "")
        # Use follow-up query as the 'clear query' context for phase 2
        follow_up_query = record.get("follow_up", {}).get("query", "")
        prompt = prompt.replace('[[clear_query]]', follow_up_query)

    prompt = prompt.replace('[[GT_SQL]]', reference_sql if isinstance(reference_sql, str) else '\n'.join(reference_sql))
    sql_segments = _get_sql_segments(reference_sql)
    prompt = prompt.replace('[[SQL_Glot]]', sql_segments)
    prompt = prompt.replace('[[DB_schema]]', db_schema)

    return prompt

def parse_encoder_response(response: str) -> str:
    """Extracts the action label (e.g., labeled(...)) from the encoder response."""
    # Look for <s>...</s> tags first
    s_match = re.search(r'(?:<s>)?(.*?)</s>', response, re.DOTALL)
    if s_match:
        action_part = s_match.group(1).strip()
        # Further clean if needed, e.g., remove justifications outside the action
        action_match = re.match(r'(labeled\(.*?\)|unlabeled\(\)|unanswerable\()\)?', action_part, re.IGNORECASE)
        if action_match:
            return action_match.group(1) # Return the core action
        else:
            # Fallback if structure inside <s> is weird
             return action_part if action_part else "unanswerable()"
        
    # Fallback if no <s> tags
    # Look for lines starting with the expected actions
    lines = response.strip().split('\n')
    for line in lines:
        line_clean = line.strip()
        action_match = re.match(r'(labeled\(.*?\)|unlabeled\(\)|unanswerable\()\)?', line_clean, re.IGNORECASE)
        if action_match:
            return action_match.group(1)

    # Final fallback
    logger.warning(f"Could not parse encoder action from response: {response[:100]}... Defaulting to unanswerable()")
    return "unanswerable()" 


if __name__ == "__main__":
    # txt = "<thought>\nOkay, the user wants the Modulation Complexity Score (MCS) and the Signal-to-Noise Ratio (SNR) as signal details. They also clarified that \"several occurrences\" means more than 5 signals.\n\nNow I have a better understanding of the request. I need to:\n\n1.  Identify the tables containing signal encoding methods, MCS, SNR, complexity measure, and quality ratio.\n2.  Join the necessary tables.\n3.  Group the results by signal encoding method type.\n4.  Calculate the count, average complexity measure, and average quality ratio for each method.\n5.  Include the MCS and SNR in the output.\n6.  Filter for methods with more than 5 signals.\n\nLet's start by exploring the database schema to identify the relevant tables and columns.\n</thought>\n<interaction_object>Environment</interaction_object>\n<action>get_schema()</action>\n\nObservation: ```\nDatabase schema:\nCREATE TABLE Signals (\nSignalID INTEGER PRIMARY KEY,\nSignalType VARCHAR(50),\nSignalStrength DECIMAL(10, 2),\nFrequency DECIMAL(10, 2),\nEncodingMethod VARCHAR(50),\nComplexityMeasure INTEGER,\nQualityRatio DECIMAL(10, 2),\nMCS INTEGER,\nSNR DECIMAL(10, 2)\n);\nCREATE TABLE Observatories (\nObservID INTEGER PRIMARY KEY,\nObservStation VARCHAR(50),\nObservDate DATE,\nLunarStage VARCHAR(20),\nSignal INTEGER,\nFOREIGN KEY (Signal) REFERENCES Signals(SignalID)\n);\n```\n\n[SYSTEM NOTE: Remaining budget: 19.0/22.0]\n"
    txt = "<thought>\nBased on your clarifications:\n1. We will compute the score using the formula:\n   p.TechSigProb * (1 - p.NatSrcProb) * p.SigUnique * (0.5 + p.AnomScore/10)\n2. The classification is:\n   • 'Low' if the computed score is less than 0.25\n   • 'Medium' if the computed score is between 0.25 and 0.75 (i.e., less than 0.75)\n   • 'High' if the computed score is greater than or equal to 0.75.\n3. The BFR measure is computed from the Signals table as:\n   s.BwHz / (s.CenterFreqMhz * 1000000)\n4. The anomaly metric is the standard deviation of p.AnomScore.\n5. We assume these two tables (SignalProbabilities and Signals) are related by a common column (assumed here as SignalID).\n\nBelow is the SQL query that calculates the classification, counts the signals in each classification group, computes the average BFR measure, and the standard deviation of the anomaly metric.\n\n</thought>\n<interaction_object>User</interaction_object>\n<action>\nsubmit(\"SELECT \\\n  CASE \\\n    WHEN (p.TechSigProb * (1 - p.NatSrcProb) * p.SigUnique * (0.5 + p.AnomScore/10)) < 0.25 THEN 'Low' \\\n    WHEN (p.TechSigProb * (1 - p.NatSrcProb) * p.SigUnique * (0.5 + p.AnomScore/10)) < 0.75 THEN 'Medium' \\\n    ELSE 'High' \\\n  END AS classification, \\\n  COUNT(*) AS signal_count, \\\n  AVG(s.BwHz / (s.CenterFreqMhz * 1000000)) AS avg_bfr_measure, \\\n  STDDEV(p.AnomScore) AS anomaly_std_dev \\\nFROM SignalProbabilities p \\\nJOIN Signals s ON p.SignalID = s.SignalID \\\nGROUP BY classification;\")\n</action>"
    # print(txt)
    thought, obj, action = parse_agent_response(txt)
    print("thought: ", thought)
    print("obj: ", obj)
    print("action: ", action)


