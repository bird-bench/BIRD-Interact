import math
import psycopg2
import json
import os
import numpy as np
import re
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
import queue

from collections import Counter
from itertools import chain, groupby
from operator import itemgetter
from scipy.stats import kendalltau
from typing import Dict, List, Tuple, Union

from src.envs.ic_env import (
    BaseEnv,
    AGENT_OBS, EVAL_OBS, CORRUPT_GOLD, ACTION_EXEC, REWARD
)


import ast


def strip_outer_quotes(s: str) -> str:
    """Remove one matching pair of outer quotes (triple or single) from a string."""
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        return s[3:-3]
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def parse_action_arg(action: str, prefix: str) -> str:
    """
    Extract the string argument from action text like 'execute("...")'.
    """
    match = re.search(rf'{re.escape(prefix)}\((.*)\)', action, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    else:
        raw = action[len(prefix)+1:-1].strip()
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, str):
            return result
    except Exception:
        pass
    return strip_outer_quotes(raw)


from src.envs.bird_interact_env.test_case_utils.db_utils import execute_queries, reset_and_restore_database
from src.envs.bird_interact_env.test_case_utils.test_utils import test_case_default
from src.config.db_config import get_db_config

# Set up logger
from rich.logging import RichHandler
import logging
handler = RichHandler(show_time=False)
handler.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
logger.addHandler(handler)

KNOWLEDGE_VISIBLE_FIELDS = ["id", "knowledge", "description", "definition"]

MAX_RESULT_LENGTH = 1000

class BirdInteractSqlEnv(BaseEnv):
    """Gym environment for SQL with bird-interact specific functionality"""
    name = "bird_interact_sql"

    def __init__(self, **kwargs):
        super(BirdInteractSqlEnv, self).__init__(**kwargs)
        self.logger = logger
        self.column_meanings = {}
        self.external_knowledge = {}
        self.agent_external_knowledge = {}  # Knowledge visible to the agent (may be incomplete)
        self.schema = None
        self.phase = 1  # Start with phase 1
        # Get fresh database configuration
        self.sql_config = get_db_config()

    def _load_schema(self, schema_path: str) -> None:
        # Load schema in json format
        self.logger.info(f"Loading schema from {schema_path}")
        try:
            with open(schema_path, "r") as f:
                self.schema = f.read()
            self.logger.info(f"Loaded schema")
        except Exception as e:
            self.logger.error(f"Failed to load schema: {e}")
            self.schema = "Schema not available"

    def _load_column_meanings(self, column_meaning_path: str) -> None:
        """Load column meanings from JSON file"""
        self.logger.info(f"Loading column meanings from {column_meaning_path}")
        try:
            with open(column_meaning_path, "r") as f:
                self.column_meanings = json.load(f)
            # case-insensitive
            self.column_meanings = {k.lower(): v for k, v in self.column_meanings.items()}
            self.logger.info(f"Loaded column meanings")
        except Exception as e:
            self.logger.error(f"Failed to load column meanings: {e}")
            self.column_meanings = {}

    def _load_external_knowledge(self, kb_path: str) -> None:
        """Load external knowledge from JSON file"""
        self.logger.info(f"Loading external knowledge from {kb_path}")
        try:
            self.external_knowledge = {}
            with open(kb_path, "r") as f:
                for line in f:
                    knowledge = json.loads(line.strip())
                    self.external_knowledge[knowledge["knowledge"]] = knowledge
            self.logger.info(f"Loaded external knowledge")
        except Exception as e:
            self.logger.error(f"Failed to load external knowledge: {e}")
            self.external_knowledge = {}

    def _filter_knowledge_for_agent(self) -> None:
        """
        Create filtered knowledge dictionary for agent by removing knowledge
        marked as deleted_knowledge in the record's knowledge_ambiguity field.
        """
        self.agent_external_knowledge = self.external_knowledge.copy()
        
        if "knowledge_ambiguity" in self.record:
            deleted_knowledge_ids = []
            for ambiguity in self.record["knowledge_ambiguity"]:
                if "deleted_knowledge" in ambiguity and ambiguity["deleted_knowledge"] is not None:
                    deleted_knowledge_ids.append(ambiguity["deleted_knowledge"])
            
            self.logger.info(f"Filtering out knowledge IDs: {deleted_knowledge_ids}")
            
            # Remove knowledge with matching IDs
            if deleted_knowledge_ids:
                to_remove = []
                for k_name, k_info in self.agent_external_knowledge.items():
                    if k_info.get("id") in deleted_knowledge_ids:
                        to_remove.append(k_name)
                
                for k_name in to_remove:
                    self.logger.info(f"Removing knowledge: {k_name}")
                    del self.agent_external_knowledge[k_name]
        
        self.logger.info(f"Agent knowledge has {len(self.agent_external_knowledge)} entries vs {len(self.external_knowledge)} total entries")

    def reset_db(self):
        # Use instance sql_config instead of module-level SQL_CONFIG
        self.sql_config['dbname'] = self.record['selected_database']
        reset_and_restore_database(self.record['selected_database'], self.sql_config['password'])
        self.cnx = psycopg2.connect(
            host=self.sql_config['host'],
            port=self.sql_config['port'],
            user=self.sql_config['user'],
            password=self.sql_config['password'],
            dbname=self.sql_config['dbname']
        )
        self.cur = self.cnx.cursor()

    def reset_db_extra_info(self):
        """
        Reset extra information for the environment, e.g. the schema, column meaning, and kb for current db.
        """
        try:
            # Get database folder path
            db_path = os.path.join(os.path.dirname(self.data_loader.data_path), self.record['selected_database'])
            
            # Load schema
            schema_path = os.path.join(db_path, f"{self.record['selected_database']}_schema.txt")
            self._load_schema(schema_path)
            
            # Load column meanings
            column_meaning_path = os.path.join(db_path, f"{self.record['selected_database']}_column_meaning_base.json")
            self._load_column_meanings(column_meaning_path)
            
            # Load external knowledge
            kb_path = os.path.join(db_path, f"{self.record['selected_database']}_kb.jsonl")
            self._load_external_knowledge(kb_path)
            
            # Filter knowledge for agent based on knowledge_ambiguity
            self._filter_knowledge_for_agent()
        except Exception as e:
            self.logger.error(f"Error loading database extra info: {e}")

    def execute_preprocess_sql(self):
        """Execute preprocessing SQL commands"""
        if "preprocess_sql" in self.record and self.record["preprocess_sql"]:
            self.logger.info("Executing preprocess SQL commands")
            preprocess_cmds = self.record["preprocess_sql"]
            if isinstance(preprocess_cmds, str):
                preprocess_cmds = [preprocess_cmds]
            
            for cmd in preprocess_cmds:
                try:
                    self.cur.execute(cmd)
                    self.cnx.commit()
                    self.logger.info(f"Executed preprocess command: {cmd}")
                except Exception as e:
                    self.logger.error(f"Error executing preprocess command: {e}")
    
    def execute_cleanup_sql(self):
        """Execute cleanup SQL commands before phase 2"""
        if "clean_up_sqls" in self.record and self.record["clean_up_sqls"]:
            self.logger.info("Executing cleanup SQL commands")
            cleanup_cmds = self.record["clean_up_sqls"]
            if isinstance(cleanup_cmds, str):
                cleanup_cmds = [cleanup_cmds]
            
            for cmd in cleanup_cmds:
                try:
                    self.cur.execute(cmd)
                    self.cnx.commit()
                    self.logger.info(f"Executed cleanup command: {cmd}")
                except Exception as e:
                    self.logger.error(f"Error executing cleanup command: {e}")
    
    def start_phase_2(self):
        """Prepare environment for phase 2"""
        self.execute_cleanup_sql()
        self.phase = 2
        if "follow_up" in self.record and "sol_sql" in self.record["follow_up"]:
            self.gold = self.record["follow_up"]["sol_sql"]
            if isinstance(self.gold, list):
                self.gold = '\n'.join(self.gold)
        self.logger.info(f"Started phase 2 with gold SQL: {self.gold}")

    def reset(self, index: int = None) -> Tuple[str, Dict]:
        """
        Create new session and reset environment variables

        Args:
            index (`int`) - index of query, gold pair to use for new session. If None, random index is used.
        """
        # Reset instance variables
        self.info = {}
        self.trajectory = []
        self.observation = None
        self.phase = 1

        # Set query, gold command
        self.logger.info("-------------\nNew task episode initialized")
        self.query_idx = np.random.randint(0, len(self.data_loader)) if index is None else index
        self.record = self.data_loader.get(self.query_idx)
        self.query = self.record["amb_user_query"]
        self.gold = self.record["sol_sql"] if "sol_sql" in self.record else "N/A"
        if isinstance(self.gold, list):
            self.gold = '\n'.join(self.gold)
        self.logger.info(f"Query: {self.query}")
        self.logger.info(f"Gold PSQL: {self.gold}")
        self.observation = self.query
        self.reward = None

        # reset the database
        self.logger.info(f"Resetting the database for {self.record['selected_database']}")
        self.reset_db()
        # reset the extra info
        self.reset_db_extra_info()

        # Execute preprocessing SQL commands
        self.execute_preprocess_sql()
        
        return self.observation, self.info

    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """
        Process an action from the agent and return the observation.
        Uses utilities from test_case_utils_from_another_repo for compatibility.
        
        Args:
            action: String action to execute
            
        Returns:
            Tuple[str, float, bool, dict]: (observation, reward, done, info)
        """
        reward = 0.0
        done = False
        info = {}
        
        # if action == "skip":
        #     return "skipped", reward, True, info
            
        # if action.startswith("submit"):
        #     # Record the action
        #     self.trajectory.append((action, None))
        #     return self.observation, 1.0, True, {ACTION_EXEC: True}
        
        # Process the action based on type
        if action.startswith("execute("):
            # Extract SQL command
            sql = parse_action_arg(action, "execute")
            
            # Use execute_queries from test_case_utils_from_another_repo
            result, error, timeout = execute_queries(sql, self.record['selected_database'], self.cnx)
            
            if error:
                observation = f"SQL execution error: {error}"
                self.info[ACTION_EXEC] = False
            elif timeout:
                observation = "SQL execution timed out"
                self.info[ACTION_EXEC] = False
            else:
                # Format result for display
                if result:
                    formatted = []
                    if self.cur.description:
                        # Get column names
                        cols = [desc[0] for desc in self.cur.description]
                        formatted.append(" | ".join(cols))
                        formatted.append("-" * sum(len(c) + 3 for c in cols))
                        
                        for row in result:
                            formatted.append(" | ".join(str(cell) for cell in row))
                        
                        observation = "\n".join(formatted)
                    else:
                        observation = str(result)
                    words_count = len(observation.split())
                    if words_count > MAX_RESULT_LENGTH:
                        observation = " ".join(observation.split()[:MAX_RESULT_LENGTH]) + "..."
                else:
                    observation = "Query executed successfully. No results to display."
                self.info[ACTION_EXEC] = True
        elif action.startswith("get_schema()"):
            observation = self.get_schema()
            self.info[ACTION_EXEC] = True
        elif action.startswith("get_all_column_meanings()"):
            observation = self.get_all_column_meanings()
            self.info[ACTION_EXEC] = True
        elif action.startswith("get_column_meaning("):
            try:
                # Parse the arguments
                match = re.search(r"get_column_meaning\((.*)\)", action, re.DOTALL)
                params_str = match.group(1).strip() if match else action[19:-1].strip()
                try:
                    parsed = ast.literal_eval(params_str)
                    if isinstance(parsed, tuple):
                        parts = list(parsed)
                    else:
                        parts = [str(parsed)]
                except Exception:
                    parts = [strip_outer_quotes(p.strip()) for p in params_str.split(",")]
                if len(parts) != 2:
                    observation = "Error: get_column_meaning requires two arguments: table_name, column_name"
                    self.info[ACTION_EXEC] = False
                else:
                    table_name, column_name = parts
                    observation = self.get_column_meaning(table_name, column_name)
                    self.info[ACTION_EXEC] = True
            except Exception as e:
                observation = f"Error parsing arguments for get_column_meaning: {e}"
                self.info[ACTION_EXEC] = False
        elif action.startswith("get_all_external_knowledge_names()"):
            observation = str(self.get_all_external_knowledge_names())
            self.info[ACTION_EXEC] = True
        elif action.startswith("get_knowledge_definition("):
            try:
                # Extract knowledge name
                knowledge_name = parse_action_arg(action, "get_knowledge_definition")
                observation = self.get_knowledge_definition(knowledge_name)
                self.info[ACTION_EXEC] = True
            except Exception as e:
                observation = f"Error: {str(e)}"
                self.info[ACTION_EXEC] = False
        elif action.startswith("get_all_knowledge_definitions()"):
            observation = self.get_all_knowledge_definitions()
            self.info[ACTION_EXEC] = True
        else:
            observation = f"Unknown action: {action}"
            self.info[ACTION_EXEC] = False
        
        # Record the action and observation
        self.observation = observation
        self.trajectory.append((action, self.observation))
        self.logger.info(f"Action: {action}")
        self.logger.info(f"Observation: {self.observation}")
        
        return self.observation, reward, done, self.info

    def exec_action(self, action: str) -> str:
        """ Not used """
        pass

    def execute_sql(self, sql: str) -> str:
        """Execute SQL and return the result"""
        try:
            self.cur.execute(sql)
            # Get the results if there are any
            if self.cur.description is not None:
                # Fetch data
                rows = self.cur.fetchall()
                # Get column names
                cols = [desc[0] for desc in self.cur.description]
                
                # Format as readable text
                result = []
                result.append(" | ".join(cols))
                result.append("-" * sum(len(c) + 3 for c in cols))
                
                for row in rows:
                    result.append(" | ".join(str(cell) for cell in row))
                
                return "\n".join(result)
            else:
                # No results, e.g. for INSERT, UPDATE, etc.
                row_count = self.cur.rowcount
                return f"Query executed successfully. {row_count} rows affected."
        except Exception as e:
            return f"SQL execution error: {e}"
    
    # Test case evaluation methods
    def preprocess_results(self, results):
        """
        Convert any date/datetime in results into 'YYYY-MM-DD' strings.
        Leave other types untouched.
        """
        processed = []
        for row in results:
            new_row = []
            for item in row:
                if isinstance(item, (date, datetime)):
                    new_row.append(item.strftime('%Y-%m-%d'))
                else:
                    new_row.append(item)
            processed.append(tuple(new_row))
        return processed
    
    def remove_comments(self, sql_list):
        """
        Remove all SQL comments from each query string in the list.
        """
        if isinstance(sql_list, str):
            sql_list = [sql_list]
            
        cleaned = []
        for sql in sql_list:
            # remove block comments
            no_block = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
            # remove line comments, keep newline
            no_line  = re.sub(r'--.*?(\r\n|\r|\n)', r'\1', no_block)
            # collapse extra blank lines
            no_blank = re.sub(r'\n\s*\n+', '\n', no_line)
            cleaned.append(no_blank.strip())
        return cleaned
    
    def remove_distinct(self, sql_list):
        """
        Strip out all DISTINCT tokens (case-insensitive).
        """
        if isinstance(sql_list, str):
            sql_list = [sql_list]
            
        cleaned = []
        for q in sql_list:
            tokens = q.split()
            tokens = [t for t in tokens if t.lower() != 'distinct']
            cleaned.append(' '.join(tokens))
        return cleaned
    
    def remove_order_by(self, sql_list):
        """
        Remove all ORDER BY ... clauses
        """
        if isinstance(sql_list, str):
            sql_list = [sql_list]
            
        pattern = re.compile(
            r'\bORDER\s+BY\b'       # ORDER BY
            r'.*?'                     # non-greedy
            r'(?=(?:\bLIMIT\b|\bFETCH\b|\bOFFSET\b|\)|;|$))',
            flags=re.IGNORECASE | re.DOTALL
        )
        cleaned = []
        for sql in sql_list:
            no_ob = pattern.sub('', sql)
            # normalize whitespace
            no_ob = re.sub(r'\s+', ' ', no_ob).strip()
            cleaned.append(no_ob)
        return cleaned
    
    def process_decimals(self, results, decimal_places):
        """
        Round any Decimal or float in the result set to `decimal_places`.
        """
        quantizer = Decimal(1).scaleb(-decimal_places)
        rounded = []
        for row in results:
            new_row = []
            for item in row:
                if isinstance(item, Decimal):
                    new_row.append(item.quantize(quantizer, rounding=ROUND_HALF_UP))
                elif isinstance(item, float):
                    new_row.append(round(item, decimal_places))
                else:
                    new_row.append(item)
            rounded.append(tuple(new_row))
        return rounded
    
    def ex_base(self, pred_sqls, sol_sqls, decimal_places=2):
        """
        Compare result-sets of two lists of SQL queries
        """
        if not pred_sqls or not sol_sqls:
            return False, "Empty SQL query"

        # Execute predicted SQL
        try:
            self.cur.execute(pred_sqls[0])
            predicted_res = self.cur.fetchall() if self.cur.description is not None else []
        except Exception as err:
            return False, f"Error executing predicted SQL: {err}"
        
        # Execute solution SQL
        try:
            self.cur.execute(sol_sqls[0])
            ground_res = self.cur.fetchall() if self.cur.description is not None else []
        except Exception as err:
            return False, f"Error executing solution SQL: {err}"

        # Process results
        predicted_res = self.preprocess_results(predicted_res)
        ground_res = self.preprocess_results(ground_res)
        
        if not predicted_res and not ground_res:
            return True, "Both queries return empty results"
        
        if not predicted_res:
            return False, "Predicted SQL returns empty results but solution SQL doesn't"
        
        if not ground_res:
            return False, "Solution SQL returns empty results but predicted SQL doesn't"

        # Round decimals
        if decimal_places is not None:
            predicted_res = self.process_decimals(predicted_res, decimal_places)
            ground_res = self.process_decimals(ground_res, decimal_places)

        # Compare results
        if set(predicted_res) == set(ground_res):
            return True, "SQL passed the test case"
        else:
            return False, "Results don't match"
    
    def run_test_case(self, sql: str) -> Tuple[bool, str]:
        """
        Run the test case for the given SQL.
        Uses approach from eval_bird_interact_batch.py:
        1. First check if the SQL is executable
        2. Check for custom test_cases in the data record
        3. If no custom test cases, use default test_case function

        Returns:
            Tuple[bool, str]: (passed, message)
        """
        # Check for test cases based on the current phase
        test_cases = []
        if self.phase == 2 and "follow_up" in self.record and "test_cases" in self.record["follow_up"]:
            test_cases = self.record["follow_up"]["test_cases"]
        elif "test_cases" in self.record:
            test_cases = self.record["test_cases"]
        
        # First check if the SQL is executable and get pred_result
        try:
            # Use execute_queries to get the result (handles errors better)
            pred_query_result, pred_err, pred_to = execute_queries(sql, self.record['selected_database'], self.cnx)
            if pred_err:
                return False, f"Error executing SQL: {pred_err}"
            if pred_to:
                return False, "SQL execution timed out"
        except Exception as err:
            return False, f"Error executing SQL: {err}"
        
        # Get sol_sql based on the current phase
        if self.phase == 2 and "follow_up" in self.record:
            sol_sqls = self.record["follow_up"]["sol_sql"]
            conditions = self.record["follow_up"].get("conditions", {})
        else:
            sol_sqls = self.record.get("sol_sql", "")
            conditions = self.record.get("conditions", {})
        
        if not isinstance(sol_sqls, list):
            sol_sqls = [sol_sqls]
        
        # Store the pred_query_result in globals for test case functions
        globals()['pred_query_result'] = pred_query_result
        
        # Use test_case_default if no custom test cases
        if not test_cases:
            try:
                # Run the default test case function
                test_case_default([sql], sol_sqls, self.record['selected_database'], self.cnx, conditions)
                return True, "SQL passed default test case"
            except AssertionError as e:
                return False, f"Default test case failed: {str(e)}"
            except Exception as e:
                self.logger.error(f"Error running default test case: {e}")
                return False, f"Error in default test case: {str(e)}"
        
        # Run custom test cases if present
        try:
            # For each test case, run it individually
            for test_case_code in test_cases:
                if not isinstance(test_case_code, str):
                    # Skip non-string test cases (shouldn't happen in normal data)
                    continue
                    
                # Set up the test case environment
                local_ns = {}
                
                # Execute the test case code
                try:
                    exec(test_case_code, globals(), local_ns)
                    test_case_func = local_ns.get('test_case')
                    
                    if test_case_func and callable(test_case_func):
                        # Run the custom test case function
                        test_case_func([sql], sol_sqls, self.record['selected_database'], self.cnx)
                except AssertionError as e:
                    # The test cases use assertions to signal failure
                    return False, f"Test case assertion failed: {str(e)}"
                except Exception as e:
                    self.logger.error(f"Error running custom test case: {e}")
                    return False, f"Error in test case execution: {str(e)}"
            
            # If we get here, all test cases passed (no assertions were triggered)
            return True, "SQL passed all custom test cases"
        except Exception as e:
            self.logger.error(f"Error running test cases: {e}")
            return False, f"Error in test case execution: {str(e)}"
    
    def get_schema(self) -> str:
        """Get the schema of the database"""
        return self.schema
    
    def get_all_column_meanings(self) -> str:
        """Get the meaning of all columns in the database"""
        return json.dumps(self.column_meanings, indent=2)
    
    def get_column_meaning(self, table_name: str, column_name: str) -> str:
        """Get the meaning of a specific column"""
        # case-insensitive
        key = f"{self.record['selected_database']}|{table_name.lower()}|{column_name.lower()}"
        return self.column_meanings.get(key, "Column meaning not found")
    
    def get_all_external_knowledge_names(self) -> List[str]:
        """Get all external knowledge names"""
        return list(self.agent_external_knowledge.keys())
    
    def get_knowledge_definition(self, knowledge_name: str) -> str:
        """Get external knowledge by name"""
        if knowledge_name in self.agent_external_knowledge:
            knowledge = self.agent_external_knowledge[knowledge_name]
            # only return the visible fields
            knowledge = {k: knowledge[k] for k in KNOWLEDGE_VISIBLE_FIELDS}
            return json.dumps(knowledge, indent=2)
        return "Knowledge not found"
    
    def get_all_knowledge_definitions(self) -> str:
        """Get all external knowledge definitions"""
        return json.dumps([{k: v[k] for k in KNOWLEDGE_VISIBLE_FIELDS} for v in self.agent_external_knowledge.values()], indent=2)

    def get_reward(self) -> Tuple[float, Dict]:
        # Not currently used but kept for compatibility
        self.info = {}
        self.info[AGENT_OBS] = self.observation
        self.info[REWARD] = 0.0
        self.reward = self.info[REWARD]
        return self.reward, self.info

    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self.cur.close()
        self.cnx.close()
        # Use instance sql_config instead of module-level SQL_CONFIG
        reset_and_restore_database(self.record['selected_database'], self.sql_config['password'])
        self.logger.info("Agent, evaluation env stopped")
    
    ############################
    ### MARK: Helper methods ###
    ############################
    def get_intersect_items(self, my_list: List, my_dict: Dict) -> List:
        """
        Returns the intersection of a list and a dictionary.
        """
        result = []
        for item in my_list:
            if item in my_dict:
                my_dict[item] -= 1
                if my_dict[item] == 0:
                    del my_dict[item]
                result.append(item)
        return result