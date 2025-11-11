import json
import os
import re
import sqlite3
from typing import Dict, Any, Tuple, Optional

from src.envs.bird_interact_env.test_case_utils_sqlite.db_utils import execute_queries, reset_and_restore_database
from src.envs.bird_interact_env.test_case_utils_sqlite.test_utils import test_case_default

# Set up logger
import logging
logger = logging.getLogger(__name__)

MAX_RESULT_LENGTH = 500
KNOWLEDGE_VISIBLE_FIELDS = ["id", "knowledge", "description", "definition"]

# --- Database Connection Cache --- (Simple example, might need more robust handling)
_db_connections = {}
_db_cursors = {}
_db_configs = {}

# --- Data Cache (Schema, Knowledge, Column Meanings) ---
_schema_cache = {}
_column_meanings_cache = {}
_external_knowledge_cache = {}
_agent_external_knowledge_cache = {}

def _resolve_sqlite_db_path(db_name: str, data_path_base: str, record: Dict[str, Any]) -> str:
    """Resolve the SQLite database file path from inputs.

    Priority:
      1) record['db_file_path'] if present
      2) data_path_base/<db_name>/<db_name>.sqlite
    """
    explicit = record.get('db_file_path') if isinstance(record, dict) else None
    if explicit:
        return explicit
    return os.path.join(data_path_base, db_name, f"{db_name}.sqlite")


def get_db_connection(db_path: str):
    """Gets or creates a SQLite connection for the given db_path."""
    conn = _db_connections.get(db_path)
    if conn is None:
        logger.info(f"Creating new SQLite connection for: {db_path}")
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = OFF")
            _db_connections[db_path] = conn
            _db_cursors[db_path] = conn.cursor()
        except Exception as e:
            logger.error(f"Failed to connect to SQLite database {db_path}: {e}")
            raise
    return _db_connections[db_path], _db_cursors[db_path]

def close_db_connection(db_path: str):
    """Closes the database connection if it exists."""
    if db_path in _db_cursors:
        _db_cursors[db_path].close()
        del _db_cursors[db_path]
    if db_path in _db_connections:
        _db_connections[db_path].close()
        del _db_connections[db_path]
    if db_path in _db_configs:
        del _db_configs[db_path]
    logger.info(f"Closed connection for database: {db_path}")

def reset_and_reconnect_db(db_path: str):
    """Resets the SQLite database file from its template and reconnects."""
    logger.info(f"Resetting SQLite database: {db_path}")
    close_db_connection(db_path)
    # Password is unused for SQLite reset; pass empty string and module logger
    reset_and_restore_database(db_path, "", logger)
    # Reconnect
    conn, cur = get_db_connection(db_path)
    logger.info(f"Reconnected to SQLite database: {db_path}")
    return conn, cur

def load_db_data_if_needed(db_name: str, data_path_base: str):
    """Loads schema, column meanings, and knowledge for a db if not already cached."""
    if db_name not in _schema_cache:
        db_folder_path = os.path.join(data_path_base, db_name)
        # Load Schema
        schema_path = os.path.join(db_folder_path, f"{db_name}_schema.txt")
        try:
            with open(schema_path, "r") as f:
                _schema_cache[db_name] = f.read()
            logger.debug(f"Loaded schema for {db_name}")
        except Exception as e:
            logger.error(f"Failed to load schema for {db_name} from {schema_path}: {e}")
            _schema_cache[db_name] = "Schema not available"

        # Load Column Meanings
        col_mean_path = os.path.join(db_folder_path, f"{db_name}_column_meaning_base.json")
        try:
            with open(col_mean_path, "r") as f:
                meanings = json.load(f)
            # Case-insensitive keys
            _column_meanings_cache[db_name] = {k.lower(): v for k, v in meanings.items()}
            logger.debug(f"Loaded column meanings for {db_name}")
        except Exception as e:
            logger.error(f"Failed to load column meanings for {db_name} from {col_mean_path}: {e}")
            _column_meanings_cache[db_name] = {}

        # Load External Knowledge
        kb_path = os.path.join(db_folder_path, f"{db_name}_kb.jsonl")
        try:
            kb = {}
            with open(kb_path, "r") as f:
                for line in f:
                    knowledge = json.loads(line.strip())
                    kb[knowledge["knowledge"]] = knowledge
            _external_knowledge_cache[db_name] = kb
            logger.debug(f"Loaded external knowledge for {db_name}")
        except Exception as e:
            logger.error(f"Failed to load external knowledge for {db_name} from {kb_path}: {e}")
            _external_knowledge_cache[db_name] = {}

# --- Action Execution Functions ---

def execute_env_action(action: str, sample_status: 'SampleStatus', data_path_base: str) -> Tuple[str, bool]:
    """
    Handles actions directed towards the Environment.
    Returns (observation, success_flag)
    """
    db_name = sample_status.original_data['selected_database']
    db_path = _resolve_sqlite_db_path(db_name, data_path_base, sample_status.original_data)
    load_db_data_if_needed(db_name, data_path_base)
    observation = ""
    success = False

    try:
        conn, cur = get_db_connection(db_path)

        if action.startswith("execute("):
            # IMPORTANT: Reset DB before execution as per requirements
            conn, cur = reset_and_reconnect_db(db_path)
            # Execute any preprocess_sql for the *current* phase before the action
            # (Requires knowing the current phase and preprocess SQL for that phase)
            # Simplified: Assuming preprocess is done once at the start

            sql = action[8:-1].strip().strip("'\"")
            result, error, timeout = execute_queries(sql, db_path, conn)
            if error:
                observation = f"SQL execution error: {error}"
                success = False
            elif timeout:
                observation = "SQL execution timed out"
                success = False
            else:
                # Format result for display (incorporating logic similar to BirdInteractSqlEnv)
                if result is not None:
                    # Check if the result is likely from a SELECT (list of tuples)
                    is_select_result = isinstance(result, list) and all(isinstance(row, tuple) for row in result)

                    if is_select_result:
                        try:
                            # Attempt to format as table
                            formatted = []
                            # We cannot rely on cur.description here (query executed via helper)
                            for row in result:
                                truncated_row = []
                                for cell in row:
                                    cell_str = str(cell)
                                    if len(cell_str) > 100:
                                        cell_str = cell_str[:97] + "..."
                                    truncated_row.append(cell_str)
                                formatted.append(" | ".join(truncated_row))
                            observation = "\n".join(formatted) if formatted else "Query executed, empty result set."
                            if not formatted and result:
                                observation = str(result)
                                if len(result) == 1 and result[0] is not None and len(result[0]) == 1:
                                    observation = f"Result: {result[0][0]}"

                            # Truncate overall observation if too long
                            words_count = len(observation.split())
                            if words_count > MAX_RESULT_LENGTH:
                                observation = " ".join(observation.split()[:MAX_RESULT_LENGTH]) + "..."
                            success = True
                        except Exception as format_e:
                            logger.error(f"Error formatting SQL result: {format_e}", exc_info=True)
                            observation = f"Query executed, but error formatting results: {format_e}"
                            success = True # Query ran, formatting failed
                    else:
                        # Handle non-SELECT results or empty SELECT results
                        # sqlite3's rowcount is not reliable for SELECT; provide generic message or result
                        if result is not None:
                            observation = f"Query executed successfully. Result: {str(result)}"
                        else:
                            observation = "Query executed successfully."
                        success = True
                else:
                    # If execute_queries returned None and no error/timeout (should indicate success unless specific commands failed silently)
                    observation = "Query executed successfully."
                    success = True

        elif action == "get_schema()":
            observation = _schema_cache.get(db_name, "Schema not available")
            success = True

        elif action == "get_all_column_meanings()":
            observation = json.dumps(_column_meanings_cache.get(db_name, {}), indent=2)
            success = True

        elif action.startswith("get_column_meaning("):
            match = re.search(r"get_column_meaning\((.*)\)", action)
            if match:
                params_str = match.group(1).strip()
                try:
                    # Attempt to parse assuming comma-separated strings
                    parts = [p.strip().strip("'\"") for p in params_str.split(",")]
                    if len(parts) == 2:
                        table_name, column_name = parts
                        key = f"{db_name}|{table_name.lower()}|{column_name.lower()}"
                        observation = _column_meanings_cache.get(db_name, {}).get(key, "Column meaning not found")
                        success = True
                    else:
                        observation = "Error: Invalid arguments for get_column_meaning. Expected table_name, column_name."
                except Exception as parse_e:
                    observation = f"Error parsing arguments for get_column_meaning: {parse_e}"
            else:
                observation = "Error: Could not parse arguments for get_column_meaning."

        elif action == "get_all_external_knowledge_names()":
            # Filter knowledge based on ambiguity settings for the agent
            agent_kb = _filter_knowledge_for_agent(db_name, sample_status.original_data)
            observation = str(list(agent_kb.keys()))
            success = True

        elif action.startswith("get_knowledge_definition("):
             match = re.search(r"get_knowledge_definition\((.*)\)", action)
             if match:
                knowledge_name = match.group(1).strip().strip("'\"")
                agent_kb = _filter_knowledge_for_agent(db_name, sample_status.original_data)
                if knowledge_name in agent_kb:
                    knowledge = agent_kb[knowledge_name]
                    visible_knowledge = {k: knowledge[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in knowledge}
                    observation = json.dumps(visible_knowledge, indent=2)
                else:
                    observation = "Knowledge not found or not accessible."
                success = True
             else:
                observation = "Error: Could not parse arguments for get_knowledge_definition."

        elif action == "get_all_knowledge_definitions()":
            agent_kb = _filter_knowledge_for_agent(db_name, sample_status.original_data)
            visible_kbs = []
            for k_info in agent_kb.values():
                visible_kbs.append({k: k_info[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in k_info})
            observation = json.dumps(visible_kbs, indent=2)
            success = True

        else:
            observation = f"""Unknown Environment action: {action} Your availabel actions to Database are 
            execute(sql): Execute a SQL query
            get_schema(): Get the schema of the database
            get_all_column_meanings(): Get all column meanings
            get_all_external_knowledge_names(): Get all external knowledge names
            get_knowledge_definition(knowledge_name): Get the definition of a specific knowledge
            get_all_knowledge_definitions(): Get all knowledge definitions
            """
            success = False

    except sqlite3.Error as db_err:
        observation = f"Error (SQLite DB): {db_err}"
        success = False
        try:
            if db_path in _db_connections:
                _db_connections[db_path].rollback()
        except Exception:
            pass
    except BaseException as e: # Temporarily catch BaseException for debugging
        # logger.error(f"Error executing environment action '{action}' for db {db_name}: {e}", exc_info=True)
        observation = f"Error executing action: {e}"
        success = False

    return observation, success

def execute_submit_action(sql: str, sample_status: 'SampleStatus', data_path_base: str) -> Tuple[str, float, bool, bool, bool]:
    """
    Handles the submit(sql) action.
    Resets the DB, runs the test case, determines reward and phase completion.
    Returns (observation, reward, phase1_completed, phase2_completed, task_finished)
    """
    db_name = sample_status.original_data['selected_database']
    db_path = _resolve_sqlite_db_path(db_name, data_path_base, sample_status.original_data)
    current_phase = sample_status.current_phase
    phase_rewards = {1: 1} # Define rewards per phase
    observation = ""
    reward = 0.0
    task_finished = False
    phase1_completed = sample_status.phase1_completed
    phase2_completed = sample_status.phase2_completed

    try:
        # --- Database Preparation ---
        # Always reset DB before any submission attempt
        conn, cur = reset_and_reconnect_db(db_path)
        logger.debug(f"Sample {sample_status.idx}, Phase {current_phase}: Database reset for submission.")

        # Load necessary data (schema, etc.) if not already loaded (though reset might clear caches? Check load logic)
        load_db_data_if_needed(db_name, data_path_base)

        record = sample_status.original_data

        # Deprecated: Follow-up phase is not used in Mini-Interact.
        # For Phase 2, conditionally execute preparatory SQL based on category
        if current_phase == 2 and record.get('category') == 'Management':
            logger.debug(f"Sample {sample_status.idx}, Phase 2: Running preparatory SQL (Category: Management).")

            # 1. Get and Run Preprocess SQL
            preprocess_sql_list = record.get('preprocess_sql', [])
            if isinstance(preprocess_sql_list, str):
                preprocess_sql_list = [preprocess_sql_list] if preprocess_sql_list else []

            if preprocess_sql_list:
                logger.debug(f"Sample {sample_status.idx}: Executing preprocess SQL: {preprocess_sql_list}")
                for pp_sql in preprocess_sql_list:
                    if pp_sql:
                        try:
                            cur.execute(pp_sql)
                        except Exception as pp_err:
                            raise RuntimeError(f"Error executing preprocess SQL '{pp_sql}': {pp_err}") from pp_err

            # 2. Get and Run Successful Phase 1 SQL (if it exists)
            if sample_status.successful_phase1_sql:
                logger.debug(f"Sample {sample_status.idx}: Executing successful Phase 1 SQL: {sample_status.successful_phase1_sql[:100]}...")
                try:
                    cur.execute(sample_status.successful_phase1_sql)
                except Exception as p1_err:
                    logger.error(f"Error executing stored Phase 1 SQL: {p1_err}")
                    raise RuntimeError(f"Error executing stored Phase 1 SQL: {p1_err}") from p1_err

                # 3. Get and Run Cleanup SQL (after successful Phase 1 SQL)
                cleanup_sql_list = record.get('clean_up_sqls', [])
                if isinstance(cleanup_sql_list, str):
                    cleanup_sql_list = [cleanup_sql_list] if cleanup_sql_list else []

                if cleanup_sql_list:
                    logger.debug(f"Sample {sample_status.idx}: Executing cleanup SQL: {cleanup_sql_list}")
                    for cu_sql in cleanup_sql_list:
                        if cu_sql:
                            try:
                                cur.execute(cu_sql)
                            except Exception as cu_err:
                                raise RuntimeError(f"Error executing cleanup SQL '{cu_sql}': {cu_err}") from cu_err
            else:
                logger.warning(f"Sample {sample_status.idx}, Phase 2: No successful Phase 1 SQL found in status to execute.")

            # Commit preparatory changes (preprocess, phase1, cleanup)
            conn.commit()
            logger.debug(f"Sample {sample_status.idx}: Preparatory SQL committed.")
        elif current_phase == 2:
            logger.debug(f"Sample {sample_status.idx}, Phase 2: Skipping preparatory SQL execution (Category: {record.get('category')}).")

        # --- Run Test Case --- (Adapted from BirdInteractSqlEnv.run_test_case)
        passed = False
        message = "Test case execution failed."

        # Determine correct test cases and solution SQL based on phase
        test_cases = []
        sol_sqls = []

        if current_phase == 2 and "follow_up" in record and record["follow_up"]:
            test_cases = record["follow_up"].get("test_cases", [])
            sol_sqls = record["follow_up"].get("sol_sql", [])
            conditions = record["follow_up"].get("conditions", {})
            category = record["follow_up"].get("category", "Query")
        elif current_phase == 1:
            test_cases = record.get("test_cases", [])
            sol_sqls = record.get("sol_sql", [])
            conditions = record.get("conditions", {})
            category = record.get("category", "Query")
        else: # Phase 1 but no test cases defined? Or invalid phase?
            message = f"Cannot run test cases for Phase {current_phase}. No relevant data found."
            passed = False
            # Skip the rest of the test case logic if no test cases/sol SQL for the phase
            logger.warning(f"Sample {sample_status.idx}: {message}")
            # Need to decide how to proceed here. Let's assume failure.

        if not isinstance(sol_sqls, list):
            sol_sqls = [sol_sqls] if sol_sqls else []

        # Only proceed with test case execution if we have solution SQLs to compare against
        if sol_sqls:
            try:
                # Check executability of the *submitted* SQL (sql parameter)
                pred_query_result, pred_err, pred_to = execute_queries(sql, db_path, conn)

                if pred_err:
                    message = f"Error executing submitted SQL: {pred_err}"
                elif pred_to:
                    message = "Submitted SQL execution timed out"
                else:
                    # Store result for test cases if needed
                    exec_globals = {'pred_query_result': pred_query_result, 'execute_queries': execute_queries}
                    exec_locals = {}

                    if category == "Query": # Use default test case
                        logger.debug(f"Sample {sample_status.idx}, Phase {current_phase}: Using default test case.")
                        try:
                            if isinstance(sql, str):
                                sql = [sql]
                            test_case_default(sql, sol_sqls, db_path, conn, conditions=conditions)
                            passed = True
                            message = "SQL passed default test case."
                        except AssertionError as e:
                            message = f"Default test case failed: {str(e)}"
                        except Exception as e:
                            # logger.error(f"Error running default test case for {db_name}, Phase {current_phase}: {e}")
                            message = f"Error in default test case execution: {str(e)}"
                    else: # Use custom test cases
                        logger.debug(f"Sample {sample_status.idx}, Phase {current_phase}: Using {len(test_cases)} custom test cases.")
                        all_custom_passed = True
                        failure_messages = []
                        for i, test_case_code in enumerate(test_cases):
                            if not isinstance(test_case_code, str):
                                logger.warning(f"Sample {sample_status.idx}: Custom test case {i+1} is not a string, skipping.")
                                continue
                            try:
                                exec(test_case_code, exec_globals, exec_locals)
                                test_case_func = exec_locals.get('test_case')
                                if test_case_func and callable(test_case_func):
                                    logger.debug(f"Sample {sample_status.idx}: Running custom test case {i+1}")
                                    if isinstance(sql, str):
                                        sql = [sql]
                                    test_case_func(sql, sol_sqls, db_path, conn)
                                    logger.debug(f"Sample {sample_status.idx}: Custom test case {i+1} passed.")
                                else:
                                    raise RuntimeError(f"Could not find callable 'test_case' function in custom test case {i+1}")
                            except AssertionError as e:
                                logger.info(f"Sample {sample_status.idx}: Test case {i+1} assertion failed: {str(e)}")
                                all_custom_passed = False
                                failure_messages.append(f"Test case {i+1} assertion failed: {str(e)}")
                                break # Stop on first failure
                            except Exception as e:
                                # logger.error(f"Error running custom test case {i+1} for {db_name}, Phase {current_phase}: {e}", exc_info=True)
                                all_custom_passed = False
                                failure_messages.append(f"Error in custom test case {i+1} execution: {str(e)}")
                                break # Stop on first failure

                        if all_custom_passed:
                            passed = True
                            message = "SQL passed all custom test cases."
                        else:
                            message = "SQL failed custom test cases: " + "; ".join(failure_messages)

            except Exception as exec_err:
                # logger.error(f"General error during test case execution for {db_name}, Phase {current_phase}: {exec_err}", exc_info=True)
                message = f"Error during test case execution: {exec_err}"
        # --- End Test Case --- #

        if passed:
            logger.info(f"Sample {sample_status.idx}, Phase {current_phase}: Submission PASSED. Message: {message}")
            phase_reward = phase_rewards.get(current_phase, 0)
            reward = (sample_status.last_reward or 0) + phase_reward
            if current_phase == 1:
                phase1_completed = True
                # ensure saved sql is stored as a stsring
                if isinstance(sql, list):
                    sample_status.successful_phase1_sql = "\n".join(sql)
                else:
                    sample_status.successful_phase1_sql = sql # Store successful phase 1 SQL
                # Check if there is a phase 2
                if "follow_up" in record and record["follow_up"] and record["follow_up"].get("query"):
                    observation = f"Phase 1 SQL Correct! (Reward: {phase_reward} points). Moving to Phase 2."
                    task_finished = False # Continue to phase 2
                else:
                    observation = f"Phase 1 SQL Correct! (Reward: {phase_reward} points). No Phase 2. Task finished."
                    task_finished = True
            elif current_phase == 2:
                phase2_completed = True
                observation = f"Phase 2 SQL Correct! (Reward: {phase_reward} points). Task finished."
                task_finished = True
        else:
            logger.info(f"Sample {sample_status.idx}, Phase {current_phase}: Submission FAILED. Reason: {message}")
            observation = f"Submitted SQL failed test case in Phase {current_phase}. Reason: {message} Please try again."
            reward = 0
            task_finished = False # Continue in the current phase

    except Exception as e:
        logger.error(f"Error executing submit action for db {db_path}, Phase {current_phase}: {e}", exc_info=True)
        observation = f"Error processing submission: {e}"
        reward = 0
        task_finished = False # Assume task cannot finish due to error

    return observation, reward, phase1_completed, phase2_completed, task_finished

def _filter_knowledge_for_agent(db_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters the full knowledge base based on the record's knowledge_ambiguity.
    Returns the knowledge dictionary visible to the agent.
    Uses cached full knowledge.
    """
    # Check cache for already filtered knowledge
    cache_key = (db_name, record.get('instance_id', db_name)) # Use instance_id if available
    if cache_key in _agent_external_knowledge_cache:
        return _agent_external_knowledge_cache[cache_key]

    full_kb = _external_knowledge_cache.get(db_name, {})
    if not full_kb:
        return {}

    agent_kb = full_kb.copy()
    deleted_knowledge_ids = set()
    if "knowledge_ambiguity" in record and record["knowledge_ambiguity"]:
        for ambiguity in record["knowledge_ambiguity"]:
            if "deleted_knowledge" in ambiguity and ambiguity["deleted_knowledge"] is not None:
                deleted_knowledge_ids.add(ambiguity["deleted_knowledge"])

    if deleted_knowledge_ids:
        to_remove_keys = []
        for k_name, k_info in agent_kb.items():
            if k_info.get("id") in deleted_knowledge_ids:
                to_remove_keys.append(k_name)
        for k_name in to_remove_keys:
            del agent_kb[k_name]
        logger.debug(f"Filtered knowledge for {db_name}: removed IDs {deleted_knowledge_ids}")

    # Cache the filtered result
    _agent_external_knowledge_cache[cache_key] = agent_kb
    return agent_kb

