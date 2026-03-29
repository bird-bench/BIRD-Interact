"""ADK tools for a-interact mode.

These tools allow the system agent to interact with:
1. DB Environment (port 6002): execute SQL, get schema, get column meanings, get knowledge
2. User Simulator (port 6001): ask clarification questions
3. Submission (port 6002): submit final SQL for evaluation

All tools route through the FastAPI services via HTTP.
Budget deduction and trajectory logging are handled by callbacks (callbacks.py).
"""

import json
import logging
import httpx
from typing import Optional

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from shared.config import settings

logger = logging.getLogger(__name__)

# ── Global task context (fallback, set by orchestrator before each task) ──
_current_task_id: str = ""


def set_current_task_id(task_id: str):
    global _current_task_id
    _current_task_id = task_id


def _get_task_id(tool_context: Optional[ToolContext] = None) -> str:
    if tool_context:
        tid = tool_context.state.get("task_id", "")
        if tid:
            return tid
    return _current_task_id


def _db_url(path: str) -> str:
    return f"http://localhost:{settings.db_env_port}{path}"


def _user_url(path: str) -> str:
    return f"http://localhost:{settings.user_sim_port}{path}"


# ── DB Environment Tools ──

def execute_sql(sql: str, tool_context: ToolContext) -> str:
    """Execute a SQL query against the PostgreSQL database and return the results.
    Use this to explore the database, test queries, or verify your SQL before submitting.
    Cost: 1 bird-coin.

    Args:
        sql: The PostgreSQL SQL query to execute.

    Returns:
        The query results formatted as a table, or an error message.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            resp = client.post(_db_url("/execute"),
                               json={"task_id": task_id, "sql": sql})
            if resp.status_code != 200:
                return f"SQL Error: Server returned status {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            if data.get("success"):
                return data.get("result", "Query executed successfully.")
            else:
                return f"SQL Error: {data.get('error') or 'Execution failed (no details)'}"
    except Exception as e:
        return f"Error calling DB environment: {type(e).__name__}: {e}"


def get_schema(tool_context: ToolContext) -> str:
    """Get the full database schema (CREATE TABLE statements) for the current task's database.
    Cost: 1 bird-coin.

    Returns:
        The database schema as text.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/schema"),
                               json={"task_id": task_id})
            return resp.json().get("schema", "Schema not available")
    except Exception as e:
        return f"Error: {e}"


def get_all_column_meanings(tool_context: ToolContext) -> str:
    """Get the meanings/descriptions of all columns in the database.
    Cost: 1 bird-coin.

    Returns:
        JSON string with column meanings for all tables.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/all_column_meanings"),
                               json={"task_id": task_id})
            return resp.json().get("column_meanings", "{}")
    except Exception as e:
        return f"Error: {e}"


def get_column_meaning(table_name: str, column_name: str, tool_context: ToolContext) -> str:
    """Get the meaning/description of a specific column in a table.
    Cost: 0.5 bird-coins.

    Args:
        table_name: Name of the table.
        column_name: Name of the column.

    Returns:
        The column meaning/description.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/column_meaning"),
                               json={"task_id": task_id,
                                     "table_name": table_name,
                                     "column_name": column_name})
            return resp.json().get("meaning", "Column meaning not found")
    except Exception as e:
        return f"Error: {e}"


def get_all_external_knowledge_names(tool_context: ToolContext) -> str:
    """Get the names of all available external knowledge entries for this database.
    Use this to discover what domain knowledge is available.
    Cost: 0.5 bird-coins.

    Returns:
        JSON list of knowledge entry names.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/knowledge_names"),
                               json={"task_id": task_id})
            return json.dumps(resp.json().get("names", []))
    except Exception as e:
        return f"Error: {e}"


def get_knowledge_definition(knowledge_name: str, tool_context: ToolContext) -> str:
    """Get the definition/details of a specific external knowledge entry.
    Cost: 0.5 bird-coins.

    Args:
        knowledge_name: The name of the knowledge entry to look up.

    Returns:
        JSON string with the knowledge definition.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/knowledge"),
                               json={"task_id": task_id,
                                     "knowledge_name": knowledge_name})
            return resp.json().get("knowledge", "Knowledge not found")
    except Exception as e:
        return f"Error: {e}"


def get_all_knowledge_definitions(tool_context: ToolContext) -> str:
    """Get all external knowledge definitions for this database.
    Cost: 1 bird-coin.

    Returns:
        JSON string with all knowledge definitions.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            resp = client.post(_db_url("/knowledge"),
                               json={"task_id": task_id})
            return resp.json().get("knowledge", "[]")
    except Exception as e:
        return f"Error: {e}"


# ── User Simulator Tool ──

def ask_user(question: str, tool_context: ToolContext) -> str:
    """Ask the user a clarification question about their query.
    Use this when the user's request is ambiguous and you need more information.
    Cost: 2 bird-coins.

    Args:
        question: The clarification question to ask the user.

    Returns:
        The user's response to your question.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            resp = client.post(_user_url("/ask"),
                               json={"task_id": task_id,
                                     "question": question})
            answer = resp.json().get("answer", "No response from user.")
            history = tool_context.state.get("dialogue_history", [])
            history.append({"role": "agent", "content": question})
            history.append({"role": "user", "content": answer})
            tool_context.state["dialogue_history"] = history
            return answer
    except Exception as e:
        return f"Error: {e}"


# ── Submit Tool ──

def submit_sql(sql: str, tool_context: ToolContext) -> str:
    """Submit your final SQL query for evaluation.
    This tests your SQL against the ground truth. Only submit when confident.
    Cost: 3 bird-coins.

    Args:
        sql: The final PostgreSQL SQL query to submit.

    Returns:
        Evaluation result including pass/fail, reward, and any follow-up instructions.
    """
    task_id = _get_task_id(tool_context)
    try:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            resp = client.post(_db_url("/submit"),
                               json={"task_id": task_id, "sql": sql})
            data = resp.json()

            # Update session state based on result
            if data.get("passed"):
                reward = data.get("reward", 0.0)
                tool_context.state["total_reward"] = tool_context.state.get("total_reward", 0.0) + reward
                phase = data.get("phase_completed")
                if phase == 1:
                    tool_context.state["phase1_completed"] = True
                    tool_context.state["current_phase"] = 2
                    if data.get("has_follow_up"):
                        try:
                            client.post(_user_url("/phase_transition"), json={"task_id": task_id})
                        except Exception as exc:
                            logger.warning("Phase transition failed for %s: %s", task_id, exc)
                    else:
                        tool_context.state["task_done"] = True
                elif phase == 2:
                    tool_context.state["phase2_completed"] = True
                    tool_context.state["task_done"] = True

            # Build response message
            raw_msg = data.get("message", "")
            # Store raw message for orchestrator (has [exec_err_flg] for debug routing)
            tool_context.state["_last_submit_raw"] = raw_msg
            # Clean message for agent
            agent_msg = raw_msg.replace("[exec_err_flg] ", "")
            parts = [agent_msg]
            if data.get("reward", 0) > 0:
                parts.append(f"Reward: {data['reward']}")
            if data.get("has_follow_up"):
                parts.append(f"Follow-up question: {data['follow_up_query']}")
            budget = tool_context.state.get("budget_remaining", 0)
            parts.append(f"Budget remaining: {budget} bird-coins")
            return "\n".join(parts)
    except Exception as e:
        return f"Error: {e}"


# ── Build tool list for ADK Agent ──

def get_ainteract_tools():
    """Return list of FunctionTool instances for a-interact mode."""
    return [
        FunctionTool(execute_sql),
        FunctionTool(get_schema),
        FunctionTool(get_all_column_meanings),
        FunctionTool(get_column_meaning),
        FunctionTool(get_all_external_knowledge_names),
        FunctionTool(get_knowledge_definition),
        FunctionTool(get_all_knowledge_definitions),
        FunctionTool(ask_user),
        FunctionTool(submit_sql),
    ]
