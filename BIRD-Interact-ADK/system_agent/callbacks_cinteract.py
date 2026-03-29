"""ADK callbacks for c-interact mode: fixed workflow enforcement.

After the agent calls submit_sql, the next before_model_callback
returns a forced LlmResponse to end the run. This gives the agent
exactly 1 submit per run_session call. The orchestrator drives
phases by sending separate messages (debug, follow-up).
"""

import json
import logging
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

MAX_MODEL_TURNS = 60


def _preview(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return text[:limit] + "...<truncated>" if len(text) > limit else text


async def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Stop the agent after it has submitted this phase."""
    turns = callback_context.state.get("model_turns", 0) + 1
    callback_context.state["model_turns"] = turns

    if turns > MAX_MODEL_TURNS:
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Maximum turns reached. Task ended.")],
            ),
        )

    if callback_context.state.get("task_done", False):
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Task completed.")],
            ),
        )

    # Stop after submit — flag is reset by adk_runtime before each run_session
    if callback_context.state.get("_submitted_this_phase", False):
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="SQL submitted. Awaiting result.")],
            ),
        )

    return None


async def before_tool_callback(
    tool, args: dict, tool_context: ToolContext
) -> dict | None:
    """No gating needed — before_model_callback handles stop."""
    return None


async def after_tool_callback(
    tool, args: dict, tool_context: ToolContext, tool_response
) -> dict | None:
    """Record tool event + mark phase as submitted after submit_sql."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)

    trajectory = tool_context.state.get("tool_trajectory", [])
    trajectory.append({
        "type": "tool",
        "tool": tool_name,
        "args": args,
        "result": _preview(tool_response),
    })
    tool_context.state["tool_trajectory"] = trajectory

    if tool_name == "submit_sql":
        tool_context.state["_submitted_this_phase"] = True

    # Show remaining clarification turns after ask_user
    if tool_name == "ask_user":
        max_turn = tool_context.state.get("max_turn", 0)
        asks_used = sum(1 for t in trajectory if t.get("tool") == "ask_user")
        remaining = max(0, max_turn - asks_used)
        return str(tool_response) + f"\n\n[SYSTEM NOTE: Clarification turns remaining: {remaining}/{max_turn}]"

    return None
