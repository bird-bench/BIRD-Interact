"""ADK callbacks for a-interact mode: budget management and turn limiting."""

import json
import logging
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from shared.config import settings

logger = logging.getLogger(__name__)

MAX_MODEL_TURNS = 60

TOOL_COSTS = {
    "execute_sql": 1.0,
    "get_schema": 1.0,
    "get_all_column_meanings": 1.0,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1.0,
    "ask_user": 2.0,
    "submit_sql": 3.0,
}


def _preview(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return text[:limit] + "...<truncated>" if len(text) > limit else text


async def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Cap LLM invocations at MAX_MODEL_TURNS."""
    turns = callback_context.state.get("model_turns", 0) + 1
    callback_context.state["model_turns"] = turns
    if turns > MAX_MODEL_TURNS:
        logger.warning("Max model turns (%d) reached, forcing stop.", MAX_MODEL_TURNS)
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(
                    text="Maximum interaction turns reached. Task ended."
                )],
            ),
        )

    if callback_context.state.get("task_done", False):
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Task completed.")],
            ),
        )

    budget = callback_context.state.get("budget_remaining", None)
    if budget is not None and budget < 0:
        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part.from_text(text="Budget exhausted. Task ended.")],
            ),
        )

    return None


async def before_tool_callback(
    tool, args: dict, tool_context: ToolContext
) -> dict | None:
    """Deduct budget. Free submit exit when exhausted."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    cost = TOOL_COSTS.get(tool_name)
    if cost is None:
        return None

    budget = tool_context.state.get("budget_remaining", 0)

    if budget < cost:
        tool_context.state["_budget_before"] = budget
        if tool_name == "submit_sql":
            tool_context.state["budget_remaining"] = -1
            return None  # free exit, -1 signals stop after this
        return {
            "error": f"Budget exhausted ({budget:.1f} remaining). "
            "You MUST call submit_sql now with your best SQL."
        }

    tool_context.state["_budget_before"] = budget
    remaining = budget - cost
    # After submit drains budget to 0, signal stop with -1
    if tool_name == "submit_sql" and remaining <= 0:
        remaining = -1
    tool_context.state["budget_remaining"] = remaining
    return None


async def after_tool_callback(
    tool, args: dict, tool_context: ToolContext, tool_response
) -> dict | None:
    """Record tool event in trajectory and append budget note to response."""
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    cost = TOOL_COSTS.get(tool_name, 0)
    budget_before = tool_context.state.get("_budget_before")
    budget_after = tool_context.state.get("budget_remaining")
    initial = tool_context.state.get("initial_budget", 0)

    trajectory = tool_context.state.get("tool_trajectory", [])
    trajectory.append({
        "type": "tool",
        "tool": tool_name,
        "args": args,
        "result": _preview(tool_response),
        "cost": cost,
        "budget_before": budget_before,
        "budget_after": budget_after,
    })
    tool_context.state["tool_trajectory"] = trajectory

    # Append budget note to agent-visible response (matches reference implementation)
    if budget_after is not None and budget_after >= 0:
        budget_note = f"\n\n[SYSTEM NOTE: Remaining budget: {budget_after:.1f}/{initial:.1f}]"
        return str(tool_response) + budget_note
    return None
