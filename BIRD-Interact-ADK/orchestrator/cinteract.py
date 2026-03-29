"""BIRD-Interact ADK Orchestrator - c-interact conversational pipeline.

Uses the same ADK agent as a-interact but with only 2 tools (ask_user, submit_sql).
The orchestrator drives the phase structure by sending messages to the ADK session:
  Phase 1: Clarification + submit (agent loops with ask_user / submit_sql)
  Phase 1 Debug: One more chance (if failed)
  Phase 2: Follow-up question (if applicable)
  Phase 2 Debug: One more chance (if failed)

ALL three components are behind FastAPI ports:
  - System Agent:    port 6000 (ADK session with tools)
  - User Simulator:  port 6001 (encoder-decoder)
  - DB Environment:  port 6002 (SQL execution, evaluation, schema, knowledge)
"""

import asyncio
import json
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import httpx
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Service URLs ──

SYSTEM_AGENT_URL = f"http://localhost:{settings.system_agent_port}"
USER_SIM_URL = f"http://localhost:{settings.user_sim_port}"
DB_ENV_URL = f"http://localhost:{settings.db_env_port}"


async def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Service helpers ──

async def init_task_on_services(task_id: str, task_data: dict):
    payload = {"task_id": task_id, "task_data": {**task_data, "_interact_mode": "c-interact"}}
    await _post(f"{DB_ENV_URL}/init_task", payload)
    await _post(f"{USER_SIM_URL}/init_task", payload)


async def get_schema_service(task_id: str) -> str:
    data = await _post(f"{DB_ENV_URL}/schema", {"task_id": task_id})
    return data.get("schema", "")


async def get_knowledge_service(task_id: str) -> str:
    data = await _post(f"{DB_ENV_URL}/knowledge", {"task_id": task_id})
    return data.get("knowledge", "[]")


async def phase_transition_service(task_id: str):
    await _post(f"{USER_SIM_URL}/phase_transition", {"task_id": task_id})


async def cleanup_task_service(task_id: str):
    try:
        await _post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=30.0)
    except Exception as e:
        logger.warning("Cleanup failed for %s: %s", task_id, e)


# ── ADK session helpers ──

async def init_agent_session(task_id: str, state: dict):
    return await _post(f"{SYSTEM_AGENT_URL}/init_session", {
        "task_id": task_id,
        "mode": "c-interact",
        "state": state,
    })


async def run_agent_session(task_id: str, message: str) -> dict:
    return await _post(
        f"{SYSTEM_AGENT_URL}/run_session",
        {"task_id": task_id, "mode": "c-interact", "message": message},
        timeout=600.0,
    )


# ── Main pipeline ──

async def run_single_task(task_data: dict) -> Dict[str, Any]:
    instance_id = task_data["instance_id"]
    db_name = task_data["selected_database"]
    logger.info("Starting task: %s (db: %s)", instance_id, db_name)
    start_time = time.time()

    # Init services
    await init_task_on_services(instance_id, task_data)

    try:
        # Get schema + knowledge for the instruction
        db_schema = await get_schema_service(instance_id)
        external_kg = await get_knowledge_service(instance_id)

        # c-interact clarification budget (per task, paper Section 3.1):
        # max_turn = n_ambiguities + patience (number of ask_user rounds allowed)
        n_critical = len(task_data.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
        n_knowledge = len(task_data.get("knowledge_ambiguity", []))
        max_turn = n_critical + n_knowledge + settings.patience

        # Init ADK session with state (instruction placeholders filled from state)
        session_state = {
            "task_id": instance_id,
            "mode": "c-interact",
            "db_name": db_name,
            "db_schema": db_schema,
            "external_kg": external_kg,
            "max_turn": max_turn,
            "phase_max_turns": max_turn * 3,
            "model_turns": 0,
            "tool_trajectory": [],
            "dialogue_history": [],
        }
        await init_agent_session(instance_id, session_state)
        all_adk_events = []

        # ── Phase 1: Clarify + Submit ──
        logger.info("  [%s] Phase 1: %d clarification turns", instance_id, max_turn)
        phase1_msg = (
            f"User Query:\n{task_data.get('amb_user_query', '')}\n\n"
            f"You have {max_turn} clarification turns. "
            f"Ask questions with ask_user to resolve ambiguities, "
            f"then call submit_sql with your final PostgreSQL query."
        )
        result = await run_agent_session(instance_id, phase1_msg)
        state = result.get("state", {})
        all_adk_events.extend(state.get("adk_events", []))
        p1_passed = state.get("phase1_completed", False)
        logger.info("  [%s] Phase 1: %s", instance_id, "PASS" if p1_passed else "FAIL")

        # ── Phase 1 Debug ──
        debug_passed = False
        if not p1_passed:
            logger.info("  [%s] Phase 1 Debug...", instance_id)
            raw_msg = state.get("_last_submit_raw", "")

            if "[exec_err_flg]" in raw_msg:
                err_detail = raw_msg.split("[exec_err_flg] ")[-1]
                debug_msg = f"Your SQL is not executable: {err_detail}\nPlease fix and call submit_sql."
            else:
                debug_msg = "Your SQL is not correct. You have one more chance. Please fix and call submit_sql."

            result = await run_agent_session(instance_id, debug_msg)
            state = result.get("state", {})
            all_adk_events.extend(state.get("adk_events", []))
            debug_passed = state.get("phase1_completed", False)
            logger.info("  [%s] Phase 1 Debug: %s", instance_id, "PASS" if debug_passed else "FAIL")

        # ── Phase 2: Follow-up ──
        has_follow_up = bool(task_data.get("follow_up") and task_data["follow_up"].get("sol_sql"))
        p2_passed = False
        best_p1 = p1_passed or debug_passed

        if best_p1 and has_follow_up:
            follow_up_query = task_data["follow_up"].get("query", "")
            logger.info("  [%s] Phase 2: Follow-up", instance_id)
            await phase_transition_service(instance_id)

            fu_msg = (
                f"Phase 1 is complete. Here is a follow-up question:\n\n{follow_up_query}\n\n"
                f"Generate the PostgreSQL query and call submit_sql."
            )
            result = await run_agent_session(instance_id, fu_msg)
            state = result.get("state", {})
            all_adk_events.extend(state.get("adk_events", []))
            p2_passed = state.get("phase2_completed", False)
            logger.info("  [%s] Phase 2: %s", instance_id, "PASS" if p2_passed else "FAIL")

            # ── Phase 2 Debug ──
            if not p2_passed:
                logger.info("  [%s] Phase 2 Debug...", instance_id)

                raw_msg = state.get("_last_submit_raw", "")

                if "[exec_err_flg]" in raw_msg:
                    err_detail = raw_msg.split("[exec_err_flg] ")[-1]
                    p2_debug_msg = f"Your SQL is not executable: {err_detail}\nPlease fix and call submit_sql."
                else:
                    p2_debug_msg = "Your SQL is not correct. You have one more chance. Please fix and call submit_sql."

                result = await run_agent_session(instance_id, p2_debug_msg)
                state = result.get("state", {})
                all_adk_events.extend(state.get("adk_events", []))
                p2_passed = state.get("phase2_completed", False)
                logger.info("  [%s] Phase 2 Debug: %s", instance_id, "PASS" if p2_passed else "FAIL")

        # ── Results ──
        elapsed = time.time() - start_time
        total_reward = state.get("total_reward", 0.0)

        result = {
            "task_id": instance_id,
            "instance_id": instance_id,
            "database": db_name,
            "phase1_passed": best_p1,
            "phase2_passed": p2_passed,
            "has_follow_up": has_follow_up,
            "total_reward": total_reward,
            "elapsed_seconds": elapsed,
            "tool_trajectory": state.get("tool_trajectory", []),
            "dialogue_history": state.get("dialogue_history", []),
            "adk_events": all_adk_events,
        }
        logger.info("Task %s done. Reward: %.2f, Time: %.1fs", instance_id, total_reward, elapsed)
        return result
    finally:
        await cleanup_task_service(instance_id)


# ── Batch evaluation ──

async def run_evaluation(data_path: str, output_path: str, limit: int = None):
    tasks = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    if limit:
        tasks = tasks[:limit]
    logger.info("C-Interact: Evaluating %d tasks", len(tasks))

    results = []
    total_reward = 0.0
    p1_count = 0
    p2_count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for i, td in enumerate(tasks):
        logger.info("=== Task %d/%d: %s ===", i + 1, len(tasks), td["instance_id"])
        try:
            r = await run_single_task(td)
            results.append(r)
            total_reward += r["total_reward"]
            if r["phase1_passed"]:
                p1_count += 1
            if r["phase2_passed"]:
                p2_count += 1
        except Exception as e:
            logger.error("Error: %s: %s", td["instance_id"], e)
            traceback.print_exc()
            results.append({"task_id": td["instance_id"], "error": str(e), "total_reward": 0})

        # Save periodically
        if (i + 1) % 5 == 0 or i == len(tasks) - 1:
            n = len(results)
            output = {
                "metrics": {
                    "total_tasks": n,
                    "total_reward": total_reward,
                    "average_reward": total_reward / n if n else 0,
                    "phase1_rate": p1_count / n if n else 0,
                    "phase2_rate": p2_count / n if n else 0,
                    "phase1_count": p1_count,
                    "phase2_count": p2_count,
                },
                "results": results,
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2, default=str)

    n = len(tasks)
    logger.info(
        "\nDone! Tasks: %d, Avg Reward: %.4f, P1: %d/%d (%.1f%%), P2: %d/%d",
        n, total_reward / n if n else 0, p1_count, n, p1_count / n * 100 if n else 0, p2_count, n,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=settings.data_path)
    parser.add_argument("--output", default="results/eval_cinteract.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.data, args.output, args.limit))


if __name__ == "__main__":
    main()
