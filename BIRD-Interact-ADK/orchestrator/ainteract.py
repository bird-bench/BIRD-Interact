"""BIRD-Interact ADK Orchestrator - a-interact agent pipeline.

This version delegates the full tool-use loop to the ADK-backed system-agent
service on port 6000. The orchestrator only:
1. Initializes the DB environment and user simulator services
2. Initializes an agent session on the system-agent service
3. Sends the initial user request once
4. Reads the final session state for metrics
"""

import argparse
import asyncio
import json
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

SYSTEM_AGENT_URL = f"http://localhost:{settings.system_agent_port}"
USER_SIM_URL = f"http://localhost:{settings.user_sim_port}"
DB_ENV_URL = f"http://localhost:{settings.db_env_port}"


async def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def calculate_initial_budget(task_data: Dict[str, Any]) -> float:
    """a-interact budget in bird-coins (per task, paper Section 3.2).

    Formula: 6 + 2 * m_amb + 2 * patience
      - 6 = ENV_INTERACT(3) + SUBMIT(3) base budget
      - 2 * m_amb = one ask_user (cost=2) per ambiguity point
      - 2 * patience = extra exploration tolerance
      - patience=3 in config = patience_budget=6 in reference (we multiply by 2)
    """
    critical = len(task_data.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
    knowledge = len(task_data.get("knowledge_ambiguity", []))
    m_amb = critical + knowledge
    return 6.0 + 2.0 * m_amb + 2.0 * settings.patience


async def init_task_on_services(task_id: str, task_data: dict):
    payload = {
        "task_id": task_id,
        "task_data": {**task_data, "_interact_mode": "a-interact"},
    }
    await _post(f"{DB_ENV_URL}/init_task", payload)
    await _post(f"{USER_SIM_URL}/init_task", payload)
    logger.info("  [%s] Services initialized", task_id)


async def init_agent_session(task_id: str, task_data: dict, budget: float):
    state = {
        "task_id": task_id,
        "db_name": task_data["selected_database"],
        "user_query": task_data.get("amb_user_query", ""),
        "current_phase": 1,
        "budget_remaining": budget,
        "initial_budget": budget,
        "total_reward": 0.0,
        "dialogue_history": [],
        "tool_trajectory": [],
        "adk_events": [],
        "phase1_completed": False,
        "phase2_completed": False,
        "task_done": False,
    }
    return await _post(
        f"{SYSTEM_AGENT_URL}/init_session",
        {"task_id": task_id, "mode": "a-interact", "state": state, "reset": True},
        timeout=30.0,
    )


async def run_agent_session(task_id: str, message: str):
    return await _post(
        f"{SYSTEM_AGENT_URL}/run_session",
        {"task_id": task_id, "mode": "a-interact", "message": message},
        timeout=1800.0,
    )


async def cleanup_task_service(task_id: str):
    try:
        await _post(f"{DB_ENV_URL}/cleanup_task", {"task_id": task_id}, timeout=30.0)
    except Exception as e:
        logger.warning("Cleanup failed for %s: %s", task_id, e)


async def run_single_task(task_data: dict) -> Dict[str, Any]:
    instance_id = task_data["instance_id"]
    db_name = task_data["selected_database"]
    logger.info("Starting task: %s (db: %s)", instance_id, db_name)
    start_time = time.time()

    await init_task_on_services(instance_id, task_data)

    try:
        initial_budget = calculate_initial_budget(task_data)
        await init_agent_session(instance_id, task_data, initial_budget)

        initial_message = (
            f"Database: {db_name}\n"
            f"Task ID: {instance_id}\n\n"
            f"User Query:\n{task_data.get('amb_user_query', '')}\n\n"
            f"You have a budget of {initial_budget:.1f} bird-coins. "
            f"Use your tools to explore the database, clarify ambiguities with the user, "
            f"and submit your final SQL efficiently."
        )

        run_result = await run_agent_session(instance_id, initial_message)
        state = run_result.get("state", {})
        elapsed = time.time() - start_time

        result = {
            "task_id": instance_id,
            "instance_id": instance_id,
            "database": db_name,
            "phase1_passed": state.get("phase1_completed", False),
            "phase2_passed": state.get("phase2_completed", False),
            "has_follow_up": bool(task_data.get("follow_up") and task_data["follow_up"].get("sol_sql")),
            "total_reward": state.get("total_reward", 0.0),
            "elapsed_seconds": elapsed,
            "budget_used": initial_budget - max(0, state.get("budget_remaining", initial_budget)),
            "budget_remaining": max(0, state.get("budget_remaining", initial_budget)),
            "dialogue_history": state.get("dialogue_history", []),
            "tool_trajectory": state.get("tool_trajectory", []),
            "adk_events": state.get("adk_events", []),
            "final_response": run_result.get("response", ""),
        }
        logger.info(
            "Task %s done. Reward: %.2f, Budget used: %.1f, Time: %.1fs",
            instance_id,
            result["total_reward"],
            result["budget_used"],
            elapsed,
        )
        return result
    finally:
        await cleanup_task_service(instance_id)


async def run_evaluation(data_path: str, output_path: str, limit: int = None):
    tasks = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    if limit:
        tasks = tasks[:limit]
    logger.info("A-Interact: Evaluating %d tasks", len(tasks))

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

        if (i + 1) % 5 == 0 or i == len(tasks) - 1:
            n = len(results)
            output = {
                "mode": "a-interact",
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
    if n:
        logger.info(
            "\nDone! Tasks: %d, Avg Reward: %.4f, P1: %d/%d (%.1f%%), P2: %d/%d (%.1f%%)",
            n,
            total_reward / n,
            p1_count,
            n,
            p1_count / n * 100,
            p2_count,
            n,
            p2_count / n * 100,
        )


def main():
    parser = argparse.ArgumentParser(description="BIRD-Interact a-interact evaluation")
    parser.add_argument("--data", default=settings.data_path)
    parser.add_argument("--output", default="results/eval_ainteract.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.data, args.output, args.limit))


if __name__ == "__main__":
    main()
