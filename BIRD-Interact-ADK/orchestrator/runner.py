"""Unified parallel evaluation runner for BIRD-Interact benchmark."""

import asyncio
import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Awaitable, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_parallel_evaluation(
    tasks: List[dict],
    run_single_task: Callable[[dict], Awaitable[Dict[str, Any]]],
    output_path: str,
    concurrency: int = 5,
    mode: str = "a-interact",
):
    semaphore = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []
    results_lock = asyncio.Lock()
    total_reward = 0.0
    p1_count = 0
    p2_count = 0
    completed = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    async def _save():
        n = len(results)
        if n == 0:
            return
        output = {
            "mode": mode,
            "metrics": {
                "total_tasks": n,
                "total_reward": total_reward,
                "average_reward": total_reward / n,
                "phase1_rate": p1_count / n,
                "phase2_rate": p2_count / n,
                "phase1_count": p1_count,
                "phase2_count": p2_count,
            },
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

    async def _run_one(i: int, td: dict):
        nonlocal total_reward, p1_count, p2_count, completed
        instance_id = td["instance_id"]
        async with semaphore:
            logger.info("=== Task %d/%d: %s ===", i + 1, len(tasks), instance_id)
            try:
                r = await run_single_task(td)
            except Exception as e:
                logger.error("Error: %s: %s", instance_id, e)
                traceback.print_exc()
                r = {"task_id": instance_id, "error": str(e), "total_reward": 0}

        async with results_lock:
            results.append(r)
            total_reward += r.get("total_reward", 0)
            if r.get("phase1_passed"):
                p1_count += 1
            if r.get("phase2_passed"):
                p2_count += 1
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                await _save()

    await asyncio.gather(*[_run_one(i, td) for i, td in enumerate(tasks)])
    await _save()

    n = len(tasks)
    if n:
        logger.info(
            "\nDone! Tasks: %d, Avg Reward: %.4f, P1: %d/%d (%.1f%%), P2: %d/%d (%.1f%%)",
            n, total_reward / n, p1_count, n, p1_count / n * 100,
            p2_count, n, p2_count / n * 100,
        )


def load_tasks(data_path: str, limit: int = None) -> List[dict]:
    tasks = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    if limit:
        tasks = tasks[:limit]
    return tasks


async def run_oracle_task(task_data: dict) -> Dict[str, Any]:
    """Submit ground-truth SQL directly — no LLM, tests evaluation pipeline."""
    import httpx
    db_env = f"http://localhost:{settings.db_env_port}"
    user_sim = f"http://localhost:{settings.user_sim_port}"

    async def _post(url, payload, timeout=60.0):
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c:
            r = await c.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    instance_id = task_data["instance_id"]
    sol_sql = task_data.get("sol_sql", [])
    if isinstance(sol_sql, str):
        sol_sql = [sol_sql]
    fu = task_data.get("follow_up", {})
    fu_sql = fu.get("sol_sql", [])
    if isinstance(fu_sql, str):
        fu_sql = [fu_sql]
    has_follow_up = bool(fu and fu_sql)

    await _post(f"{db_env}/init_task", {
        "task_id": instance_id,
        "task_data": {**task_data, "_interact_mode": "a-interact"},
    })

    try:
        p1_passed = False
        p2_passed = False
        total_reward = 0.0

        if sol_sql:
            r1 = await _post(f"{db_env}/submit", {"task_id": instance_id, "sql": sol_sql[0]})
            p1_passed = r1.get("passed", False)
            if p1_passed:
                total_reward += r1.get("reward", 0.0)

            if p1_passed and has_follow_up:
                try:
                    await _post(f"{user_sim}/init_task", {"task_id": instance_id, "task_data": task_data})
                    await _post(f"{user_sim}/phase_transition", {"task_id": instance_id})
                except Exception:
                    pass
                r2 = await _post(f"{db_env}/submit", {"task_id": instance_id, "sql": fu_sql[0]})
                p2_passed = r2.get("passed", False)
                if p2_passed:
                    total_reward += r2.get("reward", 0.0)

        return {
            "task_id": instance_id,
            "instance_id": instance_id,
            "database": task_data["selected_database"],
            "phase1_passed": p1_passed,
            "phase2_passed": p2_passed,
            "has_follow_up": has_follow_up,
            "total_reward": total_reward,
        }
    finally:
        try:
            await _post(f"{db_env}/cleanup_task", {"task_id": instance_id})
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="BIRD-Interact parallel evaluation")
    parser.add_argument("--mode", choices=["a-interact", "c-interact", "oracle"], default="a-interact")
    parser.add_argument("--data", default=settings.data_path)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    output = args.output or f"results/eval_{args.mode.replace('-', '_')}.json"

    if args.mode == "oracle":
        run_single_task = run_oracle_task
    elif args.mode == "a-interact":
        from orchestrator.ainteract import run_single_task
    else:
        from orchestrator.cinteract import run_single_task

    tasks = load_tasks(args.data, args.limit)
    logger.info("%s: Evaluating %d tasks with concurrency=%d", args.mode, len(tasks), args.concurrency)

    asyncio.run(run_parallel_evaluation(
        tasks=tasks,
        run_single_task=run_single_task,
        output_path=output,
        concurrency=args.concurrency,
        mode=args.mode,
    ))


if __name__ == "__main__":
    main()
