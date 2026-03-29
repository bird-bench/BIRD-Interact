"""Simulated agent test harness — tests the full pipeline without LLM calls.

Directly calls service endpoints in the same order an agent would,
using ground-truth SQL to guarantee passes. Catches connection pool issues,
phase transition bugs, and endpoint errors much faster than real agent runs.

Usage:
    python -m orchestrator.test_harness --limit 30 --concurrency 5
    python -m orchestrator.test_harness --tasks alien_1,alien_M_1
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from shared.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_ENV = f"http://localhost:{settings.db_env_port}"
USER_SIM = f"http://localhost:{settings.user_sim_port}"
SYSTEM_AGENT = f"http://localhost:{settings.system_agent_port}"


async def _post(url, payload, timeout=60.0):
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as c:
        r = await c.post(url, json=payload)
        r.raise_for_status()
        return r.json()


async def simulate_task(td: dict) -> Dict:
    """Simulate a full agent interaction for one task."""
    task_id = td["instance_id"]
    db_name = td["selected_database"]
    category = td.get("category", "Query")
    sol_sql = td.get("sol_sql", [])
    if isinstance(sol_sql, str):
        sol_sql = [sol_sql]
    fu = td.get("follow_up", {})
    fu_sql = fu.get("sol_sql", [])
    if isinstance(fu_sql, str):
        fu_sql = [fu_sql]
    has_follow_up = bool(fu and fu_sql)

    issues = []
    steps = []
    p1_passed = False
    p2_passed = False

    def check(name, condition, detail=""):
        if not condition:
            issues.append(f"{name}: {detail}")
        steps.append({"step": name, "ok": condition, "detail": detail})

    try:
        # 1. Init task
        r = await _post(f"{DB_ENV}/init_task", {
            "task_id": task_id,
            "task_data": {**td, "_interact_mode": "a-interact"},
        })
        check("init_task", r.get("status") == "ok", f"status={r.get('status')}")

        # 2. Schema
        r = await _post(f"{DB_ENV}/schema", {"task_id": task_id})
        schema = r.get("schema", "")
        check("get_schema", len(schema) > 100, f"len={len(schema)}")

        # 3. Knowledge
        r = await _post(f"{DB_ENV}/knowledge", {"task_id": task_id})
        knowledge = r.get("knowledge", "")
        check("get_knowledge", len(knowledge) > 10, f"len={len(knowledge)}")

        # 4. Execute SQL — simple SELECT
        r = await _post(f"{DB_ENV}/execute", {"task_id": task_id, "sql": "SELECT 1 AS test"})
        check("execute_select_1", r.get("success") is True, f"success={r.get('success')}, error={r.get('error')}")

        # 5. Execute SQL — query actual table
        r = await _post(f"{DB_ENV}/execute", {"task_id": task_id, "sql": f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'"})
        check("execute_count_tables", r.get("success") is True, f"result={r.get('result','')[:100]}, error={r.get('error')}")

        # 6. Execute SQL — non-SELECT should be blocked
        r = await _post(f"{DB_ENV}/execute", {"task_id": task_id, "sql": "DROP TABLE IF EXISTS nonexistent"})
        check("execute_non_select_blocked", r.get("success") is False, f"success={r.get('success')}, error={r.get('error','')[:80]}")

        # 7. Execute SQL — SELECT with comment prefix (was a bug)
        r = await _post(f"{DB_ENV}/execute", {"task_id": task_id, "sql": "-- test comment\nSELECT 1"})
        check("execute_comment_prefix", r.get("success") is True, f"success={r.get('success')}, error={r.get('error')}")

        # 8. Init user sim
        r = await _post(f"{USER_SIM}/init_task", {"task_id": task_id, "task_data": td})
        check("init_user_sim", r.get("status") == "ok")

        # 9. Ask user
        r = await _post(f"{USER_SIM}/ask", {"task_id": task_id, "question": "What does the main metric mean in this context?"})
        answer = r.get("answer", "")
        check("ask_user_response", len(answer) > 50, f"len={len(answer)}")
        check("ask_user_not_truncated", not answer.endswith("...") and len(answer) > 100, f"ends=...{answer[-30:]}")

        # 10. Execute SQL — multiple queries to test connection reuse
        for i in range(3):
            r = await _post(f"{DB_ENV}/execute", {"task_id": task_id, "sql": f"SELECT {i+1} AS n"})
            check(f"execute_reuse_{i}", r.get("success") is True, f"error={r.get('error')}")

        # 11. Submit P1 — use ground truth SQL
        p1_passed = False
        if sol_sql:
            r = await _post(f"{DB_ENV}/submit", {"task_id": task_id, "sql": sol_sql[0]})
            p1_passed = r.get("passed", False)
            msg = r.get("message", "")
            check("submit_p1", True, f"passed={p1_passed}, msg={msg[:120]}")
            check("submit_p1_no_exec_err_flg", "[exec_err_flg]" not in msg, msg[:200])
            check("submit_p1_no_test_case_leaked", "ex_base returned" not in msg and "Test case failed" not in msg, msg[:200])

            if p1_passed:
                check("submit_p1_reward", "Reward: 0.7" in msg, msg[:200])
                if has_follow_up:
                    check("submit_p1_follow_up", r.get("has_follow_up") is True and bool(r.get("follow_up_query")),
                          f"has_follow_up={r.get('has_follow_up')}, query={r.get('follow_up_query','')[:80]}")

        # 12. Execute after P1 submit — test connection still works
        r = await _post(f"{DB_ENV}/execute", {"task_id": task_id, "sql": "SELECT 1 AS post_submit"})
        check("execute_after_submit", r.get("success") is True, f"error={r.get('error')}")

        # 13. Submit P2
        p2_passed = False
        if p1_passed and has_follow_up and fu_sql:
            # Phase transition on user sim
            await _post(f"{USER_SIM}/phase_transition", {"task_id": task_id})

            r = await _post(f"{DB_ENV}/submit", {"task_id": task_id, "sql": fu_sql[0]})
            p2_passed = r.get("passed", False)
            msg = r.get("message", "")
            check("submit_p2", True, f"passed={p2_passed}, msg={msg[:120]}")
            if p2_passed:
                check("submit_p2_reward", "Reward: 0.3" in msg, msg[:200])

        # 14. Submit wrong SQL — test failure message
        r = await _post(f"{DB_ENV}/submit", {"task_id": task_id, "sql": "SELECT 999999 AS wrong"})
        wrong_msg = r.get("message", "")
        check("submit_wrong_clean_msg", "Your SQL is not correct" in wrong_msg or "Error executing" in wrong_msg,
              f"msg={wrong_msg[:150]}")

        # 15. Execute after failed submit — connection should still work
        r = await _post(f"{DB_ENV}/execute", {"task_id": task_id, "sql": "SELECT 1 AS post_fail"})
        check("execute_after_failed_submit", r.get("success") is True, f"error={r.get('error')}")

        # 16. Cleanup
        r = await _post(f"{DB_ENV}/cleanup_task", {"task_id": task_id})
        check("cleanup", r.get("status") == "ok")

    except Exception as e:
        issues.append(f"EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            await _post(f"{DB_ENV}/cleanup_task", {"task_id": task_id})
        except Exception:
            pass

    return {
        "task_id": task_id,
        "database": db_name,
        "category": category,
        "p1_passed": p1_passed if sol_sql else None,
        "p2_passed": p2_passed if has_follow_up else None,
        "issues": issues,
        "steps": steps,
        "total_checks": len(steps),
        "passed_checks": sum(1 for s in steps if s["ok"]),
    }


async def run_harness(tasks: List[dict], concurrency: int = 5):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _run(td):
        async with sem:
            return await simulate_task(td)

    all_results = await asyncio.gather(*[_run(td) for td in tasks])

    total_checks = 0
    passed_checks = 0
    tasks_with_issues = 0

    for r in all_results:
        total_checks += r["total_checks"]
        passed_checks += r["passed_checks"]
        if r["issues"]:
            tasks_with_issues += 1
            logger.warning(f"{r['task_id']} ({r['database']}): {len(r['issues'])} issues")
            for issue in r["issues"]:
                logger.warning(f"  - {issue}")

    p1_tested = sum(1 for r in all_results if r["p1_passed"] is not None)
    p1_passed = sum(1 for r in all_results if r["p1_passed"] is True)
    p2_tested = sum(1 for r in all_results if r["p2_passed"] is not None)
    p2_passed = sum(1 for r in all_results if r["p2_passed"] is True)

    print(f"\n{'='*60}")
    print(f"Test Harness Results: {len(tasks)} tasks, concurrency={concurrency}")
    print(f"{'='*60}")
    print(f"Checks: {passed_checks}/{total_checks} passed ({passed_checks/total_checks*100:.1f}%)")
    print(f"Tasks with issues: {tasks_with_issues}/{len(tasks)}")
    print(f"Oracle P1: {p1_passed}/{p1_tested} ({p1_passed/p1_tested*100:.1f}%)" if p1_tested else "")
    print(f"Oracle P2: {p2_passed}/{p2_tested} ({p2_passed/p2_tested*100:.1f}%)" if p2_tested else "")

    if tasks_with_issues == 0:
        print("\nAll checks passed!")
    else:
        print(f"\n{tasks_with_issues} task(s) had issues — see warnings above")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Simulated agent test harness")
    parser.add_argument("--data", default=settings.data_path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task IDs")
    args = parser.parse_args()

    with open(args.data) as f:
        all_tasks = [json.loads(l) for l in f if l.strip()]

    if args.tasks:
        task_ids = args.tasks.split(",")
        tasks = [t for t in all_tasks if t["instance_id"] in task_ids]
    else:
        tasks = all_tasks[:args.limit] if args.limit else all_tasks

    logger.info(f"Testing {len(tasks)} tasks with concurrency={args.concurrency}")
    start = time.time()
    asyncio.run(run_harness(tasks, args.concurrency))
    print(f"Total time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
