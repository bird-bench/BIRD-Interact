"""DB Environment Service (Port 6002). SQL execution, submission, schema/knowledge."""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from shared.config import settings
from shared.db_utils import (
    _get_or_init_pool, close_pool, execute_queries,
    reset_and_restore_database, test_case_default,
    ex_base, remove_distinct, remove_comments, remove_round,
    create_task_db, reset_task_db, drop_task_db,
)
from shared.models import (
    ExecuteSQLRequest, ExecuteSQLResponse, InitTaskRequest,
    SchemaRequest, ColumnMeaningRequest, KnowledgeRequest,
    SubmitSQLRequest, SubmitSQLResponse,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="BIRD-Interact DB Environment", version="1.0.0")

MAX_RESULT_LENGTH = 500
KNOWLEDGE_VISIBLE_FIELDS = ["id", "knowledge", "description", "definition"]

_task_data: Dict[str, Dict[str, Any]] = {}
_schema_cache: Dict[str, str] = {}
_column_meanings_cache: Dict[str, Dict] = {}
_external_knowledge_cache: Dict[str, Dict] = {}
_submit_attempts: Dict[str, Dict[int, int]] = {}
_successful_phase1_sql: Dict[str, str] = {}


def _load_db_data(db_name: str):
    if db_name in _schema_cache:
        return
    db_folder = os.path.join(settings.db_data_path, db_name)
    # Schema
    try:
        with open(os.path.join(db_folder, f"{db_name}_schema.txt")) as f:
            _schema_cache[db_name] = f.read()
    except Exception as e:
        logger.error(f"Schema load failed for {db_name}: {e}")
        _schema_cache[db_name] = "Schema not available"
    # Column meanings
    try:
        with open(os.path.join(db_folder, f"{db_name}_column_meaning_base.json")) as f:
            raw = json.load(f)
        _column_meanings_cache[db_name] = {k.lower(): v for k, v in raw.items()}
    except Exception as e:
        logger.error(f"Column meanings load failed for {db_name}: {e}")
        _column_meanings_cache[db_name] = {}
    # Knowledge
    try:
        kb = {}
        with open(os.path.join(db_folder, f"{db_name}_kb.jsonl")) as f:
            for line in f:
                if not line.strip(): continue
                entry = json.loads(line.strip())
                kb[entry["knowledge"]] = entry
        _external_knowledge_cache[db_name] = kb
    except Exception as e:
        logger.error(f"Knowledge load failed for {db_name}: {e}")
        _external_knowledge_cache[db_name] = {}


def _filter_knowledge(db_name: str, record: Dict) -> Dict:
    full_kb = _external_knowledge_cache.get(db_name, {})
    if not full_kb: return {}
    agent_kb = full_kb.copy()
    deleted_ids = set()
    for amb in record.get("knowledge_ambiguity", []):
        dk = amb.get("deleted_knowledge")
        if dk is not None: deleted_ids.add(dk)
    if deleted_ids:
        to_remove = [k for k, v in agent_kb.items() if v.get("id") in deleted_ids]
        for k in to_remove: del agent_kb[k]
    return agent_kb


def _format_result(result, cursor_desc=None) -> str:
    if result is None: return "Query executed successfully."
    if not isinstance(result, list): return str(result)
    if not result: return "Query executed, empty result set."
    lines = []
    if cursor_desc:
        cols = [desc[0] for desc in cursor_desc]
        lines.append(" | ".join(cols))
        lines.append("-" * min(len(lines[0]), 200))
    for row in result[:100]:
        cells = [str(c)[:100] for c in row]
        lines.append(" | ".join(cells))
    text = "\n".join(lines)
    words = text.split()
    if len(words) > MAX_RESULT_LENGTH:
        text = " ".join(words[:MAX_RESULT_LENGTH]) + "..."
    return text


@app.post("/init_task")
async def init_task(req: InitTaskRequest):
    _task_data[req.task_id] = req.task_data
    _submit_attempts[req.task_id] = {1: 0, 2: 0}
    db_name = req.task_data["selected_database"]
    _load_db_data(db_name)
    task_db = await asyncio.to_thread(create_task_db, db_name, req.task_id)
    req.task_data["_task_db"] = task_db
    return {"status": "ok", "task_id": req.task_id}


def _execute_sql_sync(task_db: str, sql: str) -> ExecuteSQLResponse:
    """Blocking SQL execution — runs in thread pool."""
    try:
        pool = _get_or_init_pool(task_db)
        conn = pool.getconn()
        try:
            # Reset connection if it's in a bad state
            if conn.closed:
                pool.putconn(conn, close=True)
                conn = pool.getconn()
            try:
                conn.reset()
            except Exception as reset_err:
                logger.warning(f"conn.reset() failed for {task_db}: {reset_err}, getting fresh conn")
                try:
                    pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = pool.getconn()
            result, err, timeout, desc = execute_queries(sql, task_db, conn)
            if err:
                return ExecuteSQLResponse(result="", success=False, error=f"SQL error: {err}")
            if timeout:
                return ExecuteSQLResponse(result="", success=False, error="SQL execution timed out")
            formatted = _format_result(result, desc)
            return ExecuteSQLResponse(result=formatted, success=True)
        finally:
            try:
                pool.putconn(conn)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"execute_sql error for {task_db}: {type(e).__name__}: {e}")
        error_msg = str(e) or f"{type(e).__name__}: {repr(e)}"
        return ExecuteSQLResponse(result="", success=False, error=error_msg)


@app.post("/execute", response_model=ExecuteSQLResponse)
async def execute_sql_endpoint(req: ExecuteSQLRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    task_db = td.get("_task_db", td["selected_database"])
    # SELECT-only: prevent agent from modifying DB state (strip comments first)
    sql_cleaned = re.sub(r'--.*$', '', req.sql, flags=re.MULTILINE)
    sql_cleaned = re.sub(r'/\*.*?\*/', '', sql_cleaned, flags=re.DOTALL)
    sql_upper = sql_cleaned.strip().upper()
    if not sql_upper.startswith(("SELECT", "WITH", "EXPLAIN")):
        return ExecuteSQLResponse(result="", success=False, error="Only SELECT queries allowed in execute_sql")
    return await asyncio.to_thread(_execute_sql_sync, task_db, req.sql)


def _submit_sql_sync(req_task_id, req_sql, td, _submit_attempts, _successful_phase1_sql) -> SubmitSQLResponse:
    """Blocking submit logic — runs in thread pool."""
    base_db = td["selected_database"]
    task_db = td.get("_task_db", base_db)
    current_phase = td.get("_current_phase", 1)

    if req_task_id not in _submit_attempts:
        _submit_attempts[req_task_id] = {1: 0, 2: 0}
    _submit_attempts[req_task_id][current_phase] = _submit_attempts[req_task_id].get(current_phase, 0) + 1
    is_first_try = _submit_attempts[req_task_id][current_phase] == 1

    interact_mode = td.get("_interact_mode", "a-interact")
    phase_rewards_first = {1: 0.7, 2: 0.3}
    phase_rewards_debug = {1: 0.5, 2: 0.2}

    try:
        # Reset task DB for clean evaluation
        if current_phase == 1:
            template = f"{base_db}_template"
        else:
            # Phase 2: reset from Phase 1 snapshot (has Phase 1 state applied)
            template = td.get("_snapshot_db", f"{base_db}_template")
        reset_task_db(task_db, template)

        pool = _get_or_init_pool(task_db)
        conn = pool.getconn()
        try:
            if current_phase == 2 and td.get("follow_up"):
                fu = td["follow_up"]
                sol_sqls = fu.get("sol_sql", [])
                test_cases = fu.get("test_cases", [])
                conditions = fu.get("conditions", {})
                category = fu.get("category", "Query")
            else:
                sol_sqls = td.get("sol_sql", [])
                test_cases = td.get("test_cases", [])
                conditions = td.get("conditions", {})
                category = td.get("category", "Query")

            if isinstance(sol_sqls, str): sol_sqls = [sol_sqls]
            pred_sqls = [req_sql] if isinstance(req_sql, str) else req_sql

            passed = False
            message = "Test case execution failed."

            if sol_sqls:
                # Execute pred SQL (also serves as executability check)
                pred_query_result, pred_err, pred_to, _ = execute_queries(pred_sqls, task_db, conn)
                if pred_err:
                    message = f"[exec_err_flg] Error executing submitted SQL: {pred_err}"
                elif pred_to:
                    message = "[exec_err_flg] Submitted SQL execution timed out"
                elif category == "Query" or not test_cases:
                    try:
                        test_case_default(pred_sqls, sol_sqls, task_db, conn, conditions)
                        passed = True
                        message = "SQL passed test case."
                    except AssertionError:
                        message = "Your SQL is not correct."
                    except Exception:
                        message = "Your SQL is not correct."
                else:
                    # Compat wrapper: custom test cases expect 3-value return (result, error, timeout)
                    def _execute_queries_compat(queries, db_name, conn=None):
                        result, error, timeout, _ = execute_queries(queries, db_name, conn)
                        return result, error, timeout

                    exec_globals = {
                        "execute_queries": _execute_queries_compat, "ex_base": ex_base,
                        "remove_distinct": remove_distinct, "remove_comments": remove_comments,
                        "remove_round": remove_round,
                        "pred_query_result": pred_query_result,
                    }
                    all_passed = True
                    for i, tc_code in enumerate(test_cases):
                        if not isinstance(tc_code, str): continue
                        try:
                            exec_locals = {}
                            exec(tc_code, exec_globals, exec_locals)
                            tc_func = exec_locals.get("test_case")
                            if tc_func and callable(tc_func):
                                tc_func(pred_sqls, sol_sqls, task_db, conn)
                        except AssertionError:
                            all_passed = False
                            message = "Your SQL is not correct."
                            break
                        except Exception:
                            all_passed = False
                            message = "Your SQL is not correct."
                            break
                    if all_passed:
                        passed = True
                        message = "SQL passed all test cases."

            pool.putconn(conn)

            if passed:
                if interact_mode == "c-interact":
                    reward = phase_rewards_first.get(current_phase, 0) if is_first_try else phase_rewards_debug.get(current_phase, 0)
                else:
                    reward = phase_rewards_first.get(current_phase, 0)
                if current_phase == 1:
                    _successful_phase1_sql[req_task_id] = req_sql
                    # Apply Phase 1 SQL and snapshot for Phase 2
                    reset_task_db(task_db, f"{base_db}_template")
                    p_pool = _get_or_init_pool(task_db)
                    p_conn = p_pool.getconn()
                    try:
                        execute_queries(pred_sqls, task_db, p_conn)
                    finally:
                        p_pool.putconn(p_conn)
                    close_pool(task_db)  # must close connections before using as template
                    snapshot_db = create_task_db(task_db, "p1snap", template=task_db)
                    td["_snapshot_db"] = snapshot_db

                    has_follow_up = bool(td.get("follow_up") and td["follow_up"].get("sol_sql"))
                    if has_follow_up:
                        _submit_attempts[req_task_id][2] = 0
                        td["_current_phase"] = 2
                        follow_up_query = td["follow_up"].get("query", "Please complete the follow-up task.")
                        return SubmitSQLResponse(
                            passed=True, message=f"Phase 1 correct! (Reward: {reward}). Moving to Phase 2.",
                            reward=reward, phase_completed=1, has_follow_up=True,
                            follow_up_query=follow_up_query)
                    else:
                        return SubmitSQLResponse(
                            passed=True, message=f"Phase 1 correct! (Reward: {reward}). Task finished.",
                            reward=reward, phase_completed=1, has_follow_up=False)
                else:
                    return SubmitSQLResponse(
                        passed=True, message=f"Phase 2 correct! (Reward: {reward}). Task finished.",
                        reward=reward, phase_completed=2, has_follow_up=False)
            else:
                # Failed: restore task DB to pre-submit state for agent exploration
                if current_phase == 1:
                    reset_task_db(task_db, f"{base_db}_template")
                else:
                    snapshot = td.get("_snapshot_db")
                    if snapshot:
                        reset_task_db(task_db, snapshot)

                return SubmitSQLResponse(
                    passed=False, message=f"SQL failed Phase {current_phase}. {message}",
                    reward=0.0)
        except Exception as inner_e:
            try: pool.putconn(conn)
            except: pass
            raise inner_e
    except Exception as e:
        logger.error(f"Submit error for {req_task_id}: {e}", exc_info=True)
        return SubmitSQLResponse(passed=False, message=f"Error: {e}", reward=0.0)


@app.post("/submit", response_model=SubmitSQLResponse)
async def submit_sql_endpoint(req: SubmitSQLRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    return await asyncio.to_thread(
        _submit_sql_sync, req.task_id, req.sql, td, _submit_attempts, _successful_phase1_sql
    )


@app.post("/schema")
async def get_schema(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    return {"schema": _schema_cache.get(db_name, "Schema not available")}


@app.post("/all_column_meanings")
async def get_all_column_meanings(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    return {"column_meanings": json.dumps(_column_meanings_cache.get(db_name, {}), indent=2)}


@app.post("/column_meaning")
async def get_column_meaning(req: ColumnMeaningRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    key = f"{db_name}|{req.table_name.lower()}|{req.column_name.lower()}"
    meaning = _column_meanings_cache.get(db_name, {}).get(key, "Column meaning not found")
    return {"meaning": meaning if isinstance(meaning, str) else json.dumps(meaning)}


@app.post("/knowledge_names")
async def get_knowledge_names(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    agent_kb = _filter_knowledge(db_name, td)
    return {"names": list(agent_kb.keys())}


@app.post("/knowledge")
async def get_knowledge(req: KnowledgeRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    agent_kb = _filter_knowledge(db_name, td)
    if req.knowledge_name:
        entry = agent_kb.get(req.knowledge_name)
        if entry:
            visible = {k: entry[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in entry}
            return {"knowledge": json.dumps(visible, indent=2)}
        return {"knowledge": "Knowledge not found."}
    else:
        visible_kbs = []
        for e in agent_kb.values():
            visible_kbs.append({k: e[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in e})
        return {"knowledge": json.dumps(visible_kbs, indent=2)}


def _cleanup_task_sync(task_db, snapshot_db):
    """Blocking cleanup — runs in thread pool."""
    if snapshot_db:
        drop_task_db(snapshot_db)
    if task_db:
        drop_task_db(task_db)


@app.post("/cleanup_task")
async def cleanup_task(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td:
        return {"status": "ok", "task_id": req.task_id}
    task_db = td.get("_task_db")
    snapshot_db = td.get("_snapshot_db")
    try:
        await asyncio.to_thread(_cleanup_task_sync, task_db, snapshot_db)
    except Exception as e:
        logger.warning(f"Cleanup failed for {req.task_id}: {e}")
    _task_data.pop(req.task_id, None)
    _submit_attempts.pop(req.task_id, None)
    _successful_phase1_sql.pop(req.task_id, None)
    return {"status": "ok", "task_id": req.task_id}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "db_environment"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.db_env_port)
