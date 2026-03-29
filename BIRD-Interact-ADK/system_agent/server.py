"""System Agent Service (Port 6000).

Wraps the LLM agent behind a FastAPI endpoint so the orchestrator
talks to ALL three components through HTTP ports:
  - System Agent:    port 6000  (this service)
  - User Simulator:  port 6001
  - DB Environment:  port 6002

Supports two modes:
  - /chat: Simple prompt-in, text-out (used by c-interact)
  - /init_session + /run_session: ADK-backed session runtime (used by a-interact)
  - /chat_with_tools: legacy non-ADK fallback for old orchestrator clients
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared.config import settings
from system_agent.adk_runtime import AdkRuntime

logger = logging.getLogger(__name__)
app = FastAPI(title="BIRD-Interact System Agent", version="1.0.0")
runtime = AdkRuntime()


# ── Request / Response models ──────────────────────────────────────────────

class SessionInitRequest(BaseModel):
    task_id: str
    mode: str = "a-interact"
    state: Dict[str, Any] = {}
    reset: bool = True


class SessionRunRequest(BaseModel):
    task_id: str
    message: str
    mode: str = "a-interact"


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/init_session")
async def init_session(req: SessionInitRequest):
    """Initialize an ADK runner session for a task."""
    if not runtime.available:
        raise HTTPException(status_code=503, detail=f"ADK runtime unavailable: {runtime.error}")
    return await runtime.init_session(
        task_id=req.task_id,
        mode=req.mode,
        state=req.state,
        reset=req.reset,
    )


@app.post("/run_session")
async def run_session(req: SessionRunRequest):
    """Run one ADK turn on an existing task session."""
    if not runtime.available:
        raise HTTPException(status_code=503, detail=f"ADK runtime unavailable: {runtime.error}")
    return await runtime.run_turn(
        task_id=req.task_id,
        mode=req.mode,
        message=req.message,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "system_agent",
        "model": settings.system_agent_model,
        "adk_available": runtime.available,
        "adk_error": runtime.error,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.system_agent_port)
