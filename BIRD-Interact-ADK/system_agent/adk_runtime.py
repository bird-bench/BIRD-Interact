"""Thin ADK runtime wrapper for the system-agent service.

This module keeps the ADK dependency optional at import time so the service can
still start and report a clear health signal when `google-adk` is not installed.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class _SessionRef:
    app_name: str
    user_id: str
    session_id: str


class AdkRuntime:
    """Manage ADK runners and sessions for c-interact and a-interact."""

    def __init__(self) -> None:
        self.available = False
        self.error = ""
        self._backend: Optional[Dict[str, Any]] = None
        self._runners: Dict[str, Any] = {}
        self._session_refs: Dict[tuple[str, str], _SessionRef] = {}
        self._lock = asyncio.Lock()
        self._load_backend()

    def _load_backend(self) -> None:
        if self._backend is not None or self.error:
            return
        try:
            runners_mod = importlib.import_module("google.adk.runners")
            genai_types = importlib.import_module("google.genai.types")
            agent_mod = importlib.import_module("system_agent.agent")
            build_agent = getattr(agent_mod, "build_agent")

            backend: Dict[str, Any] = {
                "types": genai_types,
                "build_agent": build_agent,
            }

            if hasattr(runners_mod, "InMemoryRunner"):
                backend["runner_cls"] = getattr(runners_mod, "InMemoryRunner")
                backend["runner_kind"] = "in_memory"
            else:
                backend["runner_cls"] = getattr(runners_mod, "Runner")
                sessions_mod = importlib.import_module("google.adk.sessions")
                backend["session_service_cls"] = getattr(sessions_mod, "InMemorySessionService")
                backend["runner_kind"] = "legacy"

            self._backend = backend
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False
            logger.warning("ADK runtime unavailable: %s", exc)

    async def _get_runner(self, mode: str) -> tuple[Any, str]:
        if mode in self._runners:
            return self._runners[mode]
        if not self.available or self._backend is None:
            raise RuntimeError(self.error or "ADK runtime unavailable")

        app_name = f"bird_interact_{mode.replace('-', '_')}"
        agent = self._backend["build_agent"](mode)

        if self._backend["runner_kind"] == "in_memory":
            runner = self._backend["runner_cls"](agent=agent, app_name=app_name)
        else:
            session_service = self._backend["session_service_cls"]()
            runner = self._backend["runner_cls"](
                agent=agent,
                app_name=app_name,
                session_service=session_service,
            )

        self._runners[mode] = (runner, app_name)
        return runner, app_name

    def _make_text_message(self, text: str) -> Any:
        if not self.available or self._backend is None:
            raise RuntimeError(self.error or "ADK runtime unavailable")

        types_mod = self._backend["types"]
        try:
            return types_mod.Content(
                role="user",
                parts=[types_mod.Part(text=text)],
            )
        except TypeError:
            return types_mod.Content(
                role="user",
                parts=[types_mod.Part.from_text(text=text)],
            )

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if content is None:
            return ""
        parts = getattr(content, "parts", None) or []
        texts = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
        return "\n".join(texts).strip()

    @staticmethod
    def _event_is_final(event: Any) -> bool:
        is_final = getattr(event, "is_final_response", None)
        if callable(is_final):
            try:
                return bool(is_final())
            except Exception:
                return False
        return False

    @staticmethod
    def _session_id(session: Any) -> str:
        return getattr(session, "id", None) or getattr(session, "session_id", "")

    @staticmethod
    def _preview(value: Any, limit: int = 4000) -> str:
        try:
            if isinstance(value, (dict, list)):
                text = json.dumps(value, ensure_ascii=False)
            else:
                text = str(value)
        except Exception:
            text = repr(value)
        if len(text) > limit:
            return text[:limit] + "...<truncated>"
        return text

    def _serialize_part(self, part: Any) -> Dict[str, Any]:
        text = getattr(part, "text", None)
        if text:
            return {"type": "text", "text": self._preview(text)}

        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            return {
                "type": "function_call",
                "name": getattr(function_call, "name", ""),
                "id": getattr(function_call, "id", ""),
                "args": getattr(function_call, "args", {}) or {},
            }

        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            return {
                "type": "function_response",
                "name": getattr(function_response, "name", ""),
                "id": getattr(function_response, "id", ""),
                "response": self._preview(getattr(function_response, "response", "")),
            }

        return {"type": "unknown", "repr": self._preview(part)}

    def _serialize_event(self, event: Any) -> Dict[str, Any]:
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) or []
        return {
            "type": "adk_event",
            "author": getattr(event, "author", ""),
            "invocation_id": getattr(event, "invocation_id", ""),
            "branch": getattr(event, "branch", ""),
            "final": self._event_is_final(event),
            "content": {
                "role": getattr(content, "role", "") if content else "",
                "parts": [self._serialize_part(part) for part in parts],
            },
        }

    async def init_session(
        self,
        task_id: str,
        mode: str,
        state: Optional[Dict[str, Any]] = None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        async with self._lock:
            runner, app_name = await self._get_runner(mode)
            key = (mode, task_id)
            if key in self._session_refs and not reset:
                ref = self._session_refs[key]
                return {
                    "task_id": task_id,
                    "mode": mode,
                    "session_id": ref.session_id,
                    "adk_available": True,
                }

            user_id = f"user_{task_id}"
            session = await runner.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                state=state or {},
            )
            session_state = getattr(session, "state", {}) or {}
            session_state.setdefault("tool_trajectory", [])
            session_state.setdefault("adk_events", [])
            session.state = session_state
            ref = _SessionRef(
                app_name=app_name,
                user_id=user_id,
                session_id=self._session_id(session),
            )
            self._session_refs[key] = ref
            return {
                "task_id": task_id,
                "mode": mode,
                "session_id": ref.session_id,
                "adk_available": True,
            }

    async def run_turn(self, task_id: str, mode: str, message: str,
                       **kwargs) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError(self.error or "ADK runtime unavailable")

        key = (mode, task_id)
        if key not in self._session_refs:
            await self.init_session(task_id=task_id, mode=mode, state={}, reset=False)

        runner, _ = await self._get_runner(mode)
        ref = self._session_refs[key]
        new_message = self._make_text_message(message)

        # Reset per-phase flags via append_event (the ADK-native way)
        if mode == "c-interact":
            from google.adk.events import Event, EventActions
            session = await runner.session_service.get_session(
                app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
            )
            if session:
                reset_event = Event(
                    author="system",
                    actions=EventActions(state_delta={"_submitted_this_phase": False}),
                )
                await runner.session_service.append_event(session, reset_event)

        final_text = ""
        turn_events = [{
            "type": "user_message",
            "message": self._preview(message, limit=1200),
        }]
        async for event in runner.run_async(
            user_id=ref.user_id,
            session_id=ref.session_id,
            new_message=new_message,
        ):
            turn_events.append(self._serialize_event(event))
            content = getattr(event, "content", None)
            text = self._extract_text_from_content(content)
            if self._event_is_final(event) and text:
                final_text = text
            elif text:
                final_text = text

        session = await runner.session_service.get_session(
            app_name=ref.app_name,
            user_id=ref.user_id,
            session_id=ref.session_id,
        )
        session_state = getattr(session, "state", {}) or {}
        adk_events = session_state.get("adk_events", [])
        adk_events.extend(turn_events)
        session_state["adk_events"] = adk_events
        session.state = session_state
        return {
            "task_id": task_id,
            "mode": mode,
            "session_id": ref.session_id,
            "response": final_text,
            "state": session_state,
            "adk_available": True,
        }
