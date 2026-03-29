"""Pydantic models for inter-service communication."""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class InitTaskRequest(BaseModel):
    task_id: str
    task_data: Dict[str, Any]


class ExecuteSQLRequest(BaseModel):
    sql: str
    task_id: str


class ExecuteSQLResponse(BaseModel):
    result: str
    success: bool
    error: Optional[str] = None


class SubmitSQLRequest(BaseModel):
    sql: str
    task_id: str


class SubmitSQLResponse(BaseModel):
    passed: bool
    message: str
    reward: float = 0.0
    phase_completed: Optional[int] = None
    has_follow_up: bool = False
    follow_up_query: Optional[str] = None


class SchemaRequest(BaseModel):
    task_id: str


class ColumnMeaningRequest(BaseModel):
    task_id: str
    table_name: str
    column_name: str


class KnowledgeRequest(BaseModel):
    task_id: str
    knowledge_name: Optional[str] = None


class AskUserRequest(BaseModel):
    question: str
    task_id: str


class AskUserResponse(BaseModel):
    answer: str


class PhaseTransitionRequest(BaseModel):
    task_id: str
