"""
Prompt Generator - Core logic for generating user simulator prompts.

This module provides functions to generate prompts for different versions
of the user simulator (base, step1, step2).

IMPORTANT: This module follows strict error handling - if any required field
is missing, it will raise an error instead of using fallback values.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .sql_parser import format_multiple_sqls
from .prompts import get_template


# =============================================================================
# EXCEPTIONS
# =============================================================================

class MissingFieldError(KeyError):
    """Raised when a required field is missing from data."""
    pass


class PromptGenerationError(ValueError):
    """Raised when prompt generation fails."""
    pass


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _require_field(data: Dict[str, Any], field: str, context: str = "") -> Any:
    """
    Get a required field from data, raising error if missing.

    Args:
        data: Dictionary to get field from.
        field: Field name to retrieve.
        context: Additional context for error message.

    Returns:
        The field value.

    Raises:
        MissingFieldError: If field is missing or None.
    """
    if field not in data:
        raise MissingFieldError(
            f"Required field '{field}' is missing from data. {context}"
        )
    value = data[field]
    if value is None:
        raise MissingFieldError(
            f"Required field '{field}' is None. {context}"
        )
    return value


def _require_non_empty(value: Any, field_name: str, context: str = "") -> Any:
    """
    Validate that a value is non-empty.

    Args:
        value: Value to validate.
        field_name: Name of the field for error message.
        context: Additional context for error message.

    Returns:
        The value if non-empty.

    Raises:
        PromptGenerationError: If value is empty.
    """
    if not value:
        raise PromptGenerationError(
            f"Field '{field_name}' cannot be empty. {context}"
        )
    return value


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_ambiguities(
    user_query_ambiguity: Dict[str, Any],
    knowledge_ambiguity: List[Dict[str, Any]]
) -> str:
    """
    Format ambiguity information as a JSON string.

    Args:
        user_query_ambiguity: Dictionary containing user query ambiguity info.
        knowledge_ambiguity: List of knowledge ambiguity dictionaries.

    Returns:
        Formatted JSON string with both ambiguity types.

    Raises:
        PromptGenerationError: If ambiguity data is invalid.
    """
    if not isinstance(user_query_ambiguity, dict):
        raise PromptGenerationError(
            f"user_query_ambiguity must be a dict, got {type(user_query_ambiguity)}"
        )
    if not isinstance(knowledge_ambiguity, list):
        raise PromptGenerationError(
            f"knowledge_ambiguity must be a list, got {type(knowledge_ambiguity)}"
        )

    return (
        "user_query_ambiguity: \n"
        + json.dumps(user_query_ambiguity, indent=4)
        + "\n\nknowledge_ambiguity: \n"
        + json.dumps(knowledge_ambiguity, indent=4)
    )


def load_db_schema(db_name: str, db_base_path: Union[str, Path]) -> str:
    """
    Load database schema from file.

    Args:
        db_name: Name of the database.
        db_base_path: Base path to the DBs directory.

    Returns:
        Database schema as string.

    Raises:
        FileNotFoundError: If schema file doesn't exist.
        PromptGenerationError: If db_name is empty.
    """
    if not db_name:
        raise PromptGenerationError("db_name cannot be empty")
    if not db_base_path:
        raise PromptGenerationError("db_base_path cannot be empty")

    db_base_path = Path(db_base_path)
    schema_path = db_base_path / db_name / f"{db_name}_schema.txt"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    content = schema_path.read_text(encoding="utf-8")
    if not content.strip():
        raise PromptGenerationError(f"Schema file is empty: {schema_path}")

    return content


def extract_action_from_response(response: str) -> str:
    """
    Extract action from step1 response.

    The response format is expected to have action between <s> and </s> tags.

    Args:
        response: Raw response string from step1.

    Returns:
        Extracted action string.

    Raises:
        PromptGenerationError: If response is empty or action cannot be extracted.
    """
    if not response:
        raise PromptGenerationError("Response is empty, cannot extract action")

    # Find content between </s> tags (from the end)
    cut_idx = response.find("</s>")
    if cut_idx != -1:
        extracted = response[:cut_idx].strip()
    else:
        extracted = response

    # Find content after <s> tag
    if "<s>" in extracted:
        cut_idx = extracted.find("<s>")
        extracted = extracted[cut_idx:].replace("<s>", "").strip()

    if not extracted:
        raise PromptGenerationError(
            f"Could not extract action from response: {response[:200]}..."
        )

    return extracted


# =============================================================================
# PROMPT GENERATORS
# =============================================================================

def generate_base_prompt(
    data: Dict[str, Any],
    clarification_question: str,
    db_schema: str,
) -> str:
    """
    Generate prompt for base version user simulator.

    Args:
        data: Data dictionary containing:
            - selected_database: Database name (REQUIRED)
            - amb_user_query: Ambiguous user query (REQUIRED)
            - user_query_ambiguity: User query ambiguity info (REQUIRED)
            - knowledge_ambiguity: Knowledge ambiguity info (REQUIRED)
            - sol_sql: List of solution SQL statements (REQUIRED)
        clarification_question: The clarification question from AI (REQUIRED).
        db_schema: Database schema string (REQUIRED).

    Returns:
        Generated prompt string.

    Raises:
        MissingFieldError: If any required field is missing.
        PromptGenerationError: If prompt generation fails.
    """
    # Validate inputs
    _require_non_empty(clarification_question, "clarification_question")
    _require_non_empty(db_schema, "db_schema")

    # Get required fields
    db_name = _require_field(data, "selected_database", "Context: generate_base_prompt")
    user_query = _require_field(data, "amb_user_query", "Context: generate_base_prompt")
    user_query_ambiguity = _require_field(data, "user_query_ambiguity", "Context: generate_base_prompt")
    knowledge_ambiguity = _require_field(data, "knowledge_ambiguity", "Context: generate_base_prompt")
    sol_sql = _require_field(data, "sol_sql", "Context: generate_base_prompt")

    template = get_template("base")

    # Format ambiguities
    ambiguities_json = format_ambiguities(user_query_ambiguity, knowledge_ambiguity)

    # Combine SQL statements
    if isinstance(sol_sql, str):
        sol_sql = [sol_sql]
    _require_non_empty(sol_sql, "sol_sql", "SQL list cannot be empty")
    correct_sql = "\n\n".join(sol_sql).strip()

    return template.format(
        db_name=db_name,
        db_schema=db_schema,
        user_query=user_query,
        ambiguities_json=ambiguities_json,
        correct_sql=correct_sql,
        clarification_question=clarification_question,
    )


def generate_step1_prompt(
    data: Dict[str, Any],
    clarification_question: str,
) -> str:
    """
    Generate prompt for step1 (proposed version step 1 - action classification).

    Args:
        data: Data dictionary containing:
            - user_query_ambiguity: User query ambiguity info (REQUIRED)
            - knowledge_ambiguity: Knowledge ambiguity info (REQUIRED)
            - sol_sql: List of solution SQL statements (REQUIRED)
        clarification_question: The clarification question from AI (REQUIRED).

    Returns:
        Generated prompt string.

    Raises:
        MissingFieldError: If any required field is missing.
        PromptGenerationError: If prompt generation fails.
    """
    # Validate inputs
    _require_non_empty(clarification_question, "clarification_question")

    # Get required fields
    user_query_ambiguity = _require_field(data, "user_query_ambiguity", "Context: generate_step1_prompt")
    knowledge_ambiguity = _require_field(data, "knowledge_ambiguity", "Context: generate_step1_prompt")
    sol_sql = _require_field(data, "sol_sql", "Context: generate_step1_prompt")

    template = get_template("step1")

    # Format ambiguities
    ambiguities_json = format_ambiguities(user_query_ambiguity, knowledge_ambiguity)

    # Format SQL segments
    if isinstance(sol_sql, str):
        sol_sql = [sol_sql]
    _require_non_empty(sol_sql, "sol_sql", "SQL list cannot be empty")
    sql_segments = format_multiple_sqls(sol_sql)

    return template.format(
        ambiguities_json=ambiguities_json,
        sql_segments=sql_segments,
        clarification_question=clarification_question,
    )


def generate_step2_prompt(
    data: Dict[str, Any],
    clarification_question: str,
    action: str,
    db_schema: str,
) -> str:
    """
    Generate prompt for step2 (proposed version step 2 - response generation).

    Args:
        data: Data dictionary containing:
            - amb_user_query: Ambiguous user query (REQUIRED)
            - user_query_ambiguity: User query ambiguity info (REQUIRED)
            - knowledge_ambiguity: Knowledge ambiguity info (REQUIRED)
            - sol_sql: List of solution SQL statements (REQUIRED)
        clarification_question: The clarification question from AI (REQUIRED).
        action: The action chosen in step1 (REQUIRED).
        db_schema: Database schema string (REQUIRED).

    Returns:
        Generated prompt string.

    Raises:
        MissingFieldError: If any required field is missing.
        PromptGenerationError: If prompt generation fails.
    """
    # Validate inputs
    _require_non_empty(clarification_question, "clarification_question")
    _require_non_empty(action, "action")
    _require_non_empty(db_schema, "db_schema")

    # Get required fields
    user_query = _require_field(data, "amb_user_query", "Context: generate_step2_prompt")
    user_query_ambiguity = _require_field(data, "user_query_ambiguity", "Context: generate_step2_prompt")
    knowledge_ambiguity = _require_field(data, "knowledge_ambiguity", "Context: generate_step2_prompt")
    sol_sql = _require_field(data, "sol_sql", "Context: generate_step2_prompt")

    template = get_template("step2")

    # Format ambiguities
    ambiguities_json = format_ambiguities(user_query_ambiguity, knowledge_ambiguity)

    # Format SQL
    if isinstance(sol_sql, str):
        sol_sql = [sol_sql]
    _require_non_empty(sol_sql, "sol_sql", "SQL list cannot be empty")
    correct_sql = "\n\n".join(sol_sql).strip()
    sql_segments = format_multiple_sqls(sol_sql)

    return template.format(
        db_schema=db_schema,
        ambiguities_json=ambiguities_json,
        user_query=user_query,
        correct_sql=correct_sql,
        sql_segments=sql_segments,
        clarification_question=clarification_question,
        action=action,
    )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def generate_prompts_batch(
    data_list: List[Dict[str, Any]],
    questions_list: List[Dict[str, Any]],
    prompt_type: str,
    db_base_path: Optional[Union[str, Path]] = None,
    step1_responses: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate prompts for a batch of data.

    Args:
        data_list: List of data dictionaries.
        questions_list: List of question dictionaries containing clarification questions.
        prompt_type: One of "base", "step1", or "step2".
        db_base_path: Base path to DBs directory (required for "base" and "step2").
        step1_responses: List of step1 responses (required for "step2").

    Returns:
        List of dictionaries with generated prompts.

    Raises:
        ValueError: If required arguments are missing.
        MissingFieldError: If required fields are missing from data.
        PromptGenerationError: If prompt generation fails.
    """
    if prompt_type in ["base", "step2"] and db_base_path is None:
        raise ValueError(f"db_base_path is required for prompt_type '{prompt_type}'")

    if prompt_type == "step2" and step1_responses is None:
        raise ValueError("step1_responses is required for prompt_type 'step2'")

    results = []
    schema_cache = {}

    for idx, (data, questions) in enumerate(zip(data_list, questions_list)):
        db_name = _require_field(data, "selected_database", f"Context: batch item {idx}")

        # Load schema if needed (with caching)
        db_schema = None
        if prompt_type in ["base", "step2"]:
            if db_name not in schema_cache:
                schema_cache[db_name] = load_db_schema(db_name, db_base_path)
            db_schema = schema_cache[db_name]

        # Process each question type
        for question_key, question_data in questions.items():
            if not question_key.startswith("question_"):
                continue

            for q_type in ["normal", "verbose", "concise"]:
                clarification_q = question_data.get(q_type)
                if not clarification_q:
                    continue

                # Generate prompt based on type
                if prompt_type == "base":
                    prompt = generate_base_prompt(data, clarification_q, db_schema)
                elif prompt_type == "step1":
                    prompt = generate_step1_prompt(data, clarification_q)
                elif prompt_type == "step2":
                    if idx >= len(step1_responses):
                        raise PromptGenerationError(
                            f"No step1 response for index {idx}"
                        )
                    step1_resp = step1_responses[idx]
                    response = _require_field(step1_resp, "response", f"Context: step1 response {idx}")
                    action = extract_action_from_response(response)
                    prompt = generate_step2_prompt(
                        data, clarification_q, action, db_schema
                    )
                else:
                    raise ValueError(f"Unknown prompt_type: {prompt_type}")

                result = {
                    "instance_id": _require_field(data, "instance_id", f"Context: batch item {idx}"),
                    "selected_database": db_name,
                    "question_id": question_key,
                    "question_type": q_type,
                    "clarification_question": clarification_q,
                    "prompt": prompt,
                }

                # Copy additional fields from original data
                for key in ["user_query_ambiguity", "knowledge_ambiguity"]:
                    if key in data:
                        result[key] = data[key]

                results.append(result)

    return results
