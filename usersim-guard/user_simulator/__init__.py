"""
User Simulator for Text-to-SQL Interaction Tasks.

A library for generating user simulator prompts and evaluating responses
in Text-to-SQL interaction scenarios.

Example Usage:
    >>> from user_simulator import UserSimulatorPipeline, generate_prompts
    >>>
    >>> # Quick usage with high-level function
    >>> generate_prompts(
    ...     data_path="data.jsonl",
    ...     output_path="output.jsonl",
    ...     prompt_type="step1"
    ... )
    >>>
    >>> # Or use the pipeline class for more control
    >>> pipeline = UserSimulatorPipeline(prompt_type="step1")
    >>> results = pipeline.process_dataset("data.jsonl")
"""

from .sql_parser import segment_sql, format_sql_segments, format_multiple_sqls
from .prompts import (
    USER_SIMULATOR_BASE,
    USER_SIMULATOR_STEP1,
    USER_SIMULATOR_STEP2,
    TEMPLATES,
    get_template,
)
from .prompt_generator import (
    generate_base_prompt,
    generate_step1_prompt,
    generate_step2_prompt,
    extract_action_from_response,
    format_ambiguities,
    load_db_schema,
    MissingFieldError,
    PromptGenerationError,
)
from .pipeline import (
    UserSimulatorPipeline,
    PipelineConfig,
    create_pipeline,
    generate_prompts,
)
from .data_loader import (
    DataLoader,
    create_data_loader,
    load_jsonl,
    save_jsonl,
    load_source_data,
    merge_with_source,
    normalize_instance_id,
)
from .llm_as_judge import (
    generate_judge_prompt,
    generate_judge_prompts_from_files,
    extract_rating,
    parse_judge_results,
    JudgeEvaluationError,
)

__version__ = "1.0.0"
__author__ = "User Simulator Team"

__all__ = [
    # SQL Parser
    "segment_sql",
    "format_sql_segments",
    "format_multiple_sqls",
    # Prompts
    "USER_SIMULATOR_BASE",
    "USER_SIMULATOR_STEP1",
    "USER_SIMULATOR_STEP2",
    "TEMPLATES",
    "get_template",
    # Prompt Generator
    "generate_base_prompt",
    "generate_step1_prompt",
    "generate_step2_prompt",
    "extract_action_from_response",
    "format_ambiguities",
    "load_db_schema",
    # Exceptions
    "MissingFieldError",
    "PromptGenerationError",
    "JudgeEvaluationError",
    # Pipeline
    "UserSimulatorPipeline",
    "PipelineConfig",
    "create_pipeline",
    "generate_prompts",
    # Data Loader
    "DataLoader",
    "create_data_loader",
    "load_jsonl",
    "save_jsonl",
    "load_source_data",
    "merge_with_source",
    "normalize_instance_id",
    # LLM-as-Judge
    "generate_judge_prompt",
    "generate_judge_prompts_from_files",
    "extract_rating",
    "parse_judge_results",
]
