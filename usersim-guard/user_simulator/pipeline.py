"""
Pipeline - HuggingFace-style unified pipeline for user simulator prompt generation.

This module provides a clean, easy-to-use interface for generating prompts,
following HuggingFace's design patterns.

Example Usage:
    >>> from user_simulator import UserSimulatorPipeline
    >>>
    >>> # Initialize pipeline
    >>> pipeline = UserSimulatorPipeline(
    ...     db_base_path="/path/to/DBs",
    ...     prompt_type="base"
    ... )
    >>>
    >>> # Process single item
    >>> result = pipeline(data_item, question_item)
    >>>
    >>> # Process dataset
    >>> results = pipeline.process_dataset(
    ...     data_path="data.jsonl",
    ...     questions_path="questions.jsonl",
    ...     output_path="output.jsonl"
    ... )
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .sql_parser import format_multiple_sqls
from .prompts import get_template
from .prompt_generator import (
    generate_base_prompt,
    generate_step1_prompt,
    generate_step2_prompt,
    extract_action_from_response,
    load_db_schema,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for UserSimulatorPipeline."""

    prompt_type: str = "base"
    """Type of prompt to generate: "base", "step1", or "step2"."""

    db_base_path: Optional[str] = None
    """Base path to DBs directory. Required for "base" and "step2" types."""

    question_types: List[str] = field(default_factory=lambda: ["normal", "verbose", "concise"])
    """List of question types to process."""

    cache_schemas: bool = True
    """Whether to cache loaded database schemas."""

    def __post_init__(self):
        if self.prompt_type in ["base", "step2"] and not self.db_base_path:
            raise ValueError(
                f"db_base_path is required for prompt_type '{self.prompt_type}'"
            )


# =============================================================================
# PIPELINE CLASS
# =============================================================================

class UserSimulatorPipeline:
    """
    HuggingFace-style pipeline for generating user simulator prompts.

    This pipeline supports three modes:
    - "base": Generate prompts using the base template
    - "step1": Generate prompts for action classification (step 1)
    - "step2": Generate prompts for response generation (step 2)

    Example:
        >>> pipeline = UserSimulatorPipeline(
        ...     prompt_type="step1"
        ... )
        >>> result = pipeline(data, questions)
    """

    SUPPORTED_TYPES = ["base", "step1", "step2"]

    def __init__(
        self,
        prompt_type: str = "base",
        db_base_path: Optional[Union[str, Path]] = None,
        question_types: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the pipeline.

        Args:
            prompt_type: Type of prompt to generate ("base", "step1", "step2").
            db_base_path: Base path to DBs directory.
            question_types: List of question types to process.
            **kwargs: Additional configuration options.
        """
        if prompt_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported prompt_type: {prompt_type}. "
                f"Supported types: {self.SUPPORTED_TYPES}"
            )

        self.prompt_type = prompt_type
        self.db_base_path = Path(db_base_path) if db_base_path else None
        self.question_types = question_types or ["normal", "verbose", "concise"]
        self._schema_cache: Dict[str, str] = {}

        if prompt_type in ["base", "step2"] and not db_base_path:
            raise ValueError(
                f"db_base_path is required for prompt_type '{prompt_type}'"
            )

    def _get_schema(self, db_name: str) -> str:
        """Load database schema with caching."""
        if db_name not in self._schema_cache:
            self._schema_cache[db_name] = load_db_schema(db_name, self.db_base_path)
        return self._schema_cache[db_name]

    def _generate_single(
        self,
        data: Dict[str, Any],
        clarification_question: str,
        question_id: str,
        question_type: str,
        action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a single prompt."""
        db_name = data.get("selected_database", "")

        if self.prompt_type == "base":
            db_schema = self._get_schema(db_name)
            prompt = generate_base_prompt(data, clarification_question, db_schema)

        elif self.prompt_type == "step1":
            prompt = generate_step1_prompt(data, clarification_question)

        elif self.prompt_type == "step2":
            if action is None:
                raise ValueError("action is required for step2 prompt type")
            db_schema = self._get_schema(db_name)
            prompt = generate_step2_prompt(
                data, clarification_question, action, db_schema
            )

        return {
            "instance_id": data.get("instance_id", ""),
            "selected_database": db_name,
            "question_id": question_id,
            "question_type": question_type,
            "clarification_question": clarification_question,
            "prompt": prompt,
        }

    def __call__(
        self,
        data: Dict[str, Any],
        questions: Optional[Dict[str, Any]] = None,
        clarification_question: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate prompt(s) for a single data item.

        Args:
            data: Data dictionary with instance information.
            questions: Dictionary of questions (with question_* keys).
            clarification_question: Direct clarification question string.
            action: Action string (required for step2 type).
            **kwargs: Additional arguments.

        Returns:
            Single result dict if clarification_question is provided,
            or list of result dicts if questions dict is provided.
        """
        if clarification_question is not None:
            return self._generate_single(
                data=data,
                clarification_question=clarification_question,
                question_id=kwargs.get("question_id", "direct"),
                question_type=kwargs.get("question_type", "normal"),
                action=action,
            )

        if questions is None:
            raise ValueError(
                "Either 'questions' dict or 'clarification_question' string must be provided"
            )

        results = []
        for question_key, question_data in questions.items():
            if not question_key.startswith("question_"):
                continue

            for q_type in self.question_types:
                clarification_q = question_data.get(q_type, "")
                if not clarification_q:
                    continue

                result = self._generate_single(
                    data=data,
                    clarification_question=clarification_q,
                    question_id=question_key,
                    question_type=q_type,
                    action=action,
                )
                results.append(result)

        return results

    def process_dataset(
        self,
        data_path: Union[str, Path],
        questions_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        step1_responses_path: Optional[Union[str, Path]] = None,
        return_results: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Process an entire dataset and optionally save results.

        Args:
            data_path: Path to input JSONL file with data.
            questions_path: Path to JSONL file with questions.
            output_path: Path to save output JSONL file.
            step1_responses_path: Path to step1 responses (for step2 type).
            return_results: Whether to return results list.

        Returns:
            List of result dictionaries if return_results is True.
        """
        data_path = Path(data_path)
        results = []

        with open(data_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f]

        questions_list = None
        if questions_path:
            with open(questions_path, "r", encoding="utf-8") as f:
                questions_list = [json.loads(line) for line in f]

        step1_responses = None
        if self.prompt_type == "step2" and step1_responses_path:
            with open(step1_responses_path, "r", encoding="utf-8") as f:
                step1_responses = [json.loads(line) for line in f]

        for idx, data in enumerate(data_list):
            if questions_list and idx < len(questions_list):
                questions = questions_list[idx]
            else:
                questions = None
                clarification_q = data.get("clarification_question")

            action = None
            if self.prompt_type == "step2" and step1_responses:
                if idx < len(step1_responses):
                    action = extract_action_from_response(
                        step1_responses[idx].get("response", "")
                    )

            if questions:
                item_results = self(data, questions=questions, action=action)
            elif clarification_q:
                item_results = [self(
                    data,
                    clarification_question=clarification_q,
                    question_id=data.get("question_id", ""),
                    question_type=data.get("question_type", "normal"),
                    action=action,
                )]
            else:
                continue

            results.extend(item_results if isinstance(item_results, list) else [item_results])

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return results if return_results else None

    def __repr__(self) -> str:
        return (
            f"UserSimulatorPipeline("
            f"prompt_type='{self.prompt_type}', "
            f"question_types={self.question_types})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(
    prompt_type: str,
    db_base_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> UserSimulatorPipeline:
    """
    Factory function to create a pipeline.

    Args:
        prompt_type: Type of prompt ("base", "step1", "step2").
        db_base_path: Base path to DBs directory.
        **kwargs: Additional pipeline configuration.

    Returns:
        Configured UserSimulatorPipeline instance.
    """
    return UserSimulatorPipeline(
        prompt_type=prompt_type,
        db_base_path=db_base_path,
        **kwargs,
    )


def generate_prompts(
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    prompt_type: str,
    questions_path: Optional[Union[str, Path]] = None,
    db_base_path: Optional[Union[str, Path]] = None,
    step1_responses_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    High-level function to generate prompts from a dataset.

    Args:
        data_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        prompt_type: Type of prompt ("base", "step1", "step2").
        questions_path: Path to questions JSONL file.
        db_base_path: Base path to DBs directory.
        step1_responses_path: Path to step1 responses (for step2 type).
        **kwargs: Additional pipeline configuration.

    Returns:
        List of generated prompt dictionaries.
    """
    pipeline = create_pipeline(
        prompt_type=prompt_type,
        db_base_path=db_base_path,
        **kwargs,
    )

    return pipeline.process_dataset(
        data_path=data_path,
        questions_path=questions_path,
        output_path=output_path,
        step1_responses_path=step1_responses_path,
        return_results=True,
    )
