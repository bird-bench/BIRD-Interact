"""
LLM-as-Judge Evaluation Module

Evaluates user simulator responses by checking if the simulated user
handled AI clarification questions appropriately.

Rating scale (labeled/unlabeled):
- A. perfect: User handled everything correctly
- B. normal: User's response is acceptable but not ideal
- C. bad: User acted incorrectly

Rating scale (una - unanswerable):
- A. good: User correctly refused to answer
- B. bad: User incorrectly answered (should have refused)
"""

import json
import re
from typing import Dict, Any, List


class JudgeEvaluationError(ValueError):
    """Raised when judge evaluation encounters an error."""
    pass


class MissingFieldError(KeyError):
    """Raised when a required field is missing from data."""
    pass


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# Prompt template for labeled/unlabeled data (3-class: A/B/C)
JUDGE_PROMPT_TEMPLATE = """<|The Start of Task Description|>
You are reviewing an AI–User conversation about a Text-to-SQL task.
- The AI may ask clarification questions to resolve ambiguity — this is allowed.
- However, the AI might also attempt to directly obtain the ground-truth SQL, database schema, or a step-by-step solution — these are not allowed, and the user should refuse to answer such requests. For example, the questions like "Which table" or "Which column" are usually unanswerable.

Your job is to check whether the user's response is appropriate:
1. First, analyze the conversation to determine whether the user answered properly:  (1). If the question is answerable — check that the user answered correctly **without providing ground-truth SQL, schema details, or solution steps**; and  (2). If the question is unanswerable — the user should **refuse to answer**. Then explain your reasoning inside `<t>...</t>`.
2. Then, give the user a rating from:
   - **A. perfect** → if the user handled everything correctly,
   - **B. normal** → if the user's response is acceptable but not ideal,
   - **C. bad** → if the user acted incorrectly.

Only provide the rating letter enclosed in `<s>...</s>`.

<|The End of Task Description|>

<|The Start of the Ask for Clarification Question from AI|>
{clarification_question}
<|The End of the Ask for Clarification Question from AI|>

<|The Start of the Answer from User|>
{user_answer}
<|The End of the Answer from User|>

**Expected Output Format:**
`<t>[YOUR-THOUGHT-BEFORE-YOUR-CHOICE]</t>; <s>[YOUR-CHOICE-ABOUT-USER-ANSWER-QUALITY]</s>`

**You Generation:**
- You: <t>"""

# Prompt template for UNA (unanswerable) data (2-class: A/B)
JUDGE_PROMPT_TEMPLATE_UNA = """<|The Start of Task Description|>
You are reviewing an AI–User conversation about a Text-to-SQL task.

**IMPORTANT**: The question from AI is UNANSWERABLE. The user should NOT know and CANNOT answer this question because it asks about information the user doesn't have access to (such as database schema, table names, column names, SQL implementation details, or solution steps).

Your job is to check whether the user correctly **refused to answer** this unanswerable question:
1. First, analyze whether the user refused to answer or tried to answer the question. Explain your reasoning inside `<t>...</t>`.
2. Then, give the user a rating:
   - **A. good** → if the user correctly **refused to answer** (e.g., using "I don't know", "It's out of scope", "I can not answer your question" etc.).
   - **B. bad** → if the user incorrectly **tried to directly answer** the question.

Only provide the rating letter enclosed in `<s>...</s>`.

<|The End of Task Description|>

<|The Start of the Unanswerable Question from AI|>
{clarification_question}
<|The End of the Unanswerable Question from AI|>

<|The Start of the Answer from User|>
{user_answer}
<|The End of the Answer from User|>

**Expected Output Format:**
`<t>[YOUR-THOUGHT-BEFORE-YOUR-CHOICE]</t>; <s>[YOUR-CHOICE]</s>`

**You Generation:**
- You: <t>"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _require_field(data: Dict[str, Any], field: str, context: str = "") -> Any:
    """
    Require a field to exist and not be None.

    Args:
        data: Dictionary to check
        field: Field name to require
        context: Additional context for error message

    Returns:
        The field value

    Raises:
        MissingFieldError: If field is missing or None
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


def extract_answer_text(text: str) -> str:
    """
    Extract content wrapped in <s>...</s> tags.

    - If <s> exists, extract content between <s> and </s>
    - If <s> is missing but </s> exists, extract from beginning to </s>
    - If </s> doesn't exist, return original text

    Args:
        text: Response text containing <s>...</s> tags

    Returns:
        Extracted answer text
    """
    start_tag = "<s>"
    end_tag = "</s>"

    end_idx = text.find(end_tag)
    if end_idx == -1:
        return text  # No </s>, return original text

    start_idx = text.find(start_tag)
    if start_idx != -1 and start_idx < end_idx:
        return text[start_idx + len(start_tag):end_idx]
    else:
        return text[:end_idx]


def extract_question_from_prompt(text: str) -> str:
    """
    Extract clarification question from prompt.

    Supports two formats:
    1. Base format: 'AI Asks for Clarification:xxx\nYour answer to AI: <s>'
    2. Step2 format: '<|The Start of Question from AI Collaborator|>\nxxx\n<|The End of Question from AI Collaborator|>'

    Args:
        text: Prompt text containing the clarification question

    Returns:
        Extracted clarification question

    Raises:
        JudgeEvaluationError: If question cannot be extracted
    """
    # Try base format first
    base_prefix = "AI Asks for Clarification:"
    base_suffix = "\nYour answer to AI: <s>"

    start_idx = text.find(base_prefix)
    end_idx = text.find(base_suffix)

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        return text[start_idx + len(base_prefix):end_idx].strip()

    # Try step2 format
    step2_prefix = "<|The Start of Question from AI Collaborator|>"
    step2_suffix = "<|The End of Question from AI Collaborator|>"

    start_idx = text.find(step2_prefix)
    end_idx = text.find(step2_suffix)

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        return text[start_idx + len(step2_prefix):end_idx].strip()

    raise JudgeEvaluationError(
        f"Cannot extract clarification question from prompt. "
        f"Expected base format with '{base_prefix}' or step2 format with '{step2_prefix}'"
    )


def extract_rating(judge_response: str, data_type: str = "labeled") -> str:
    """
    Extract rating from judge response.

    Args:
        judge_response: Response from judge LLM containing <s>...</s>
        data_type: Type of data ("labeled", "unlabeled", or "una")

    Returns:
        Rating letter (A, B, C for labeled/unlabeled; A, B for una)

    Raises:
        JudgeEvaluationError: If rating cannot be extracted
    """
    answer = extract_answer_text(judge_response)

    if data_type == "una":
        # UNA only has A or B
        match = re.search(r'[AB]', answer.upper())
    else:
        # Labeled/unlabeled has A, B, or C
        match = re.search(r'[ABC]', answer.upper())

    if match:
        return match.group(0)

    raise JudgeEvaluationError(
        f"Cannot extract rating from judge response: {answer}"
    )


def generate_judge_prompt(
    clarification_question: str,
    user_answer: str,
    data_type: str = "labeled"
) -> str:
    """
    Generate a judge prompt for evaluating a user's response.

    Args:
        clarification_question: The AI's clarification question
        user_answer: The simulated user's answer
        data_type: Type of data ("labeled", "unlabeled", or "una")

    Returns:
        Formatted judge prompt
    """
    if data_type == "una":
        template = JUDGE_PROMPT_TEMPLATE_UNA
    else:
        template = JUDGE_PROMPT_TEMPLATE

    return template.format(
        clarification_question=clarification_question,
        user_answer=user_answer
    )


def generate_judge_prompts_from_files(
    prompt_file: str,
    response_file: str,
    output_file: str,
    data_type: str = "labeled"
) -> List[Dict[str, Any]]:
    """
    Generate judge prompts from prompt and response files.

    Args:
        prompt_file: Path to JSONL file containing original prompts
        response_file: Path to JSONL file containing LLM responses
        output_file: Path to output JSONL file for judge prompts
        data_type: Type of data ("labeled", "unlabeled", or "una")

    Returns:
        List of judge prompt data dictionaries

    Raises:
        MissingFieldError: If required fields are missing
        FileNotFoundError: If input files don't exist
    """
    # Load prompts indexed by composite key
    prompt_dict = {}
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            instance_id = _require_field(data, "instance_id", "in prompt file")
            question_id = _require_field(data, "question_id", "in prompt file")
            question_type = _require_field(data, "question_type", "in prompt file")

            key = f"{instance_id}{question_id}{question_type}"
            prompt_dict[key] = data

    # Load responses
    with open(response_file, 'r', encoding='utf-8') as f:
        responses = [json.loads(line) for line in f]

    # Generate judge prompts
    output = []
    for data in responses:
        instance_id = _require_field(data, "instance_id", "in response file")
        question_id = _require_field(data, "question_id", "in response file")
        question_type = _require_field(data, "question_type", "in response file")
        response = _require_field(data, "response", "in response file")

        key = f"{instance_id}{question_id}{question_type}"

        if key not in prompt_dict:
            raise MissingFieldError(
                f"No matching prompt found for key: {key}"
            )

        prompt_data = prompt_dict[key]
        original_prompt = _require_field(prompt_data, "prompt", "in prompt data")

        # Extract question and answer
        clarification_question = extract_question_from_prompt(original_prompt)
        user_answer = extract_answer_text(response)

        # Generate judge prompt
        judge_prompt = generate_judge_prompt(
            clarification_question, user_answer, data_type
        )

        out_data = {
            "instance_id": instance_id,
            "question_id": question_id,
            "question_type": question_type,
            "clarification_question": clarification_question,
            "user_answer": user_answer,
            "prompt": judge_prompt,
            "data_type": data_type
        }
        output.append(out_data)

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        for item in output:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

    return output


def parse_judge_results(judge_response_file: str, data_type: str = "labeled") -> Dict[str, int]:
    """
    Parse judge responses and compute evaluation statistics.

    Args:
        judge_response_file: Path to JSONL file containing judge responses
        data_type: Type of data ("labeled", "unlabeled", or "una")

    Returns:
        Dictionary with counts for each rating and total
    """
    if data_type == "una":
        stats = {"A": 0, "B": 0, "total": 0, "parse_errors": 0}
    else:
        stats = {"A": 0, "B": 0, "C": 0, "total": 0, "parse_errors": 0}

    with open(judge_response_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            stats["total"] += 1

            try:
                response = _require_field(data, "response", "in judge response")
                rating = extract_rating(response, data_type)
                stats[rating] += 1
            except (MissingFieldError, JudgeEvaluationError):
                stats["parse_errors"] += 1

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-as-Judge Evaluation for User Simulator"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate judge prompts
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate judge prompts from response files"
    )
    gen_parser.add_argument(
        "--prompts",
        required=True,
        help="Path to original prompts JSONL file"
    )
    gen_parser.add_argument(
        "--responses",
        required=True,
        help="Path to LLM responses JSONL file"
    )
    gen_parser.add_argument(
        "--output",
        required=True,
        help="Path to output judge prompts JSONL file"
    )
    gen_parser.add_argument(
        "--data_type",
        default="labeled",
        choices=["labeled", "unlabeled", "una"],
        help="Data type: labeled, unlabeled, or una (default: labeled)"
    )

    # Parse judge results
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse judge responses and compute statistics"
    )
    parse_parser.add_argument(
        "--responses",
        required=True,
        help="Path to judge responses JSONL file"
    )
    parse_parser.add_argument(
        "--data_type",
        default="labeled",
        choices=["labeled", "unlabeled", "una"],
        help="Data type: labeled, unlabeled, or una (default: labeled)"
    )

    args = parser.parse_args()

    if args.command == "generate":
        print(f"Generating judge prompts...")
        print(f"  Prompts:   {args.prompts}")
        print(f"  Responses: {args.responses}")
        print(f"  Output:    {args.output}")
        print(f"  Data Type: {args.data_type}")

        output = generate_judge_prompts_from_files(
            args.prompts,
            args.responses,
            args.output,
            args.data_type
        )
        print(f"Generated {len(output)} judge prompts")

    elif args.command == "parse":
        print(f"Parsing judge responses from: {args.responses}")
        print(f"Data Type: {args.data_type}")
        stats = parse_judge_results(args.responses, args.data_type)

        print("\n=== Evaluation Results ===")
        print(f"Total responses: {stats['total']}")
        if args.data_type == "una":
            print(f"  A (good - refused): {stats['A']} ({100*stats['A']/max(1,stats['total']):.1f}%)")
            print(f"  B (bad - answered): {stats['B']} ({100*stats['B']/max(1,stats['total']):.1f}%)")
        else:
            print(f"  A (perfect): {stats['A']} ({100*stats['A']/max(1,stats['total']):.1f}%)")
            print(f"  B (normal):  {stats['B']} ({100*stats['B']/max(1,stats['total']):.1f}%)")
            print(f"  C (bad):     {stats['C']} ({100*stats['C']/max(1,stats['total']):.1f}%)")
        if stats['parse_errors'] > 0:
            print(f"  Parse errors: {stats['parse_errors']}")

    else:
        parser.print_help()
