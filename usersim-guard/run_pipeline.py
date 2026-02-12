#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Pipeline Script for User Simulator.

This script provides a unified interface to:
1. Generate prompts (base, step1, step2)
2. Call API for inference
3. Generate judge prompts for evaluation
4. Parse evaluation results

Usage:
    # Generate step1 prompts
    python run_pipeline.py generate --prompt_type step1 --input data/sample.jsonl --output results/step1_prompts.jsonl

    # Run inference
    python run_pipeline.py infer --input results/step1_prompts.jsonl --output results/step1_responses.jsonl --model gpt-4o

    # Generate judge prompts
    python run_pipeline.py judge --prompts results/base_prompts.jsonl --responses results/base_responses.jsonl --output results/judge_prompts.jsonl --data_type labeled

    # Parse judge results
    python run_pipeline.py stats --responses results/judge_responses.jsonl --data_type labeled

    # Full pipeline (generate + infer)
    python run_pipeline.py full --prompt_type step1 --input data/sample.jsonl --output_dir results/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from user_simulator import (
    DataLoader,
    UserSimulatorPipeline,
    save_jsonl,
    load_jsonl,
    MissingFieldError,
    PromptGenerationError,
    extract_action_from_response,
    generate_judge_prompts_from_files,
    parse_judge_results,
)
from config import DEFAULT_PATHS


def generate_prompts(args):
    """Generate prompts from input data."""
    print(f"=== Generating {args.prompt_type} prompts ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    # Initialize data loader
    loader = DataLoader(
        source_path=args.source_data or DEFAULT_PATHS.get("source_data"),
        db_base_path=args.db_base_path or DEFAULT_PATHS.get("db_base_path"),
    )

    print("Loading data...")
    data = loader.load(args.input)
    print(f"Loaded {len(data)} records")

    # Initialize pipeline
    pipeline_kwargs = {"prompt_type": args.prompt_type}
    if args.prompt_type in ["base", "step2"]:
        pipeline_kwargs["db_base_path"] = args.db_base_path or DEFAULT_PATHS.get("db_base_path")

    pipeline = UserSimulatorPipeline(**pipeline_kwargs)

    # Load step1 responses if needed for step2
    step1_responses = None
    if args.prompt_type == "step2":
        if not args.step1_responses:
            raise ValueError("--step1_responses is required for step2 prompt type")
        step1_responses = load_jsonl(args.step1_responses)

    # Generate prompts
    print("Generating prompts...")
    results = []
    for idx, item in enumerate(data):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(data)}...")

        # Get action for step2
        action = None
        if args.prompt_type == "step2" and step1_responses:
            if idx >= len(step1_responses):
                raise PromptGenerationError(
                    f"No step1 response for index {idx}. "
                    f"Expected {len(data)} responses, got {len(step1_responses)}"
                )
            step1_resp = step1_responses[idx]
            if "response" not in step1_resp:
                raise MissingFieldError(
                    f"Step1 response at index {idx} missing 'response' field"
                )
            action = extract_action_from_response(step1_resp["response"])

        # Validate required fields
        if "clarification_question" not in item:
            raise MissingFieldError(f"Item at index {idx} missing 'clarification_question'")
        if "question_id" not in item:
            raise MissingFieldError(f"Item at index {idx} missing 'question_id'")
        if "question_type" not in item:
            raise MissingFieldError(f"Item at index {idx} missing 'question_type'")

        try:
            result = pipeline(
                item,
                clarification_question=item["clarification_question"],
                question_id=item["question_id"],
                question_type=item["question_type"],
                action=action,
            )
            results.append(result)
        except (MissingFieldError, PromptGenerationError) as e:
            print(f"Error at index {idx}: {e}")
            raise

    print(f"Saving {len(results)} prompts to {args.output}")
    save_jsonl(results, args.output)
    print("Done!")

    return results


def run_inference(args):
    """Run API inference on prompts."""
    print(f"=== Running inference ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")

    from call_api import process_prompts

    # Load data
    data_list = load_jsonl(args.input)
    expected_count = len(data_list)

    # Check if output already exists
    output_path = Path(args.output)
    if output_path.exists():
        try:
            existing_data = load_jsonl(args.output)
            if len(existing_data) >= expected_count:
                print(f"[SKIP] Output already has {len(existing_data)} records. Delete to re-run.")
                return
        except Exception:
            pass

    prompts = [d.get("prompt", "") for d in data_list]
    print(f"Loaded {len(prompts)} prompts")

    process_prompts(
        prompts=prompts,
        model=args.model,
        output_path=args.output,
        data_list=data_list,
        num_threads=args.num_threads,
        temperature=args.temperature,
    )


def generate_judge(args):
    """Generate judge prompts for evaluation."""
    print(f"=== Generating judge prompts ===")
    print(f"Prompts: {args.prompts}")
    print(f"Responses: {args.responses}")
    print(f"Output: {args.output}")
    print(f"Data type: {args.data_type}")

    output = generate_judge_prompts_from_files(
        args.prompts,
        args.responses,
        args.output,
        args.data_type
    )
    print(f"Generated {len(output)} judge prompts")


def show_stats(args):
    """Parse and display judge results statistics."""
    print(f"=== Evaluation Results ===")
    print(f"File: {args.responses}")
    print(f"Data type: {args.data_type}")

    stats = parse_judge_results(args.responses, args.data_type)

    print(f"\nTotal responses: {stats['total']}")
    if args.data_type == "una":
        print(f"  A (good - refused): {stats['A']} ({100*stats['A']/max(1,stats['total']):.1f}%)")
        print(f"  B (bad - answered): {stats['B']} ({100*stats['B']/max(1,stats['total']):.1f}%)")
    else:
        print(f"  A (perfect): {stats['A']} ({100*stats['A']/max(1,stats['total']):.1f}%)")
        print(f"  B (normal):  {stats['B']} ({100*stats['B']/max(1,stats['total']):.1f}%)")
        print(f"  C (bad):     {stats['C']} ({100*stats['C']/max(1,stats['total']):.1f}%)")
    if stats.get('parse_errors', 0) > 0:
        print(f"  Parse errors: {stats['parse_errors']}")


def run_full_pipeline(args):
    """Run full pipeline: generate prompts -> infer."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = output_dir / f"{args.prompt_type}_prompts.jsonl"
    response_file = output_dir / f"{args.prompt_type}_responses.jsonl"

    # Step 1: Generate prompts
    gen_args = argparse.Namespace(
        input=args.input,
        output=str(prompt_file),
        prompt_type=args.prompt_type,
        source_data=args.source_data,
        db_base_path=args.db_base_path,
        step1_responses=args.step1_responses,
    )
    generate_prompts(gen_args)

    # Step 2: Run inference
    infer_args = argparse.Namespace(
        input=str(prompt_file),
        output=str(response_file),
        model=args.model,
        num_threads=args.num_threads,
        temperature=args.temperature,
    )
    run_inference(infer_args)

    print(f"\n=== Pipeline Complete ===")
    print(f"Prompts: {prompt_file}")
    print(f"Responses: {response_file}")


def main():
    parser = argparse.ArgumentParser(
        description="User Simulator Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate prompts")
    gen_parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    gen_parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    gen_parser.add_argument("--prompt_type", "-t", required=True,
                           choices=["base", "step1", "step2"],
                           help="Type of prompt to generate")
    gen_parser.add_argument("--source_data", help="Path to source data file")
    gen_parser.add_argument("--db_base_path", help="Path to DBs directory")
    gen_parser.add_argument("--step1_responses", help="Step1 responses file (for step2)")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run API inference")
    infer_parser.add_argument("--input", "-i", required=True, help="Input JSONL with prompts")
    infer_parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    infer_parser.add_argument("--model", "-m", default="gpt-4o", help="Model name")
    infer_parser.add_argument("--num_threads", type=int, default=4, help="Number of threads")
    infer_parser.add_argument("--temperature", type=float, default=0, help="Temperature")

    # Judge command
    judge_parser = subparsers.add_parser("judge", help="Generate judge prompts")
    judge_parser.add_argument("--prompts", required=True, help="Original prompts file")
    judge_parser.add_argument("--responses", required=True, help="LLM responses file")
    judge_parser.add_argument("--output", "-o", required=True, help="Output judge prompts file")
    judge_parser.add_argument("--data_type", default="labeled",
                              choices=["labeled", "unlabeled", "una"],
                              help="Data type")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show evaluation statistics")
    stats_parser.add_argument("--responses", required=True, help="Judge responses file")
    stats_parser.add_argument("--data_type", default="labeled",
                              choices=["labeled", "unlabeled", "una"],
                              help="Data type")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run full pipeline")
    full_parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    full_parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    full_parser.add_argument("--prompt_type", "-t", required=True,
                            choices=["base", "step1", "step2"],
                            help="Type of prompt to generate")
    full_parser.add_argument("--model", "-m", default="gpt-4o", help="Model name")
    full_parser.add_argument("--source_data", help="Path to source data file")
    full_parser.add_argument("--db_base_path", help="Path to DBs directory")
    full_parser.add_argument("--step1_responses", help="Step1 responses file (for step2)")
    full_parser.add_argument("--num_threads", type=int, default=4, help="Number of threads")
    full_parser.add_argument("--temperature", type=float, default=0, help="Temperature")

    args = parser.parse_args()

    if args.command == "generate":
        generate_prompts(args)
    elif args.command == "infer":
        run_inference(args)
    elif args.command == "judge":
        generate_judge(args)
    elif args.command == "stats":
        show_stats(args)
    elif args.command == "full":
        run_full_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
