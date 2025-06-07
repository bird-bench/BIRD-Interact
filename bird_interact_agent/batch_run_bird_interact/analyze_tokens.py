import json
import jsonlines
import glob
import os
from typing import Dict, List, Tuple, Set
import argparse
from collections import defaultdict
import re

# Model pricing information (per 1M tokens)
MODEL_PRICING = {
    "gemini-2.0-flash-001": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3, "output": 15.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet-20250219#thinking": {"input": 3.00, "output": 15.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "o3-mini-2025-01-31": {"input": 1.10, "output": 4.40},
    "qwen3-235b-a22b": {"input": 0.14, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.00},
    "o1-preview-2024-09-12": {"input": 15.0, "output": 60.00},
    "o1-mini": {"input": 3.0, "output": 12.00},
    "o3": {"input": 10.0, "output": 40.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.00},
    "gemini-2.5-flash-preview-thinking": {"input": 0.15, "output": 3.5},
    "deepseek-r1": {"input": 0.55, "output": 2.19},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "llama4-maverick-instruct-basic": {"input": 0.18, "output": 0.60},
    "llama4-scout-instruct-basic": {"input": 0.08, "output": 0.30},
    "deepseek-r1-0528": {"input": 0.5, "output": 2.15},
    "deepseek-deepseek-r1-0528": {"input": 0.5, "output": 2.15},
}

def extract_turn_number(filename: str) -> int:
    """Extract turn number from filename."""
    match = re.search(r'turn_(\d+)', filename)
    return int(match.group(1)) if match else 0

def extract_phase_number(filename: str) -> int:
    """Extract phase number from filename."""
    match = re.search(r'phase_(\d+)', filename)
    return int(match.group(1)) if match else 0

def calculate_cost(tokens: int, price_per_million: float) -> float:
    """Calculate cost for given number of tokens."""
    return (tokens / 1_000_000) * price_per_million

def analyze_token_usage(output_dir: str, model_name: str, model_type: str="model_base", filter_270_samples: bool=False) -> Dict[str, Dict]:
    """
    Analyze token usage from raw agent output files, grouped by turns.
    
    Args:
        output_dir: Directory containing the raw output files
        model_name: Name of the model used (must be in MODEL_PRICING)
    Returns:
        Dictionary containing token and cost analysis per turn
    """
    if model_name not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(MODEL_PRICING.keys())}")
    
    # Find all agent raw output files
    if model_type == "model_base":
        agent_files = glob.glob(os.path.join(output_dir, "*.model_raw_phase_*.jsonl"))
    elif model_type == "agent":
        agent_files = glob.glob(os.path.join(output_dir, "*.agent_raw_turn_*.jsonl"))
    if not agent_files:
        raise ValueError(f"No agent raw output files found in {output_dir}")
    
    print(f"Found {len(agent_files)} agent files:")
    for f in agent_files:
        print(f"  {os.path.basename(f)}")
    
    # Initialize data structures for turn-based analysis
    turn_data = defaultdict(lambda: {
        "unique_samples": set(),
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_reasoning_tokens": 0
    })
    
    # Process each file
    for file_path in sorted(agent_files):
        if model_type == "model_base":
            turn_num = extract_phase_number(file_path)
            turn_stats = turn_data[turn_num]
        elif model_type == "agent":
            turn_num = extract_turn_number(file_path)
            turn_stats = turn_data[turn_num]
        
        with jsonlines.open(file_path, mode='r') as reader:
            for item in reader:
                if "token_usage" not in item:
                    continue
                
                sample_id = str(item.get('id', ''))
                if filter_270_samples and int(sample_id) >= 270:
                    continue
                if not sample_id:
                    print(f"Warning: Item missing 'id' field in {os.path.basename(file_path)}")
                    continue
                
                turn_stats["unique_samples"].add(sample_id)
                usage = item["token_usage"]
                turn_stats["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
                turn_stats["total_completion_tokens"] += usage.get("completion_tokens", 0)
    
    # Calculate statistics and costs for each turn
    results = {}
    pricing = MODEL_PRICING[model_name]
    
    for turn_num, stats in sorted(turn_data.items()):
        num_samples = len(stats["unique_samples"])
        if num_samples == 0:
            continue
            
        # Calculate averages
        avg_prompt = stats["total_prompt_tokens"] / num_samples
        avg_completion = stats["total_completion_tokens"] / num_samples
        avg_reasoning = stats["total_reasoning_tokens"] / num_samples
        total_output = avg_completion
        
        # Calculate costs for the specified model
        input_cost = calculate_cost(avg_prompt, pricing["input"])
        output_cost = calculate_cost(total_output, pricing["output"])
        
        results[f"turn_{turn_num}"] = {
            "num_samples": num_samples,
            "model": model_name,
            "avg_prompt_tokens": avg_prompt,
            "avg_completion_tokens": avg_completion,
            "avg_reasoning_tokens": avg_reasoning,
            "avg_total_output_tokens": total_output,
            "costs": {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost
            }
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze token usage and costs from raw agent output files')
    parser.add_argument('--output_dir', type=str, required=True, 
                      help='Directory containing the raw output files')
    parser.add_argument('--model', type=str, required=True,
                      help='Model used for the analysis')
    parser.add_argument('--model_type', type=str, required=False, default="agent",
                      choices=["agent", "model_base"],
                      help='Type of model used for the analysis')
    parser.add_argument('--filter_270_samples', action='store_true', required=False,
                      help='Filter out samples after 270')
    args = parser.parse_args()
    
    try:
        results = analyze_token_usage(args.output_dir, args.model, args.model_type, args.filter_270_samples)
        
        # Print results
        print("\n=== Token Usage and Cost Analysis ===")
        print(f"Model Used: {args.model}")
        
        for turn, stats in results.items():
            print(f"\n{turn.upper()}:")
            print(f"Number of Samples: {stats['num_samples']}")
            print("\nAverage Tokens Per Sample:")
            print(f"  Prompt Tokens: {stats['avg_prompt_tokens']:.1f}")
            print(f"  Completion Tokens: {stats['avg_completion_tokens']:.1f}")
            print(f"  Reasoning Tokens: {stats['avg_reasoning_tokens']:.1f}")
            print(f"  Total Output Tokens: {stats['avg_total_output_tokens']:.1f}")
            
            print("\nCost Per Sample (USD):")
            print(f"  Input Cost:  ${stats['costs']['input_cost']:.4f}")
            print(f"  Output Cost: ${stats['costs']['output_cost']:.4f}")
            print(f"  Total Cost:  ${stats['costs']['total_cost']:.4f}")
        
        # Save results if output file specified
        output_file = os.path.join(args.output_dir, f"token_usage_{args.model}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
            
    except Exception as e:
        print(f"Error analyzing token usage: {e}")
        raise

if __name__ == '__main__':
    main() 