#!/usr/bin/env python3
"""
Simple CLI tool to test LLM providers.

Usage:
  python llm_cli.py --provider openai --model gpt-3.5-turbo [--prompt "Your prompt here"]
  python llm_cli.py --provider gemini --model gemini-2.5-pro-preview-03-25 [--prompt "Your prompt here"]
"""

import os
import sys
import argparse
from src.llm_utils.llm_provider import LLMProvider

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test LLM providers from the command line.")
    
    parser.add_argument("--provider", type=str, choices=["openai", "gemini", "dashscope"], 
                        default="openai", help="LLM provider to use")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", 
                        help="Model ID to use")
    parser.add_argument("--prompt", type=str, 
                        default="Tell me a short joke about AI", 
                        help="Prompt to send to the LLM")
    parser.add_argument("--system", type=str, 
                        default="You are a helpful assistant.", 
                        help="System message content")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive mode")
    
    return parser.parse_args()

def run_interactive(provider):
    """Run the LLM provider in interactive mode."""
    print(f"Interactive mode with {provider.provider.upper()} model: {provider.model_id}")
    print("Type 'exit' or 'quit' to end the session.")
    print("--------------------------------------------")
    
    history = []
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("\nExiting interactive mode.")
            break
        
        # Add user message to history
        history.append({"role": "user", "content": user_input})
        
        try:
            # Get response from LLM
            response = provider.chat_completion(messages=history)
            
            # Add assistant message to history
            history.append({"role": "assistant", "content": response})
            
            # Print response
            print(f"\nAssistant: {response}")
            
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize LLM provider
        provider = LLMProvider(
            provider=args.provider,
            model_id=args.model
        )
        
        if args.interactive:
            # Run in interactive mode
            run_interactive(provider)
        else:
            # Run a single completion
            print(f"Testing {args.provider.upper()} model: {args.model}")
            print(f"Prompt: {args.prompt}")
            print("--------------------------------------------")
            
            # Get response from LLM
            response = provider.simple_completion(
                prompt=args.prompt,
                system_content=args.system
            )
            
            # Print response
            print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 