#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Client - Clean interface for calling various LLM APIs.

Usage:
    from call_api import LLMClient

    # Use built-in backends
    client = LLMClient("gpt-4o")
    response = client.call("What is 2+2?")

    # Or with messages
    response = client.call([{"role": "user", "content": "Hello!"}])

To add your own model:
    1. Add to OPENAI_MODELS in config.py (for OpenAI-compatible APIs), OR
    2. Subclass LLMClient and override the `call` method
"""

import json
import time
import threading
import requests
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODELS,
    ANTHROPIC_API_KEY,
)

# Optional imports
try:
    import anthropic
except ImportError:
    anthropic = None


class LLMClient:
    """
    Universal LLM Client - supports OpenAI-compatible and Anthropic APIs.

    Args:
        model: Model name (e.g., 'gpt-4o', 'claude-3-haiku')
        temperature: Sampling temperature (default: 0)
        max_tokens: Max tokens to generate (default: 2048)

    Example:
        >>> client = LLMClient("gpt-4o")
        >>> print(client.call("Hello!"))

    To use your own API, subclass and override `call`:
        >>> class MyClient(LLMClient):
        ...     def call(self, messages, **kwargs):
        ...         # Your API logic here
        ...         return "response text"
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 120,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Auto-detect which backend to use based on model name."""
        if self.model in OPENAI_MODELS:
            if not OPENAI_API_KEY:
                raise ValueError(f"Set OPENAI_API_KEY for model '{self.model}'")
            return "openai"

        if self.model.startswith("claude") and ANTHROPIC_API_KEY:
            if anthropic is None:
                raise ImportError("Run: pip install anthropic")
            return "anthropic"

        # Default: try OpenAI-compatible with the model name as-is
        if OPENAI_API_KEY:
            return "openai_direct"

        raise ValueError(
            f"Unknown model '{self.model}'. Options:\n"
            f"  1. Add to OPENAI_MODELS in config.py\n"
            f"  2. Set OPENAI_API_KEY to use as OpenAI-compatible\n"
            f"  3. Subclass LLMClient and override call()"
        )

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Call the LLM and get a response.

        Args:
            messages: Either a string prompt or list of message dicts
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop: Stop sequences

        Returns:
            Generated text response
        """
        # Convert string to messages format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Route to appropriate backend
        if self._backend == "openai":
            return self._call_openai(messages, temp, tokens, stop, OPENAI_MODELS[self.model])
        elif self._backend == "openai_direct":
            return self._call_openai(messages, temp, tokens, stop, self.model)
        elif self._backend == "anthropic":
            return self._call_anthropic(messages, temp, tokens, stop)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def _call_openai(self, messages, temperature, max_tokens, stop, model_id) -> str:
        """Call OpenAI-compatible API."""
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **({"stop": stop} if stop else {}),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call_anthropic(self, messages, temperature, max_tokens, stop) -> str:
        """Call Anthropic Direct API."""
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop,
        )
        return response.content[0].text


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def process_prompts(
    prompts: List[str],
    model: str,
    output_path: str,
    data_list: List[Dict] = None,
    num_threads: int = 4,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> None:
    """
    Process multiple prompts in parallel and save results.

    Args:
        prompts: List of prompt strings
        model: Model name
        output_path: Output JSONL path
        data_list: Original data to merge with responses (optional)
        num_threads: Number of parallel threads
        temperature: Sampling temperature
        max_retries: Max retries per request
    """
    client = LLMClient(model, temperature=temperature)
    data_list = data_list or [{"prompt": p} for p in prompts]
    results = [None] * len(prompts)
    lock = threading.Lock()

    def process_one(idx: int) -> None:
        prompt = prompts[idx]
        if not prompt or not prompt.strip():
            results[idx] = {"response": "[SKIPPED]", **data_list[idx]}
            return

        for attempt in range(max_retries):
            try:
                response = client.call(prompt)
                results[idx] = {"response": response, **{k: v for k, v in data_list[idx].items() if k != "prompt"}}
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    results[idx] = {"response": f"[ERROR: {e}]", **data_list[idx]}
                else:
                    time.sleep(2 ** attempt)

    print(f"Processing {len(prompts)} prompts with {num_threads} threads...")
    print(f"Model: {model} | Temperature: {temperature}")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_one, i) for i in range(len(prompts))]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    # Save results
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM inference on prompts")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL with 'prompt' field")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument("--model", "-m", default="gpt-4o", help="Model name")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Parallel threads")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    args = parser.parse_args()

    # Load prompts
    data_list = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    prompts = [d.get("prompt", "") for d in data_list]
    print(f"Loaded {len(prompts)} prompts from {args.input}")

    process_prompts(
        prompts=prompts,
        model=args.model,
        output_path=args.output,
        data_list=data_list,
        num_threads=args.threads,
        temperature=args.temperature,
    )
