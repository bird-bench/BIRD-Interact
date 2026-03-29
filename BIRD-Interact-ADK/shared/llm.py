"""Unified LLM call interface.

Release: uses LiteLlm (supports any provider).
Local override: place _local_provider.py in this directory (gitignored)
to use a custom backend.
"""

import logging

from shared.config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 5

# Try local override first (gitignored, not in release)
try:
    from shared._local_provider import call_llm, build_adk_model
except ImportError:
    # Default: LiteLlm
    def call_llm(messages: list, model_name: str = None, temperature: float = 0, max_tokens: int = 1024) -> str:
        """Call LLM via LiteLlm. Retries on rate limit / transient errors."""
        import litellm
        model_name = model_name or settings.system_agent_model
        kwargs = dict(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=MAX_RETRIES,
        )
        if settings.litellm_api_base:
            kwargs["api_base"] = settings.litellm_api_base
        if settings.litellm_api_key:
            kwargs["api_key"] = settings.litellm_api_key

        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content.strip()

    def build_adk_model(model_name: str = None):
        """Build ADK-compatible model via LiteLlm with retry config."""
        from google.adk.models.lite_llm import LiteLlm
        model_name = model_name or settings.system_agent_model
        kwargs = dict(
            model=model_name,
            max_tokens=4096,
            num_retries=MAX_RETRIES,
        )
        if settings.litellm_api_base:
            kwargs["api_base"] = settings.litellm_api_base
        if settings.litellm_api_key:
            kwargs["api_key"] = settings.litellm_api_key
        return LiteLlm(**kwargs)
