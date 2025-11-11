"""
LLM agent implementation using the LLMProvider.
"""

from typing import Optional
from .human_agent import Agent
from .llm_provider import LLMProvider

class LLMAgent(Agent):
    """An agent that uses an LLM for responses."""
    
    def __init__(self, llm_provider: LLMProvider, model_id: str):
        """
        Initialize the LLM agent.
        
        Args:
            llm_provider: The LLM provider to use
            model_id: The model ID to use
        """
        self.llm_provider = llm_provider
        self.model_id = model_id
        
    def get_response(self, prompt: str, system_content: Optional[str] = None) -> str:
        """
        Get response from the LLM.
        
        Args:
            prompt: The current prompt/context
            system_content: System message (optional)
            
        Returns:
            str: LLM's response
        """
        return self.llm_provider.simple_completion(
            prompt=prompt,
            system_content=system_content or "You are a helpful assistant."
        ) 