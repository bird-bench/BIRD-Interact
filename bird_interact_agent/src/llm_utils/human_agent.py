"""
Agent implementations for different types of agents (human, LLM, etc).
This module provides a unified interface for different agent types.
"""

from abc import ABC, abstractmethod
from typing import Optional

class Agent(ABC):
    """Base class for all agents."""
    
    @abstractmethod
    def get_response(self, prompt: str, system_content: Optional[str] = None) -> str:
        """
        Get response from the agent.
        
        Args:
            prompt: The current prompt/context
            system_content: System message (optional)
            
        Returns:
            str: Agent's response
        """
        pass

class HumanAgent(Agent):
    """A human agent that allows direct interaction with the environment."""
    
    def __init__(self, verbose=True):
        """
        Initialize the human agent.
        
        Args:
            verbose: Whether to print detailed prompts and instructions
        """
        self.verbose = verbose
        self.template = None  # Can be set by experiment wrapper if needed
        
    def get_response(self, prompt: str, system_content: str = None) -> str:
        """
        Get response from human input.
        
        Args:
            prompt: The current prompt/context
            system_content: System message (ignored for human agent)
            
        Returns:
            str: Human's response in ReAct format
        """
        if self.verbose:
            self._print_interaction_guide(prompt)
        
        # Read multiple lines until EOF
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        response = "\n".join(lines)
        return response
    
    def _print_interaction_guide(self, prompt: str):
        """Print the interaction guide and current context."""
        print("\n" + "="*50)
        print("Current Context:")
        print(prompt)
        print("="*50)
        print("\nEnter your response in ReAct format:")
        print("Format: <thought>your thought</thought>")
        print("        <interaction_object>User or Environment</interaction_object>")
        print("        <action>your action</action>")
        print("\nExample actions:")
        print("- ask('your question')")
        print("- submit('your SQL query')")
        print("- execute('SQL to test')")
        print("- get_schema()")
        print("- get_all_column_meanings()")
        print("- get_column_meaning('column_name')")
        print("- get_all_external_knowledge_names()")
        print("- get_knowledge_definition('knowledge_name')")
        print("- get_all_knowledge_definitions()")
        print("\nYour response (press Ctrl+D or Ctrl+Z when done):") 