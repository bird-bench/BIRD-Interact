"""
Token Counter Utility for BIRD-Interact Environment

This module provides functionality to track token usage in system and user simulator interactions.
"""

import tiktoken

class TokenCounter:
    """
    Utility to count tokens in system and user simulator interactions.
    Uses tiktoken for OpenAI-compatible token counting.
    """
    
    def __init__(self):
        """Initialize the token counter with empty counters for all categories."""
        # Initialize counters
        self.counts = {
            "system_input": 0,
            "system_output": 0,
            "user_simulator_input": 0,
            "user_simulator_output": 0
        }
        
        # Load the tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
        except:
            # Fallback to approximate counting if tiktoken not available
            self.tokenizer = None
            print("Warning: tiktoken not available, using approximate token counting")
    
    def count_tokens(self, text):
        """Count tokens in the given text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate token count (4 chars ≈ 1 token)
            return len(text) // 4
    
    def add_system_input(self, text):
        """Count and add system input tokens."""
        count = self.count_tokens(text)
        self.counts["system_input"] += count
        return count
    
    def add_system_output(self, text):
        """Count and add system output tokens."""
        count = self.count_tokens(text)
        self.counts["system_output"] += count
        return count
    
    def add_user_simulator_input(self, text):
        """Count and add user simulator input tokens."""
        count = self.count_tokens(text)
        self.counts["user_simulator_input"] += count
        return count
    
    def add_user_simulator_output(self, text):
        """Count and add user simulator output tokens."""
        count = self.count_tokens(text)
        self.counts["user_simulator_output"] += count
        return count
    
    def get_counts(self):
        """Get a dictionary of all token counts."""
        return self.counts
    
    def get_total(self):
        """Get the total number of tokens across all categories."""
        return sum(self.counts.values())
    
    def reset(self):
        """Reset all token counters to zero."""
        for key in self.counts:
            self.counts[key] = 0
    
    def summary(self):
        """Return a formatted summary of token usage."""
        return (
            f"Token Usage:\n"
            f"  System Input: {self.counts['system_input']}\n"
            f"  System Output: {self.counts['system_output']}\n"
            f"  User Simulator Input: {self.counts['user_simulator_input']}\n"
            f"  User Simulator Output: {self.counts['user_simulator_output']}\n"
            f"  Total: {self.get_total()}"
        )


# Global token counter instance
token_counter = TokenCounter() 