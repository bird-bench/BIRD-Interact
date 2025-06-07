import os
import time
import openai
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI
from src.llm_utils.config import model_config

# Try importing Vertex AI utils
try:
    from src.llm_utils.vertex_ai_simple import initialize_vertex_ai, call_vertex_ai
    _vertex_ai_available = True
except ImportError:
    print("WARNING: Vertex AI utilities not found. Gemini models will not be available.")
    _vertex_ai_available = False

class LLMProvider:
    """
    Centralized LLM provider that supports multiple LLM backends.
    
    Currently supported:
    - OpenAI (default)
    - Gemini (via Vertex AI)
    """
    
    def __init__(
        self, 
        provider: str = "openai", 
        model_id: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        token_counter = None,
        max_retries: int = 5,
        retry_delay: float = 5.0
    ):
        """
        Initialize the LLM provider.
        
        Args:
            provider: The LLM provider to use ('openai' or 'gemini')
            model_id: The model ID to use
            api_key: The API key for the provider
            base_url: The base URL for the provider
            token_counter: Optional token counter instance to track token usage
            max_retries: Maximum number of retries for failed API calls
            retry_delay: Initial delay between retries in seconds
        """
        self.provider = provider.lower()
        self.model_id = model_id
        self.token_counter = token_counter
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up OpenAI config if needed
        if self.provider == "openai":
            # Get API key from environment or config file
            self.api_key = api_key or model_config["openai"]["api_key"]
            # Get base URL from environment or config file
            self.base_url = base_url or model_config["openai"]["base_url"]
            
            # Set up OpenAI client
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
        # Set up Gemini config if needed
        elif self.provider == "gemini":
            if not _vertex_ai_available:
                raise ValueError("Vertex AI utilities not available. Cannot use Gemini models.")
            
            # Initialize Vertex AI
            try:
                initialize_vertex_ai()
                print(f"Using Gemini model: {self.model_id}")
            except Exception as e:
                raise ValueError(f"Failed to initialize Vertex AI: {e}")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_client(self):
        """Get the underlying client for the current provider."""
        if self.provider == "openai":
            return self.client
        elif self.provider == "gemini":
            # Return a custom wrapper for Gemini
            return self.get_gemini_wrapper()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_gemini_wrapper(self):
        """Get a wrapper for Gemini that mimics the OpenAI interface."""
        class GeminiClientWrapper:
            def __init__(self, model_name):
                self.model_name = model_name
            
            def chat_completions_create(self, model, messages, temperature=0, max_tokens=1000, **kwargs):
                # Convert OpenAI messages format to a prompt for Gemini
                prompt = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        prompt = content + "\n\n"
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"
                
                # Call Gemini API
                _, gemini_response, _ = call_vertex_ai(
                    prompt=prompt,
                    model_name=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=kwargs.get("top_p", 1.0)
                )
                
                # Create a response structure similar to OpenAI
                class Choice:
                    class Message:
                        def __init__(self, content):
                            self.content = content
                            
                    def __init__(self, content):
                        self.message = self.Message(content)
                        
                class Response:
                    def __init__(self, content):
                        self.choices = [Choice(content)]
                
                return Response(gemini_response)
        
        return GeminiClientWrapper(self.model_id)
    
    def _should_retry_error(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        error_str = str(error).lower()
        return any(err in error_str for err in ["503", "socket closed", "unavailable"])

    def _execute_with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry logic.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        current_delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:  # Don't sleep on the last attempt
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    print(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 1.5  # Exponential backoff
                continue
                # if self._should_retry_error(e):
                #     if attempt < self.max_retries - 1:  # Don't sleep on the last attempt
                #         print(f"Attempt {attempt + 1} failed with error: {e}")
                #         print(f"Retrying in {current_delay} seconds...")
                #         time.sleep(current_delay)
                #         current_delay *= 2  # Exponential backoff
                #     continue
                # else:
                #     # For other types of errors, don't retry
                #     raise
        
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} attempts failed. Last error: {last_error}")

    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0, 
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Get a chat completion from the LLM provider with retry logic.
        
        Args:
            messages: A list of message objects with 'role' and 'content'
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            top_p: The top-p value for sampling
            frequency_penalty: The frequency penalty
            presence_penalty: The presence penalty
            stop: A list of stop sequences
            
        Returns:
            The generated response text
        """
        def _chat_completion_impl():
            # Track token usage if a counter is provided
            if self.token_counter:
                for msg in messages:
                    if msg["role"] == "user":
                        self.token_counter.add_system_input(msg["content"])
            
            if self.provider == "openai" or self.provider == "dashscope":
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop
                )
                
                response_text = response.choices[0].message.content.strip()
                
            elif self.provider == "gemini":
                # Convert messages to a prompt for Gemini
                prompt = ""
                system_prompt = ""
                
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        system_prompt = content
                    elif role == "user":
                        prompt += content
                
                # Call Gemini API
                _, response_text, token_usage = call_vertex_ai(
                    prompt=system_prompt + "\n\n" + prompt,
                    model_name=self.model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                
                # If there's an error with Gemini, raise an exception
                if response_text.startswith("Error:"):
                    raise Exception(f"Gemini error: {response_text}")
            
            # Track token usage if a counter is provided
            if self.token_counter:
                self.token_counter.add_system_output(response_text)
            
            return response_text

        return self._execute_with_retry(_chat_completion_impl)

    def simple_completion(
        self, 
        prompt: str, 
        system_content: str = "You are a helpful assistant.",
        temperature: float = 0, 
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Simplified interface for getting a completion from the LLM provider with retry logic.
        
        Args:
            prompt: The prompt to send to the LLM
            system_content: The system message content
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            top_p: The top-p value for sampling
            frequency_penalty: The frequency penalty
            presence_penalty: The presence penalty
            stop: A list of stop sequences
            
        Returns:
            The generated response text
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        ) 