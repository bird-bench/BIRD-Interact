import argparse
import os
import json
import time
import itertools
import threading
import traceback
import logging  # Add logging import

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig

from src.llm_utils.config import model_config
from src.llm_utils.api_util import read_full_response
import os 

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress verbose logging from other libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)

# --- Vertex AI Specific Imports ---
try:
    import vertexai
    from google.oauth2 import service_account
    from vertexai.generative_models import (
        GenerativeModel,
        GenerationConfig as VertexGenerationConfig,
        HarmCategory as VertexHarmCategory,
        HarmBlockThreshold as VertexHarmBlockThreshold,
    )
    from google.api_core import exceptions as core_exceptions

    _vertex_ai_available = True
except ImportError:
    logging.warning(
        "google-cloud-aiplatform, google-auth or google-api-core libraries not found. Vertex AI models will not be available."
    )
    _vertex_ai_available = False
# --- End Vertex AI Imports ---

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def new_directory(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


GEMINI_API_KEYS = model_config.get("gemini-1.5-flash", [])  # Use .get for safety
if not GEMINI_API_KEYS:
    logging.warning("No API keys found for 'gemini-1.5-flash' in config.")
# Create an infinite key cycle only if keys exist
gemini_key_cycle = itertools.cycle(GEMINI_API_KEYS) if GEMINI_API_KEYS else None


# +++ Vertex AI Configuration +++
GCP_PROJECT = "Your GCP Project ID"  # Your GCP Project ID
GCP_REGION = "Your GCP Region"  # Your GCP Region
# # Consider moving this path to config.py or environment variables
GCP_CREDENTIALS_PATH = "Your GCP Credentials Path"

vertex_credentials = None
if _vertex_ai_available:  # Only attempt if library is imported
    if GCP_CREDENTIALS_PATH and os.path.exists(GCP_CREDENTIALS_PATH):
        try:
            vertex_credentials = service_account.Credentials.from_service_account_file(
                GCP_CREDENTIALS_PATH
            )
            logging.info("Vertex AI credentials loaded successfully.")
        except Exception as e:
            logging.warning(
                f"Failed to load Vertex AI credentials from {GCP_CREDENTIALS_PATH}: {e}. Will attempt to use Application Default Credentials (ADC)."
            )
            vertex_credentials = None  # Fallback to ADC
    else:
        logging.info(
            "Vertex AI credentials path not specified or not found. Attempting to use Application Default Credentials (ADC)."
        )
else:
    logging.info("Vertex AI SDK not available, skipping credential loading.")

# --- Flags and Lock for Thread-Safe Vertex AI Initialization ---
_vertex_ai_initialized = False
_vertex_ai_init_lock = threading.Lock()
# +++ End Vertex AI Configuration +++


def write_response(results, data_list, output_path):
    # This function seems designed for writing all results at once,
    # but the current multi-threaded approach writes incrementally.
    # Keep it for potential future use, but it's not called by collect_response_from_api.
    formatted_data = []
    # Assuming results is a list matching data_list
    for i, data in enumerate(data_list):
        if i < len(results):
            data["response"] = results[
                i
            ]  # Assuming results contains only the content part
            formatted_data.append(data)

    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        with open(output_path, "w") as f:
            for instance in formatted_data:
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")


def api_request(messages, engine, client, backend, **kwargs):
    """
    Calls the underlying LLM endpoint depending on the 'backend'.
    Includes more robust error handling and retry logic.
    """
    retries = 6  # Max retries for transient errors
    retry_delay = 10 # Initial delay in seconds

    for attempt in range(retries):
        try:
            # --- Existing backend logic (o1, deepseek, openai, anthropic, genai) ---
            # (Keep your existing logic here, ensure it returns tuple: (reasoning, content, usage) or raises Exception)
            if "o1" in engine or "o3" in engine or "o4" in engine:
                completion = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    max_completion_tokens=10000,
                )
                logging.debug(f"Token usage (o1/o3/o4): {completion.usage}")
                token_usage = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                    "reasoning_tokens": getattr(
                        completion.usage.completion_tokens_details,
                        "reasoning_tokens",
                        None,
                    ),
                    "rejected_prediction_tokens": getattr(
                        completion.usage.completion_tokens_details,
                        "rejected_prediction_tokens",
                        None,
                    ),
                }
                return None, completion.choices[0].message.content, token_usage
            elif engine == "claude-3-7-sonnet-20250219#thinking" or engine == "gemini-2.5-flash-preview-thinking":
                completion = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    max_tokens=10000,  # Ensure this is appropriate
                )
                reasoning_content = completion.choices[0].message.reasoning_content
                content = completion.choices[0].message.content
                token_usage = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
                logging.debug(f"Token usage (Claude Thinking): {token_usage}")
                return reasoning_content, content, token_usage
            elif backend == "openai":
                reasoning = kwargs.get("reasoning", False)
                max_tokens = kwargs.get("max_tokens", 512) if not reasoning else 10000
                completion = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0),
                    max_tokens=max_tokens,
                    top_p=kwargs.get("top_p", 1),
                    frequency_penalty=kwargs.get("frequency_penalty", 0),
                    presence_penalty=kwargs.get("presence_penalty", 0),
                    stop=kwargs.get("stop", None),
                )   
                reasoning_content = getattr(
                    completion.choices[0].message, "reasoning_content", None
                )
                if reasoning_content is None:
                    reasoning_content = getattr(
                        completion.choices[0].message, "reasoning", None
                    )
                content = completion.choices[0].message.content
                token_usage = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
                logging.debug(f"Token usage (OpenAI): {token_usage}")
                return reasoning_content, content, token_usage

            elif backend == "anthropic":
                message = client.messages.create(
                    model=engine,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0),
                    max_tokens=kwargs.get("max_tokens", 512),
                    top_p=kwargs.get("top_p", 1),
                    stop_sequences=kwargs.get("stop", None),
                )
                usage_data = message.usage
                token_usage = {
                    "prompt_tokens": usage_data.input_tokens,
                    "completion_tokens": usage_data.output_tokens,
                    "total_tokens": usage_data.input_tokens + usage_data.output_tokens,
                    "cache_creation_input_tokens": usage_data.cache_creation_input_tokens,
                    "cache_read_input_tokens": usage_data.cache_read_input_tokens,
                }
                content = message.content[0].text
                logging.debug(f"Token usage (Anthropic): {token_usage}")
                return None, content, token_usage
            elif backend == "genai":
                # Ensure key cycle is available
                if not gemini_key_cycle:
                    raise ValueError(
                        "Gemini API keys not configured for 'genai' backend."
                    )
                current_key = next(gemini_key_cycle)
                genai.configure(api_key=current_key)  # Configure key for this attempt

                response = client.generate_content(
                    messages[0]["content"],
                    generation_config=GenerationConfig(  # Ensure this is the correct GenAI config class
                        temperature=kwargs.get("temperature", 0),
                        top_p=kwargs.get("top_p", 1),
                        max_output_tokens=kwargs.get(
                            "max_tokens", 512
                        ),  # Ensure this name is correct for genai SDK
                        # presence_penalty=kwargs.get("presence_penalty", 0), # Check if supported
                        # frequency_penalty=kwargs.get("frequency_penalty", 0), # Check if supported
                        stop_sequences=kwargs.get("stop", None),
                    ),
                    # Add safety settings if needed for genai backend
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                # Check for blocked response *before* accessing text
                if not response.candidates:
                    finish_reason_msg = "Unknown"
                    if (
                        hasattr(response, "prompt_feedback")
                        and response.prompt_feedback
                    ):
                        finish_reason_msg = f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason}"
                    error_message = f"Model response blocked or empty (genai). Reason: {finish_reason_msg}"
                    logging.warning(error_message)
                    # Decide: retry or return error? Returning error for now.
                    return None, error_message, None  # No usage data if blocked

                token_usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,  # Check name: candidates? completion?
                    "total_tokens": response.usage_metadata.total_token_count,
                }
                logging.debug(f"Token usage (GenAI): {token_usage}")
                return None, response.text, token_usage

            # +++ Vertex AI Backend Logic (Modified) +++
            elif backend == "vertexai":
                if not _vertex_ai_available:
                    raise RuntimeError("Vertex AI SDK not available for this request.")
                if not messages or "content" not in messages[0]:
                    raise ValueError("Invalid message format for Vertex AI backend.")
                prompt_content = messages[0]["content"]

                vertex_generation_config = VertexGenerationConfig(
                    temperature=kwargs.get(
                        "temperature", 0
                    ),  # Match Vertex defaults or your preference
                    top_p=kwargs.get(
                        "top_p", 1
                    ),  # Match Vertex defaults or your preference
                    max_output_tokens=kwargs.get("max_tokens", 2048),
                )
                stop_sequences = kwargs.get("stop", None)
                if stop_sequences:
                    vertex_generation_config.stop_sequences = stop_sequences

                safety_settings = {
                    VertexHarmCategory.HARM_CATEGORY_HATE_SPEECH: VertexHarmBlockThreshold.BLOCK_NONE,
                    VertexHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: VertexHarmBlockThreshold.BLOCK_NONE,
                    VertexHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: VertexHarmBlockThreshold.BLOCK_NONE,
                    VertexHarmCategory.HARM_CATEGORY_HARASSMENT: VertexHarmBlockThreshold.BLOCK_NONE,
                }

                # Client is the GenerativeModel instance
                response = client.generate_content(
                    prompt_content,
                    generation_config=vertex_generation_config,
                    safety_settings=safety_settings,
                )

                # Check for blocked response *before* accessing content/usage
                if not response.candidates:
                    finish_reason_msg = "Unknown"
                    if (
                        hasattr(response, "prompt_feedback")
                        and response.prompt_feedback
                    ):
                        finish_reason_msg = f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason}"
                    # Check candidate finish reason if available even if blocked
                    elif (
                        hasattr(response, "candidates")
                        and response.candidates
                        and hasattr(response.candidates[0], "finish_reason")
                    ):
                        finish_reason_msg = f"Candidate Finish Reason: {response.candidates[0].finish_reason}"

                    error_message = f"Model response blocked or empty (Vertex AI). Reason: {finish_reason_msg}"
                    logging.warning(error_message)
                    return None, error_message, None  # Return error, no usage data

                # Extract token usage
                token_usage = None
                if hasattr(response, "usage_metadata"):
                    token_usage = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count,
                    }
                    if logging.getLogger('src.llm_utils.call_api_batch').isEnabledFor(logging.DEBUG):
                        logging.debug(f"Token usage (Vertex AI): {token_usage}")
                else:
                    logging.warning(
                        "Warning: Token usage metadata not found in Vertex AI response."
                    )

                # Extract content safely
                try:
                    generated_content = response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError) as content_err:
                    finish_reason = getattr(
                        response.candidates[0], "finish_reason", "UNKNOWN"
                    )
                    error_message = f"Failed to extract content part from Vertex AI response. Finish Reason: {finish_reason}. Error: {content_err}"
                    logging.error(f"Error: {error_message}")
                    # Return error message instead of content
                    return None, error_message, token_usage

                return None, generated_content, token_usage
            # +++ End Vertex AI Backend Logic +++

        except Exception as e:
            is_retryable = True
            if is_retryable and attempt < retries - 1:
                logging.error(f"ERROR: {e}")
                logging.info(
                    f"Retryable error detected. Waiting {retry_delay} seconds before retry..."
                )
                time.sleep(min(retry_delay, 40))
                retry_delay *= 1.5  # Exponential backoff (optional)
                # Key rotation specifically for genai backend on retry
                if backend == "genai" and gemini_key_cycle:
                    logging.info("Rotating GenAI API key for retry...")
                    # Key is configured at the start of the next loop iteration
                continue  # Go to next attempt
            else:
                logging.error("Non-retryable error or max retries reached. Failing request.")
                # Re-raise the exception to be caught by worker_function
                raise e

    # Should not be reached if retries are exhausted and exception is raised
    return None, "Error: Max retries exceeded", None


def call_api_model(
    messages,
    model_name,
    temperature=0,
    max_tokens=6000,  # Default max_tokens used if not overridden
    top_p=1,
    frequency_penalty=0,  # Note: May not be supported by all backends
    presence_penalty=0,  # Note: May not be supported by all backends
    # timeout=10, # Timeout not directly used in current api_request logic
    stop=None,
    return_format="",  # Note: May not be supported by all backends
):
    """
    Sets up the correct backend client + model engine, then calls 'api_request'.
    Includes thread-safe Vertex AI initialization.
    """
    client = None
    backend = None
    reasoning = False
    # engine usually becomes the specific model ID/name for the API call
    engine = model_name  # Default assumption, override as needed below

    # --- Backend/Client Setup Logic ---

    # --- Vertex AI Model Routing (Modified) ---
    if model_name in [
        "gemini-2.5-pro-preview-03-25",  # Use the EXACT Vertex AI model ID
        "gemini-2.0-flash-001",
        "gemini-2.0-flash",
        "gemini-2.5-flash-preview-05-20"
    ]:
        engine = model_name  # engine is the vertex model ID
        if not _vertex_ai_available:
            raise RuntimeError(
                f"Vertex AI SDK not available, cannot use model {engine}"
            )

        global _vertex_ai_initialized, _vertex_ai_init_lock, vertex_credentials
        # --- Thread-safe Vertex AI Initialization ---
        if not _vertex_ai_initialized:
            with _vertex_ai_init_lock:
                # Double-check inside lock
                if not _vertex_ai_initialized:
                    current_thread_name = threading.current_thread().name
                    logging.info(
                        f"Attempting Vertex AI Initialization (Thread: {current_thread_name})..."
                    )
                    try:
                        # Only call vertexai.init()
                        vertexai.init(
                            project=GCP_PROJECT,
                            location=GCP_REGION,
                            credentials=vertex_credentials,  # Pass loaded creds or None for ADC
                        )
                        _vertex_ai_initialized = True
                        logging.info(
                            f"Vertex AI Initialized Successfully (Thread: {current_thread_name})."
                        )
                    except Exception as e:
                        logging.error(
                            f"ERROR: Failed to initialize Vertex AI in thread {current_thread_name}: {e}"
                        )
                        raise RuntimeError(
                            f"Failed to initialize Vertex AI: {e}"
                        ) from e

        # Check flag *after* lock attempt
        if not _vertex_ai_initialized:
            raise RuntimeError("Vertex AI could not be initialized. Cannot proceed.")

        # --- Create Vertex AI Generative Model Client ---
        try:
            current_thread_name = threading.current_thread().name
            client = GenerativeModel(engine)  # engine is the Vertex Model ID
            backend = "vertexai"
            if logging.getLogger('src.llm_utils.call_api_batch').isEnabledFor(logging.DEBUG):
                logging.debug(
                    f"Creating GenerativeModel for '{engine}' (Thread: {current_thread_name})"
                )
                logging.debug(
                    f"Using Vertex AI backend for model: {engine} (Thread: {current_thread_name})"
                )
        except Exception as e:
            current_thread_name = threading.current_thread().name
            logging.error(
                f"ERROR: Failed to create Vertex AI GenerativeModel for '{engine}' in thread {current_thread_name}: {e}"
            )
            raise RuntimeError(
                f"Failed to create Vertex AI GenerativeModel for '{engine}': {e}"
            ) from e
    # --- End Vertex AI Model Routing ---

    elif model_name in [
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-thinking-exp-01-21",
    ]:
        # engine = model_name # Already set
        if not gemini_key_cycle:
            raise ValueError("Gemini API keys not configured for 'genai' backend.")
        # Key configuration happens within api_request for genai backend during retry/initial call
        client = genai.GenerativeModel(engine)
        backend = "genai"
    elif model_name in [
        "deepseek/deepseek-r1-0528"
    ]:
        engine = model_name
        client = OpenAI(
            base_url=model_config["openrouter"]["base_url"],
            api_key=model_config["openrouter"]["api_key"],
        )
        backend = "openai"
    elif model_name in [
        "gpt-4o-2024-11-20",
        "claude-3-7-sonnet-20250219#thinking",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "o3-mini-2025-01-31",
        "o1-mini",
        "o3",
        "o1-preview-2024-09-12",
        "gpt-4.1",
        "o4-mini",
        "claude-sonnet-4-20250514",
        "gemini-2.5-flash-preview-thinking",
        "qwen3-235b-a22b",
        "llama4-maverick-instruct-basic",
        "llama4-scout-instruct-basic",
    ]:
        engine = model_name
        client = OpenAI(
            base_url=model_config["openai"]["base_url"],
            api_key=model_config["openai"]["api_key"],
        )
        backend = "openai"
    else:
        logging.error(f"ERROR: Unsupported model name: {model_name}")
        raise ValueError(f"Unsupported model name: {model_name}")

    # Ensure client and backend were set
    if client is None or backend is None:
        raise RuntimeError(
            f"Could not configure client or backend for model: {model_name}"
        )

    if model_name in ["claude-3-7-sonnet-20250219#thinking", "o3-mini-2025-01-31", "o1-mini","o3", "o1-preview-2024-09-12", "o4-mini", "deepseek/deepseek-r1-0528"]:
        reasoning = True

    kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
        "return_format": return_format,  # Pass along, api_request handles if supported
        "reasoning": reasoning,
    }
    # Call the API request function
    return api_request(messages, engine, client, backend, **kwargs)


def worker_function(task, data_list, output_path, lock, stop=None, max_tokens=6000):
    """
    Processes a single prompt with robust error handling and logging.
    Writes result (success or error) to the output file incrementally.
    """
    prompt, idx, model_name, return_format = task
    messages = [{"role": "user", "content": prompt}]
    reasoning_content = None
    token_usage = None
    # Default to error message in case of failure
    content = f"Error: Processing failed unexpectedly for index {idx}"
    processed_successfully = False  # Flag to track success
    current_thread_name = threading.current_thread().name

    try:
        logging.debug(f"Worker {current_thread_name} START processing index {idx} with model {model_name}")

        # Core API call logic
        response_tuple = call_api_model(
            messages,
            model_name,
            return_format=return_format,
            # You might need to pass temperature, max_tokens etc. here if
            # call_api_model doesn't get them from args or has fixed defaults
            # that need overriding per task.
            max_tokens=max_tokens, # Example if needed
            stop=stop,
        )

        # Validate response structure
        if isinstance(response_tuple, tuple) and len(response_tuple) == 3:
            reasoning_content, content_result, token_usage = response_tuple
            # Check if the content itself indicates an error (e.g., blocked response)
            if (
                content_result is None
                or "Model response blocked" in str(content_result)
                or "Error:" in str(content_result)
            ):
                # logging.warning(f"Worker {current_thread_name} received API-level error/block for index {idx}: {content_result}")
                logging.warning(f"Worker {current_thread_name} received API-level error/block for index {idx}: ...")
                content = str(content_result)  # Store the error message as content
                processed_successfully = (
                    False  # Treat API error/block as failure for success flag
                )
            else:
                content = content_result  # Store successful content
                processed_successfully = True  # Mark as successful API call
            logging.debug(f"Worker {current_thread_name} received response for index {idx}. Success: {processed_successfully}. Content snippet: {str(content)[:100]}...")
        else:
            # Handle cases where call_api_model returns unexpected format
            error_msg = f"Error: Unexpected response format from call_api_model for index {idx}. Type: {type(response_tuple)}, Value: {response_tuple}"
            logging.error(f"Worker {current_thread_name} {error_msg}")
            content = error_msg
            processed_successfully = False

    except Exception as e:
        # Catch ANY exception during the process (init, client creation, API call)
        logging.error(f"FATAL ERROR in worker {current_thread_name} for index {idx}: {e}")
        traceback.print_exc()  # Print full traceback for debugging
        content = f"Error: Exception during processing for index {idx}: {type(e).__name__} - {e}"
        processed_successfully = False

    # --- Write result (success or error) to file safely ---
    try:
        # Use lock to ensure thread-safe file writing
        with lock:
            # Open in append mode ('a')
            with open(output_path, "a", encoding="utf-8") as f:
                # Get corresponding original data item
                # Use a copy to avoid modifying the shared data_list object
                row = data_list[idx].copy()
                # Update with results or error messages
                row["response"] = content
                row["reasoning_content"] = (
                    reasoning_content if reasoning_content else ""
                )
                row["token_usage"] = (
                    token_usage if token_usage else {}
                )  # Use empty dict for consistency
                # Add index for final sorting
                row["_index"] = idx
                # Write as JSON line
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as write_e:
        # Log critical error if writing fails
        logging.critical(f"Worker {current_thread_name} FAILED to write result for index {idx} to file {output_path}: {write_e}")

    # Return index and success status (optional, useful for tracking)
    return idx, processed_successfully


def final_sort_jsonl_by_index(file_path):
    """
    Reads an existing JSONL file, sorts it by the '_index' field,
    removes the '_index' field, and overwrites the file.
    Handles potential errors during file reading/writing.
    """
    all_data = []
    try:
        logging.info(f"Attempting to read and sort file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "_index" not in row:
                        logging.warning(
                            f"Warning: Missing '_index' in line {line_num+1}. Skipping line: {line}"
                        )
                        continue
                    all_data.append(row)
                except json.JSONDecodeError as json_err:
                    logging.warning(
                        f"Warning: Failed to decode JSON in line {line_num+1}. Error: {json_err}. Skipping line: {line}"
                    )
                    continue

        if not all_data:
            logging.warning("Warning: No valid data with '_index' found to sort.")
            return

        # Sort by '_index'
        all_data.sort(key=lambda x: x["_index"])
        logging.info(f"Successfully read {len(all_data)} records. Sorting complete.")

        # Overwrite the file, removing the '_index' field
        with open(file_path, "w", encoding="utf-8") as fout:
            for row in all_data:
                row.pop("_index", None)  # Remove index field
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        logging.info(f"Successfully overwrote {file_path} with sorted data.")

    except FileNotFoundError:
        logging.error(f"Error: File not found for sorting: {file_path}")
    except Exception as sort_err:
        logging.error(f"Error during final sorting of {file_path}: {sort_err}")


def collect_response_from_api(
    prompt_list,
    model_name,
    data_list,  # Pass the original data list to worker
    output_path,
    num_threads=8,
    start_index=0,
    return_format="",
    stop=None,
    max_tokens=6000,
):
    """
    Uses ThreadPoolExecutor to process prompts concurrently.
    Writes results incrementally and sorts the final file.
    """
    # Validate start_index
    if start_index < 0 or start_index >= len(prompt_list):
        logging.warning(
            f"Warning: start_index {start_index} is out of bounds (0-{len(prompt_list)-1}). Setting to 0."
        )
        start_index = 0

    # Prepare tasks only for the required range
    tasks = [
        (prompt_list[i], i, model_name, return_format)
        for i in range(start_index, len(prompt_list))
    ]

    if not tasks:
        logging.warning("No tasks to process based on start_index and prompt_list length.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Check if output_path includes a directory
        new_directory(output_dir)

    # --- File Handling ---
    # If starting fresh (or start_index is 0), clear the file with 'w' mode first.
    # Otherwise, we rely on append ('a') mode in the worker.
    if start_index == 0 and os.path.exists(output_path):
        logging.info(f"Clearing existing output file: {output_path}")
        try:
            open(output_path, "w").close()
        except IOError as e:
            logging.warning(
                f"Warning: Error clearing output file {output_path}: {e}. Appending may lead to duplicates if run was interrupted."
            )

    # Lock for protecting the write operation in worker_function
    lock = threading.Lock()

    # --- Thread Pool Execution ---
    logging.info(f"Starting processing {len(tasks)} tasks with {num_threads} threads...")
    successful_tasks = 0
    failed_tasks = 0
    MULTI_THREAD = True
    if MULTI_THREAD:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks
            futures = {
                executor.submit(worker_function, t, data_list, output_path, lock, stop=stop, max_tokens=max_tokens): t
                for t in tasks
            }

            # Process results as they complete
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing prompts"
            ):
                task_info = futures[future]  # Get original task info
                task_idx = task_info[1]
                try:
                    # Get result from worker (idx, success_flag)
                    result_idx, success_flag = future.result()
                    if success_flag:
                        successful_tasks += 1
                    else:
                        failed_tasks += 1
                        logging.warning(f"Task for index {result_idx} completed with failure.")
                except Exception as exc:
                    # Catch unexpected errors from the future itself (less likely with worker handling)
                    logging.error(
                        f"FATAL: Task for index {task_idx} generated an exception in future: {exc}"
                    )
                    failed_tasks += 1
    else:
        for task in tasks:
            worker_function(task, data_list, output_path, lock, stop=stop, max_tokens=max_tokens)


    logging.info(
        f"Processing complete. Successful tasks: {successful_tasks}, Failed tasks: {failed_tasks}"
    )

    # --- Final Sort ---
    # Perform a final sort of the output file based on '_index'
    if successful_tasks + failed_tasks > 0:  # Only sort if something was processed
        logging.info("Sorting the output file...")
        final_sort_jsonl_by_index(output_path)
    else:
        logging.info("No tasks were processed, skipping final sort.")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--prompt_path", type=str, default="input.jsonl")  # Make required
    args_parser.add_argument("--output_path", type=str, default="output.jsonl")  # Make required
    args_parser.add_argument(
        "--model_name", type=str, default="gemini-2.0-flash-001"
    )  # Sensible default
    args_parser.add_argument(
        "--num_threads", type=int, default=8
    )  # Add threads argument
    args_parser.add_argument("--return_format", type=str, default="")
    args_parser.add_argument("--start_index", type=int, default=0)
    args_parser.add_argument("--limit", type=int, default=None)
    args_parser.add_argument("--stop", type=str, default=None)
    # Add logging level argument
    args_parser.add_argument("--log_level", type=str, default="INFO",
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                           help="Set the logging level")
    args = args_parser.parse_args()

    # Setup logging with the specified level
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)

    # --- Load Data ---
    try:
        data_list = load_jsonl(args.prompt_path)
        if not data_list:
            logging.error(
                f"Error: Prompt file {args.prompt_path} is empty or could not be loaded."
            )
            exit(1)
        prompts = [data["prompt"] for data in data_list]
        logging.info(f"Loaded {len(prompts)} prompts. First prompt: {prompts[0][:100]}...")
    except FileNotFoundError:
        logging.error(f"Error: Prompt file not found: {args.prompt_path}")
        exit(1)
    except Exception as load_err:
        logging.error(f"Error loading prompts from {args.prompt_path}: {load_err}")
        exit(1)

    # --- Apply Limit (if any) ---
    # Limit affects the list *before* passing to collect_response_from_api
    effective_data_list = data_list
    effective_prompts = prompts
    if args.limit is not None and args.limit > 0:
        logging.info(f"Applying limit: processing first {args.limit} prompts.")
        effective_data_list = data_list[: args.limit]
        effective_prompts = prompts[: args.limit]
        if args.start_index >= args.limit:
            logging.error(
                f"Error: start_index ({args.start_index}) >= limit ({args.limit}). No prompts to process."
            )
            exit(1)

    # --- Start API Collection ---
    collect_response_from_api(
        effective_prompts,  # Use potentially limited list
        args.model_name,
        effective_data_list,  # Pass corresponding data list
        args.output_path,
        num_threads=args.num_threads,  # Use argument
        start_index=args.start_index,
        return_format=args.return_format,
        stop=args.stop,
    )

    logging.info("Script finished.")
