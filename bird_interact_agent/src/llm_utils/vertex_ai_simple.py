import argparse
import os
import json
import time
import traceback
from tqdm import tqdm

# --- Vertex AI Specific Imports ---
try:
    import vertexai
    from google.oauth2 import service_account
    from vertexai.generative_models import (
        GenerativeModel,
        GenerationConfig as VertexGenerationConfig,
        HarmCategory as VertexHarmCategory,
        HarmBlockThreshold as VertexHarmBlockThreshold
    )
    from google.api_core import exceptions as core_exceptions
    _vertex_ai_available = True
except ImportError:
    print("WARNING: google-cloud-aiplatform, google-auth or google-api-core libraries not found. Vertex AI models will not be available.")
    _vertex_ai_available = False

# --- Vertex AI Configuration ---
GCP_PROJECT = "Your GCP Project ID"  # Your GCP Project ID
GCP_REGION = "Your GCP Region"       # Your GCP Region
GCP_CREDENTIALS_PATH = "Your GCP Credentials Path"

def load_credentials():
    """Load Vertex AI credentials from service account file."""
    if not _vertex_ai_available:
        return None
        
    if GCP_CREDENTIALS_PATH and os.path.exists(GCP_CREDENTIALS_PATH):
        try:
            credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_PATH)
            print(f"Vertex AI credentials loaded successfully from {GCP_CREDENTIALS_PATH}.")
            return credentials
        except Exception as e:
            print(f"WARNING: Failed to load Vertex AI credentials: {e}. Will attempt to use Application Default Credentials (ADC).")
            return None
    else:
        print(f"INFO: Vertex AI credentials path not found. Using Application Default Credentials (ADC).")
        return None

def initialize_vertex_ai():
    """Initialize Vertex AI with project and credentials."""
    if not _vertex_ai_available:
        raise RuntimeError("Vertex AI SDK not available")
        
    try:
        vertexai.init(
            project=GCP_PROJECT,
            location=GCP_REGION,
            credentials=load_credentials()
        )
        print(f"Vertex AI Initialized Successfully. Project: {GCP_PROJECT}, Region: {GCP_REGION}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Vertex AI: {e}")

def load_jsonl(file_path):
    """Load data from a JSON Lines file."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {e}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        raise
    return data

def save_jsonl(data, file_path):
    """Save data to a JSON Lines file."""
    # Get the directory path
    directory = os.path.dirname(file_path)
    
    # Only try to create directory if there is a directory path
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Save the file
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def call_vertex_ai(prompt, model_name, temperature=0.7, max_tokens=2048, top_p=0.95):
    """Make a single call to Vertex AI API."""
    if not _vertex_ai_available:
        raise RuntimeError("Vertex AI SDK not available")

    # Configure generation parameters
    generation_config = VertexGenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )

    # Define safety settings
    safety_settings = {
        VertexHarmCategory.HARM_CATEGORY_HATE_SPEECH: VertexHarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        VertexHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: VertexHarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        VertexHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: VertexHarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        VertexHarmCategory.HARM_CATEGORY_HARASSMENT: VertexHarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    # Create model client and generate response
    model = GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Handle response
    if not response.candidates:
        error_msg = "Model response blocked or empty"
        if hasattr(response, 'prompt_feedback'):
            error_msg = f"Prompt Feedback Block Reason: {response.prompt_feedback.block_reason}"
        return None, error_msg, None

    # Extract token usage
    token_usage = None
    if hasattr(response, 'usage_metadata'):
        token_usage = {
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        }

    # Extract generated content
    try:
        content = response.candidates[0].content.parts[0].text
        return None, content, token_usage
    except (IndexError, AttributeError) as e:
        return None, f"Failed to extract content: {e}", token_usage

def process_prompts(prompt_list, model_name, data_list, output_path, start_index=0, limit=None):
    """Process prompts sequentially and save results."""
    # Initialize Vertex AI
    initialize_vertex_ai()

    # Apply limit if specified
    if limit is not None:
        prompt_list = prompt_list[:limit]
        data_list = data_list[:limit]

    # Skip to start_index
    prompt_list = prompt_list[start_index:]
    data_list = data_list[start_index:]

    results = []
    for i, (prompt, data) in enumerate(tqdm(zip(prompt_list, data_list), total=len(prompt_list), desc="Processing prompts")):
        try:
            # Call Vertex AI
            reasoning, content, token_usage = call_vertex_ai(prompt, model_name)
            
            # Prepare result
            result = data.copy()
            result["response"] = content
            result["reasoning_content"] = reasoning if reasoning else ""
            result["token_usage"] = token_usage if token_usage else {}
            results.append(result)

            # Save incrementally
            save_jsonl(results, output_path)

        except Exception as e:
            # Print full error information including traceback
            print(f"\nError processing prompt {i + start_index}:")
            print(f"Prompt: {prompt[:200]}...")  # Show first 200 chars of prompt
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            print()  # Add blank line for readability
            
            # Add error result
            result = data.copy()
            result["response"] = f"Error: {str(e)}"
            result["reasoning_content"] = ""
            result["token_usage"] = {}
            results.append(result)
            save_jsonl(results, output_path)

def main():
    parser = argparse.ArgumentParser(description="Send prompts to Vertex AI Gemini models.")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to input JSONL file with prompts")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-pro-preview-03-25", help="Vertex AI model ID")
    parser.add_argument("--start_index", type=int, default=0, help="Index to start processing from")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts to process")
    args = parser.parse_args()

    if not _vertex_ai_available:
        print("Error: Vertex AI SDK not available. Please install required packages.")
        exit(1)

    try:
        # Load data
        data_list = load_jsonl(args.prompt_path)
        if not data_list:
            print("Error: No valid data found in input file")
            exit(1)

        # Extract prompts
        prompts = []
        valid_data = []
        for item in data_list:
            if isinstance(item, dict) and 'prompt' in item:
                prompts.append(item['prompt'])
                valid_data.append(item)

        if not prompts:
            print("Error: No valid prompts found")
            exit(1)

        # Process prompts
        start_time = time.time()
        process_prompts(
            prompts,
            args.model_name,
            valid_data,
            args.output_path,
            args.start_index,
            args.limit
        )
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main() 