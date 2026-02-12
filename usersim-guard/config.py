"""
Configuration for User Simulator Pipeline.

This file contains all configuration settings for API access and data paths.
Users should modify this file to use their own API keys and endpoints.
"""

import os

# =============================================================================
# API CONFIGURATION
# =============================================================================

# OpenAI-compatible API (for GPT-4o, Gemini, or custom models)
# You can use any OpenAI-compatible endpoint (OpenAI, Azure, local vLLM, etc.)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Anthropic Direct API (for Claude models)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# OpenAI-compatible model mappings (model_name -> api_model_id)
# Add your custom models here
OPENAI_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "gemini-2.0-flash": "gemini-2.0-flash",
    # Add your custom model mappings here
    # "my-local-model": "my-local-model-id",
}


# =============================================================================
# DATA PATHS
# =============================================================================

# Path to the source data file containing full instance information
# This file should have fields: instance_id, amb_user_query, user_query_ambiguity,
# knowledge_ambiguity, sol_sql, external_knowledge, selected_database
SOURCE_DATA_PATH = os.environ.get(
    "SOURCE_DATA_PATH",
    "data/bird_interact_data.jsonl"
)

# Path to the database schemas directory
# Each database should have a subdirectory with {db_name}_schema.txt file
DB_BASE_PATH = os.environ.get(
    "DB_BASE_PATH",
    "data/databases"
)


# =============================================================================
# DEFAULT PATHS (for convenience)
# =============================================================================

DEFAULT_PATHS = {
    "source_data": SOURCE_DATA_PATH,
    "db_base_path": DB_BASE_PATH,
}
