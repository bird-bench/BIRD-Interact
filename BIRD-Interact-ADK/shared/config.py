"""Centralized configuration.

Settings are loaded in this priority (highest wins):
  1. Environment variables (e.g. PATIENCE=6 python -m orchestrator.runner ...)
  2. .env file in project root (user-specific, gitignored)
  3. Defaults defined below

Users: copy .env.example to .env and edit.
See .env.example for all available settings.
"""

from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env into os.environ so litellm/openai can read OPENAI_API_KEY etc.
load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # LLM provider
    llm_provider: str = "litellm"

    # PostgreSQL
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_user: str = "root"
    pg_password: str = "123123"
    pg_minconn: int = 1
    pg_maxconn: int = 5

    # Service ports
    system_agent_port: int = 6000
    user_sim_port: int = 6001
    db_env_port: int = 6002

    # Models (LiteLlm format: provider/model-name)
    user_sim_model: str = "anthropic/claude-haiku-4-5-20251001"
    system_agent_model: str = "anthropic/claude-sonnet-4-20250514"

    # LiteLlm proxy (optional — set if using a LiteLlm proxy server)
    litellm_api_base: str = ""
    litellm_api_key: str = ""

    # Dataset: "lite" or "full"
    dataset: str = "lite"

    # User simulator prompt version: "v1" (legacy) or "v2" (recommended)
    prompt_version: str = "v2"

    # Budget / turns
    patience: int = 3

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / f"bird-interact-{self.dataset}"

    @property
    def data_path(self) -> str:
        return str(self.data_dir / "bird_interact_data.jsonl")

    @property
    def db_data_path(self) -> str:
        return str(self.data_dir)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
