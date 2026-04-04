"""Settings management using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find project root (where .env lives) and load it
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

# Load .env into environment (pydantic-settings env_file has issues)
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE, override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    # Logfire
    logfire_token: str | None = None
    logfire_service_name: str = "sanjaya"

    # OpenAI (used by pydantic-ai)
    openai_api_key: str | None = None

    # Anthropic (used by pydantic-ai)
    anthropic_api_key: str | None = None

    # OpenRouter (primary provider)
    openrouter_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache (useful after .env changes)."""
    get_settings.cache_clear()
