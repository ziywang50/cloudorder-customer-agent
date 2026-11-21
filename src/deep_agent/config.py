"""Settings for the deep-agent path, read from the environment or `.env`.

Kept apart from `src/config.py` so this path runs without the Postgres and Redis
settings the existing paths require. `extra="ignore"` lets both share one `.env`.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepAgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DEEP_AGENT_", extra="ignore")

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEP_AGENT_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    coordinator_model: str = "claude-sonnet-5-5"
    subagent_model: str = "claude-haiku-5-5"

    backend_url: str | None = None
    """Base URL of a CloudOrder backend. Unset: call the mock backend in-process."""
    backend_timeout_s: float = 5.0
    backend_max_attempts: int = 3


@lru_cache
def get_settings() -> DeepAgentSettings:
    return DeepAgentSettings()
