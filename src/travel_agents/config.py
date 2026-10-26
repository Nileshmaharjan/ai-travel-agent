from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from the environment or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    tavily_api_key: str = ""
    aviationstack_api_key: str = ""
    http_timeout_s: float = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
