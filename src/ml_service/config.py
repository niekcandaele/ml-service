"""Configuration settings for ML Service.

Environment variables are loaded from .env file or system environment.
All settings can be overridden via environment variables with UPPERCASE names.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment-based configuration."""

    google_api_key: str | None = None
    hugging_face_hub_token: str | None = None
    port: int = 8000
    log_level: str = "INFO"
    embedding_model: str = "google/embeddinggemma-300m"
    completion_model: str = "gemini-2.0-flash-001"
    ocr_model: str = "gemini-2.0-flash-001"
    classification_model: str = "gemini-2.0-flash-001"

    model_config = {"env_file": ".env"}


# Global settings instance
settings = Settings()
