from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Settings
    openai_api_key: str = "sk-..."
    openai_model: str = "gpt-3.5-turbo"

    # App Settings
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # Vector DB Settings (추후 사용)
    # vector_db_path: str = "chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 정의되지 않은 환경변수는 무시
    )


@lru_cache
def get_settings():
    return Settings()
