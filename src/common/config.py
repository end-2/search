from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI Settings
    openai_api_key: str = "sk-..."
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

    # Qdrant Settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_child_collection: str = "child_chunks"
    qdrant_parent_collection: str = "parent_chunks"

    # Retrieval Settings
    retrieval_top_k: int = 50
    rerank_top_k: int = 5
    hybrid_dense_weight: float = 0.7
    hybrid_sparse_weight: float = 0.3

    # Chunking Settings
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # MySQL Settings
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "rag"
    mysql_password: str = "rag_password"
    mysql_database: str = "rag_service"

    # Global Scheduler Settings
    ingestion_interval_minutes: int = 60
    ingestion_enabled: bool = True

    # Slack Adapter Settings
    slack_enabled: bool = True
    slack_sync_interval_minutes: int = 0  # 0 = use global interval
    slack_bot_token: str = ""
    slack_channel_types: str = "public_channel,private_channel"
    slack_message_limit: int = 200
    slack_include_threads: bool = True

    # Jira Adapter Settings
    jira_enabled: bool = True
    jira_sync_interval_minutes: int = 0  # 0 = use global interval
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_projects: str = ""  # comma-separated project keys, empty = all
    jira_jql_filter: str = ""  # custom JQL filter
    jira_max_results: int = 100

    # App Settings
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # Ingestion Worker Settings
    ingestion_port: int = 8001

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings():
    return Settings()
