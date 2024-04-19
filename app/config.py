from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/docmind"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    chunk_size: int = 512
    chunk_overlap: int = 64

    max_upload_size_mb: int = 50

    api_key: str = ""
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


settings = Settings()
