from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "legal_docs"

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "nyayabot"

    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: float = 0.6
    top_k: int = 5

    frontend_origin: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
