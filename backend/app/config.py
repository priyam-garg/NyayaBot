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

    user_docs_collection: str = "user_docs"
    max_upload_bytes: int = 10 * 1024 * 1024
    upload_chunk_size: int = 1500
    upload_chunk_overlap: int = 200

    # HyDE: generate a hypothetical answer and embed it instead of the raw query
    hyde_enabled: bool = True
    # Cross-encoder reranking: fetch retrieval_top_k, rerank to top_k
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_top_k: int = 20

    frontend_origin: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
