from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from app.config import get_settings
from app.services.embeddings import vector_size


@lru_cache
def get_qdrant() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30.0)


def ensure_collection() -> None:
    client = get_qdrant()
    name = get_settings().qdrant_collection
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=vector_size(),
            distance=qmodels.Distance.COSINE,
        ),
    )


def search(query_vector: list[float], limit: int):
    client = get_qdrant()
    return client.search(
        collection_name=get_settings().qdrant_collection,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )
