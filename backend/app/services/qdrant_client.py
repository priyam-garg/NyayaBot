import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from app.config import get_settings
from app.services.embeddings import vector_size

log = logging.getLogger("nyayabot.qdrant")

def get_qdrant() -> QdrantClient:
    s = get_settings()
    log.info("🔗 Connecting to Qdrant | URL: %s", s.qdrant_url)
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
    collection = get_settings().qdrant_collection
    log.info("🔍 Qdrant is working fine! | Collection: %s | Query dims: %d | Limit: %d", collection, len(query_vector), limit)
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )
    log.info("✅ Qdrant search complete | Hits returned: %d", len(results))
    return results
