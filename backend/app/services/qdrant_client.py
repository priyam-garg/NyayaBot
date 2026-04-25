import logging
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException
from app.config import get_settings
from app.services.embeddings import vector_size

log = logging.getLogger("nyayabot.qdrant")

_RETRY_DELAYS = [5, 10, 20]  # seconds between retries on ConnectTimeout

def get_qdrant() -> QdrantClient:
    s = get_settings()
    log.info("🔗 Connecting to Qdrant | URL: %s", s.qdrant_url)
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60.0)

def _is_timeout(exc: Exception) -> bool:
    msg = str(exc)
    return "ConnectTimeout" in msg or "10060" in msg or "timed out" in msg.lower()

def ensure_collection() -> None:
    """Called at startup — no retries, fails fast so the server still starts."""
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
    """Search with automatic retry on Qdrant Cloud hibernation wake-up."""
    collection = get_settings().qdrant_collection
    log.info("🔍 Qdrant search requested | Collection: %s | Query dims: %d | Limit: %d", collection, len(query_vector), limit)
    
    last_exc = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, 1):
        if delay:
            log.warning(
                "Qdrant timeout — waking cluster, retry %d/%d in %ds…",
                attempt - 1, len(_RETRY_DELAYS), delay,
            )
            time.sleep(delay)
        try:
            client = get_qdrant()
            results = client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
            log.info("✅ Qdrant search complete | Hits returned: %d", len(results))
            return results
        except (ResponseHandlingException, Exception) as exc:
            if _is_timeout(exc):
                last_exc = exc
                continue
            raise

    log.error("Qdrant unreachable after %d retries", len(_RETRY_DELAYS))
    raise RuntimeError(
        "The vector database is waking up from hibernation. Please try again in ~30 seconds."
    ) from last_exc

