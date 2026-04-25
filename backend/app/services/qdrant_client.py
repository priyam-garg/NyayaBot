import logging
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException
from app.config import get_settings
from app.services.embeddings import vector_size

log = logging.getLogger("nyayabot.qdrant")

_RETRY_DELAYS = [5, 10, 20]


def get_qdrant() -> QdrantClient:
    s = get_settings()
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
        vectors_config=qmodels.VectorParams(size=vector_size(), distance=qmodels.Distance.COSINE),
    )


def ensure_user_docs_collection() -> None:
    """Called at startup — creates user_docs collection if absent."""
    client = get_qdrant()
    name = get_settings().user_docs_collection
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=vector_size(), distance=qmodels.Distance.COSINE),
    )


def search(query_vector: list[float], limit: int):
    """Search legal_docs with automatic retry on Qdrant Cloud hibernation wake-up."""
    last_exc = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, 1):
        if delay:
            log.warning("Qdrant timeout — waking cluster, retry %d/%d in %ds…", attempt - 1, len(_RETRY_DELAYS), delay)
            time.sleep(delay)
        try:
            client = get_qdrant()
            return client.search(
                collection_name=get_settings().qdrant_collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        except (ResponseHandlingException, Exception) as exc:
            if _is_timeout(exc):
                last_exc = exc
                continue
            raise

    log.error("Qdrant unreachable after %d retries", len(_RETRY_DELAYS))
    raise RuntimeError(
        "The vector database is waking up from hibernation. Please try again in ~30 seconds."
    ) from last_exc


def search_user_docs(query_vector: list[float], doc_id: str, limit: int):
    """Filtered search in user_docs restricted to a single doc_id."""
    last_exc = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, 1):
        if delay:
            log.warning("Qdrant timeout (user_docs) — retry %d/%d in %ds…", attempt - 1, len(_RETRY_DELAYS), delay)
            time.sleep(delay)
        try:
            client = get_qdrant()
            return client.search(
                collection_name=get_settings().user_docs_collection,
                query_vector=query_vector,
                query_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
                ),
                limit=limit,
                with_payload=True,
            )
        except (ResponseHandlingException, Exception) as exc:
            if _is_timeout(exc):
                last_exc = exc
                continue
            raise

    log.error("Qdrant user_docs unreachable after %d retries", len(_RETRY_DELAYS))
    raise RuntimeError(
        "The vector database is waking up from hibernation. Please try again in ~30 seconds."
    ) from last_exc


def upsert_user_doc_chunks(points: list[qmodels.PointStruct]) -> None:
    """Upsert a batch of chunks into user_docs. Synchronous — call in thread."""
    get_qdrant().upsert(
        collection_name=get_settings().user_docs_collection,
        points=points,
    )


def delete_user_doc(doc_id: str) -> None:
    """Delete all vectors for a doc_id from user_docs."""
    get_qdrant().delete(
        collection_name=get_settings().user_docs_collection,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
            )
        ),
    )
