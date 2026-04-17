from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from functools import lru_cache
from app.config import get_settings
from app.services.embeddings import embed_query
from app.services.qdrant_client import search

REFUSAL_MESSAGE = (
    "That question appears to be outside the scope of the legal documents "
    "I have access to. Please ask something related to the ingested materials."
)

SYSTEM_PROMPT = (
    "You are NyayaBot, a careful legal assistant. "
    "Answer strictly from the provided context. "
    "If the context does not contain the answer, say you don't know based on the available documents. "
    "Do not invent statutes, case names, or citations."
)


@lru_cache
def get_llm() -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=s.gemini_api_key,
        temperature=0.2,
    )


def _format_context(hits) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        payload = h.payload or {}
        src = payload.get("source", "unknown")
        text = payload.get("text", "")
        blocks.append(f"[{i}] Source: {src}\n{text}")
    return "\n\n".join(blocks)


def _dedupe_sources(hits) -> list[dict]:
    """Collapse hits to one entry per source filename, keeping the best score."""
    best: dict[str, dict] = {}
    for h in hits:
        payload = h.payload or {}
        src = payload.get("source", "unknown")
        score = float(h.score)
        if src not in best or score > best[src]["score"]:
            best[src] = {
                "source": src,
                "score": score,
                "chunk_index": int(payload.get("chunk_index", 0)),
            }
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def run_rag(query: str) -> tuple[str, bool, float | None, list[dict]]:
    """Returns (answer, refused, top_score, sources)."""
    s = get_settings()
    vector = embed_query(query)
    hits = search(vector, limit=s.top_k)

    if not hits:
        return REFUSAL_MESSAGE, True, None, []

    top_score = float(hits[0].score)
    if top_score < s.similarity_threshold:
        return REFUSAL_MESSAGE, True, top_score, []

    context = _format_context(hits)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    response = get_llm().invoke(messages)
    return response.content, False, top_score, _dedupe_sources(hits)
