import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from functools import lru_cache
from app.config import get_settings
from app.services.embeddings import embed_query
from app.services.qdrant_client import search

log = logging.getLogger("nyayabot.rag")

# Expand common Indian legal abbreviations before embedding so short queries
# match document text (e.g. "RTI" → "Right to Information Act")
_ABBREV = {
    r"\brti\b": "Right to Information Act",
    r"\bcpa\b": "Consumer Protection Act",
    r"\bipc\b": "Indian Penal Code",
    r"\bcpc\b": "Code of Civil Procedure",
    r"\bcrpc\b": "Code of Criminal Procedure",
    r"\bit act\b": "Information Technology Act",
    r"\bmva\b": "Motor Vehicles Act",
    r"\bpio\b": "Public Information Officer",
    r"\bcic\b": "Central Information Commission",
    r"\bsic\b": "State Information Commission",
}


def _expand_query(query: str) -> str:
    import re
    q = query
    for pattern, replacement in _ABBREV.items():
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    if q != query:
        log.info("RAG query expanded: %r -> %r", query, q)
    return q


REFUSAL_MESSAGE = (
    "That question appears to be outside the scope of the legal documents "
    "I have access to. Please ask something related to the ingested materials."
)

SYSTEM_PROMPT = (
    "You are NyayaBot, a helpful legal assistant specialising in Indian law. "
    "Answer the user's question thoroughly and clearly using ONLY the provided context. "
    "Structure your answer with: a direct explanation, key provisions or points (as a bulleted list if there are multiple), "
    "and any relevant time limits, penalties, or procedures mentioned in the context. "
    "Use plain language so a non-lawyer can understand. "
    "If the context does not contain enough information, say so honestly. "
    "Do not invent statutes, case names, or citations not present in the context.\n\n"
    "After your answer, add exactly this block on a new line:\n"
    "FOLLOW_UPS:\n"
    "1. <first follow-up question>\n"
    "2. <second follow-up question>\n"
    "3. <third follow-up question>\n"
    "The follow-up questions must be natural next questions a citizen would ask, "
    "strictly based on what is in the context."
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
                "source": payload.get("document_title") or src,
                "score": score,
                "chunk_index": int(payload.get("chunk_index", 0)),
            }
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def _parse_follow_ups(raw: str) -> tuple[str, list[str]]:
    """Split Gemini response into (answer, follow_up_questions)."""
    import re
    marker = re.search(r"\nFOLLOW_UPS:\s*\n", raw)
    if not marker:
        return raw.strip(), []
    answer = raw[:marker.start()].strip()
    tail = raw[marker.end():]
    questions = []
    for line in tail.splitlines():
        m = re.match(r"^\s*\d+\.\s*(.+)", line)
        if m:
            questions.append(m.group(1).strip())
    return answer, questions[:3]


def run_rag(query: str) -> tuple[str, bool, float | None, list[dict], list[str]]:
    """Returns (answer, refused, top_score, sources, follow_ups)."""
    s = get_settings()
    vector = embed_query(_expand_query(query))
    hits = search(vector, limit=s.top_k)

    if not hits:
        log.warning("RAG: no hits returned from Qdrant for query=%r", query)
        return REFUSAL_MESSAGE, True, None, [], []

    top_score = float(hits[0].score)
    scores = [round(float(h.score), 3) for h in hits]
    log.info("RAG query=%r top_score=%.3f all_scores=%s threshold=%.2f",
             query, top_score, scores, s.similarity_threshold)
    if top_score < s.similarity_threshold:
        return REFUSAL_MESSAGE, True, top_score, [], []

    context = _format_context(hits)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    raw = get_llm().invoke(messages).content
    answer, follow_ups = _parse_follow_ups(raw)
    return answer, False, top_score, _dedupe_sources(hits), follow_ups
