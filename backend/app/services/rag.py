import logging
import time
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.services.embeddings import embed_query
from app.services.qdrant_client import search, search_user_docs

log = logging.getLogger("nyayabot.rag")

# ---------------------------------------------------------------------------
# Abbreviation expansion — improves embedding match for short Indian legal terms
# ---------------------------------------------------------------------------
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
        log.info("Query expanded: %r -> %r", query, q)
    return q


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
REFUSAL_MESSAGE = (
    "That question appears to be outside the scope of the legal documents "
    "I have access to. Please ask something related to the ingested materials."
)

# Used for the final answer generation
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

# Used by HyDE to generate a hypothetical document for embedding
_HYDE_PROMPT = (
    "You are an excerpt from an Indian legal textbook or official government document. "
    "Write 2–3 sentences that directly answer the following legal question. "
    "Write as factual text only — no preamble, no 'I', no meta-commentary. "
    "Focus on provisions, sections, time limits, and procedures as they appear in Indian law.\n\n"
    "Question: {query}"
)


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
@lru_cache
def get_llm() -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=s.gemini_api_key,
        temperature=0.2,
    )


@lru_cache
def get_cross_encoder() -> CrossEncoder:
    """Load cross-encoder model once. ~80 MB download on first use."""
    model = get_settings().reranker_model
    log.info("Loading cross-encoder: %s", model)
    return CrossEncoder(model)


# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embeddings
# ---------------------------------------------------------------------------
def _hypothetical_document(query: str) -> str:
    """
    Ask the LLM to write a short passage that *would* answer the query,
    then return that passage for embedding.

    Why this works: the hypothetical answer lives in the same semantic space
    as real document chunks, whereas a question ("What is the fee?") and an
    answer ("The fee is ₹10 under Section 7(1)") embed very differently.

    Reference: Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without
    Relevance Labels" (HyDE).
    """
    prompt = _HYDE_PROMPT.format(query=query)
    hypothetical = get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
    log.info("HyDE hypothetical: %r", hypothetical[:120])
    return hypothetical


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------
def _rerank(query: str, hits: list, top_n: int) -> list:
    """
    Rerank `hits` using a cross-encoder that jointly attends to (query, passage).

    The bi-encoder (used during retrieval) encodes query and passage independently
    — fast but less accurate. The cross-encoder encodes the concatenation
    [CLS] query [SEP] passage [SEP], letting self-attention see both at once.
    This is significantly more accurate but too slow for full-corpus search,
    so we only apply it to the top retrieved candidates.

    Scores replace the original cosine similarity scores for downstream logging.
    """
    if not hits:
        return hits
    ce = get_cross_encoder()
    pairs = [(query, h.payload.get("text", "")) for h in hits]
    scores = ce.predict(pairs)  # returns numpy array of floats

    # Attach cross-encoder score to each hit (mutate a wrapper, not the original)
    scored = sorted(zip(scores.tolist(), hits), key=lambda x: x[0], reverse=True)
    reranked = []
    for ce_score, h in scored[:top_n]:
        # Wrap hit with overridden score for logging/display purposes
        h.score = float(ce_score)
        reranked.append(h)

    ce_scores = [round(float(s), 3) for s, _ in scored]
    log.info("Cross-encoder reranked %d→%d | scores: %s", len(hits), top_n, ce_scores[:top_n])
    return reranked


# ---------------------------------------------------------------------------
# Context formatting and source deduplication
# ---------------------------------------------------------------------------
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
        origin = "user_doc" if payload.get("doc_id") else "legal"
        if src not in best or score > best[src]["score"]:
            best[src] = {
                "source": payload.get("document_title") or src,
                "score": score,
                "chunk_index": int(payload.get("chunk_index", 0)),
                "origin": origin,
            }
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def _merge_hits(legal_hits: list, user_hits: list, limit: int) -> list:
    """Merge two hit lists by score descending, deduplicate by point id."""
    seen_ids: set = set()
    combined = []
    for h in legal_hits:
        if h.id not in seen_ids:
            seen_ids.add(h.id)
            combined.append(h)
    for h in user_hits:
        if h.id not in seen_ids:
            seen_ids.add(h.id)
            combined.append(h)
    combined.sort(key=lambda h: h.score, reverse=True)
    return combined[:limit]


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


# ---------------------------------------------------------------------------
# Main RAG entry point
# ---------------------------------------------------------------------------
def run_rag(query: str, doc_id: str | None = None) -> tuple[str, bool, float | None, list[dict], list[str]]:
    """
    Full pipeline:
      1. Abbreviation expansion
      2. HyDE — generate hypothetical document, embed it (better retrieval)
      3. Qdrant dense search (retrieval_top_k candidates)
      4. Merge user_docs hits (if session has an uploaded document)
      5. Cross-encoder reranking (candidates → top_k final)
      6. Similarity threshold refusal gate
      7. Gemini answer generation
    """
    t0 = time.perf_counter()
    print("\n" + "~" * 72)
    print("[RAG FLOW] Start")
    print("~" * 72)
    print(f"[RAG FLOW] Query : {query[:180]}{'...' if len(query) > 180 else ''}")
    print(f"[RAG FLOW] DocID : {doc_id or 'none'}")

    s = get_settings()
    expanded = _expand_query(query)
    if expanded != query:
        print(f"[RAG FLOW] Expanded query: {expanded}")

    # Step 2: HyDE — embed hypothetical answer instead of raw question
    if s.hyde_enabled:
        print("[RAG FLOW] HyDE enabled: generating hypothetical document...")
        try:
            hyde_text = _hypothetical_document(expanded)
            vector = embed_query(hyde_text)
            print(f"[RAG FLOW] HyDE embedding ready (chars={len(hyde_text)})")
        except Exception as exc:
            log.warning("HyDE generation failed (%s); falling back to direct query embedding", exc)
            print(f"[RAG FLOW] HyDE failed ({exc}); fallback to direct query embedding")
            vector = embed_query(expanded)
    else:
        print("[RAG FLOW] HyDE disabled: embedding direct query")
        vector = embed_query(expanded)

    # Step 3: retrieve a larger candidate pool for reranking
    print(f"[RAG FLOW] Searching legal_docs (top={s.retrieval_top_k})")
    legal_hits = search(vector, limit=s.retrieval_top_k)
    print(f"[RAG FLOW] legal_docs hits: {len(legal_hits)}")

    user_hits = []
    if doc_id:
        try:
            print(f"[RAG FLOW] Searching user_docs for doc_id={doc_id} (top={s.retrieval_top_k})")
            user_hits = search_user_docs(vector, doc_id=doc_id, limit=s.retrieval_top_k)
            print(f"[RAG FLOW] user_docs hits: {len(user_hits)}")
        except RuntimeError:
            log.warning("user_docs search failed for doc_id=%s; continuing with legal_docs only", doc_id)
            print("[RAG FLOW] user_docs search unavailable; continuing with legal_docs only")

    # Step 4: merge
    candidates = _merge_hits(legal_hits, user_hits, limit=s.retrieval_top_k * 2)
    print(f"[RAG FLOW] merged candidates: {len(candidates)}")

    if not candidates:
        log.warning("RAG: no candidates for query=%r doc_id=%r", query, doc_id)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print("[RAG FLOW] No candidates -> refusal")
        print(f"[RAG FLOW] End ({elapsed_ms} ms)")
        print("~" * 72 + "\n")
        return REFUSAL_MESSAGE, True, None, [], []

    # Step 5: cross-encoder reranking — re-score with full query-passage attention
    print(f"[RAG FLOW] Reranking candidates to top {s.top_k}")
    hits = _rerank(query, candidates, top_n=s.top_k)

    top_score = float(hits[0].score)
    print(f"[RAG FLOW] top_score (reranked): {top_score:.3f}")
    log.info("RAG query=%r top_score_after_rerank=%.3f threshold=%.2f hyde=%s",
             query, top_score, s.similarity_threshold, s.hyde_enabled)

    # Step 6: refusal gate (applied to cross-encoder scores — typically 0–10 range)
    # Cross-encoder scores are not cosine similarities; calibrate threshold accordingly.
    # ms-marco-MiniLM scores: relevant ~3–10, irrelevant ~-5–2.
    # We use a separate rerank_threshold to avoid conflating with cosine similarity.
    rerank_threshold = s.similarity_threshold if not s.hyde_enabled else -1.0
    effective_threshold = rerank_threshold if not doc_id else rerank_threshold * 0.85
    print(f"[RAG FLOW] threshold (effective): {effective_threshold:.3f}")

    # If score is clearly irrelevant (< -2 on ms-marco scale), refuse
    if top_score < -2.0:
        log.info("RAG refused: top cross-encoder score %.3f < -2.0", top_score)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print("[RAG FLOW] Refused: score below hard floor (-2.0)")
        print(f"[RAG FLOW] End ({elapsed_ms} ms)")
        print("~" * 72 + "\n")
        return REFUSAL_MESSAGE, True, top_score, [], []

    # Step 7: generate answer
    print("[RAG FLOW] Generating final answer with Gemini...")
    context = _format_context(hits)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    raw = get_llm().invoke(messages).content
    answer, follow_ups = _parse_follow_ups(raw)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[RAG FLOW] Answer ready | chars={len(answer)} | follow_ups={len(follow_ups)}")
    print(f"[RAG FLOW] End ({elapsed_ms} ms)")
    print("~" * 72 + "\n")
    return answer, False, top_score, _dedupe_sources(hits), follow_ups
