import logging
import time
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.services.embeddings import embed_query
from app.services.intent_classifier import classify_intent, IntentResult
from app.services.query_normalizer import normalize_query
from app.services.span_extractor import extract_span
from app.services.qdrant_client import search, search_user_docs

log = logging.getLogger("nyayabot.rag")

# ── Prompts ────────────────────────────────────────────────────────────────

REFUSAL_MESSAGE = (
    "That question appears to be outside the scope of the legal documents "
    "I have access to. Please ask something related to the ingested materials."
)

_FALLBACK_SYSTEM_PROMPT = (
    "You are NyayaBot, a helpful legal assistant specialising in Indian law. "
    "Answer the user's question thoroughly and clearly based on your knowledge of Indian law. "
    "Structure your answer with: a direct explanation, key provisions or points (as a bulleted list if there are multiple), "
    "and any relevant time limits, penalties, or procedures. "
    "Use plain language so a non-lawyer can understand. "
    "Do not mention that you are using general knowledge or that documents were not found.\n\n"
    "After your answer, add exactly this block on a new line:\n"
    "FOLLOW_UPS:\n"
    "1. <first follow-up question>\n"
    "2. <second follow-up question>\n"
    "3. <third follow-up question>\n"
    "The follow-up questions must be natural next questions a citizen would ask."
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

_HYDE_PROMPT = (
    "You are an excerpt from an Indian legal textbook or official government document. "
    "Write 2–3 sentences that directly answer the following legal question. "
    "Write as factual text only — no preamble, no 'I', no meta-commentary. "
    "Focus on provisions, sections, time limits, and procedures as they appear in Indian law.\n\n"
    "Question: {query}"
)


# ── Singletons ─────────────────────────────────────────────────────────────

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
    model = get_settings().reranker_model
    log.info("Loading cross-encoder: %s", model)
    return CrossEncoder(model)


# ── HyDE ──────────────────────────────────────────────────────────────────

def _hypothetical_document(query: str) -> str:
    """
    Generate a short hypothetical passage that *would* answer the query,
    then embed that passage instead of the raw question (HyDE technique).

    The hypothetical answer lives in the same semantic space as real document
    chunks, whereas a raw question and its answer embed very differently.
    Reference: Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without
    Relevance Labels."
    """
    prompt = _HYDE_PROMPT.format(query=query)
    hypothetical = get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
    log.info("HyDE hypothetical: %r", hypothetical[:120])
    return hypothetical


# ── Cross-encoder reranking ────────────────────────────────────────────────

def _rerank(query: str, hits: list, top_n: int) -> list:
    """
    Re-score candidates using a cross-encoder that jointly attends to
    (query, passage). Slower than the bi-encoder but significantly more
    accurate. Applied only to the top retrieved candidates, not the full corpus.
    """
    if not hits:
        return hits
    ce = get_cross_encoder()
    pairs = [(query, h.payload.get("text", "")) for h in hits]
    scores = ce.predict(pairs)

    scored = sorted(zip(scores.tolist(), hits), key=lambda x: x[0], reverse=True)
    reranked = []
    for ce_score, h in scored[:top_n]:
        h.score = float(ce_score)
        reranked.append(h)

    ce_scores = [round(float(s), 3) for s, _ in scored]
    log.info("Cross-encoder reranked %d→%d | scores: %s", len(hits), top_n, ce_scores[:top_n])
    return reranked


# ── Context + source helpers ───────────────────────────────────────────────

def _format_context(hits) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        payload = h.payload or {}
        src = payload.get("source", "unknown")
        sec = payload.get("section_number", "")
        text = payload.get("text", "")
        header = f"[{i}] Source: {src}" + (f" | Section {sec}" if sec else "")
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def _dedupe_sources(hits) -> list[dict]:
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
                "section_number": payload.get("section_number", ""),
                "section_title": payload.get("section_title", ""),
            }
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def _merge_hits(legal_hits: list, user_hits: list, limit: int) -> list:
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


# ── Gemini fallback ────────────────────────────────────────────────────────

def _fallback_answer(query: str) -> tuple[str, list[str]]:
    log.info("RAG fallback: answering %r directly via Gemini", query[:80])
    messages = [
        SystemMessage(content=_FALLBACK_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    try:
        raw = get_llm().invoke(messages).content
    except Exception as exc:
        if "429" in str(exc) or "Resource exhausted" in str(exc) or "ResourceExhausted" in type(exc).__name__:
            raise RuntimeError(
                "Gemini API rate limit reached (429). Please wait a minute and try again."
            ) from exc
        raise
    return _parse_follow_ups(raw)


# ── Main RAG entry point ───────────────────────────────────────────────────

def run_rag(
    query: str,
    doc_id: str | None = None,
) -> tuple[str, bool, float | None, list[dict], list[str], str | None, str | None, str | None, str | None]:
    """
    Full pipeline:
      0. Classical NLP: query normalization (tokenize, stopword, lemmatize)
      1. Intent classification (keyword-based domain detection)
      2. Abbreviation expansion
      3. HyDE — embed hypothetical answer
      4. Qdrant dense search (retrieval_top_k candidates)
      5. Merge user_doc hits
      6. Cross-encoder reranking
      7. Similarity threshold refusal gate
      8. Exact span extraction (sub-chunk sentence matching)
      9. Gemini answer generation

    Returns:
      (answer, refused, top_score, sources, follow_ups,
       intent_domain, intent_label, normalized_query, top_span)
    """
    t0 = time.perf_counter()
    print("\n" + "~" * 72)
    print("[RAG FLOW] Start")
    print("~" * 72)
    print(f"[RAG FLOW] Query : {query[:180]}{'...' if len(query) > 180 else ''}")
    print(f"[RAG FLOW] DocID : {doc_id or 'none'}")

    s = get_settings()

    # Step 0: Classical NLP preprocessing
    normalized_query, expanded = normalize_query(query)
    print(f"[RAG FLOW] Normalized : {normalized_query[:120]}")
    print(f"[RAG FLOW] Expanded   : {expanded[:120]}")

    # Step 1: Intent classification
    intent: IntentResult = classify_intent(query)
    print(f"[RAG FLOW] Intent: {intent.label} (confidence={intent.confidence})")

    # Step 2: HyDE
    if s.hyde_enabled:
        print("[RAG FLOW] HyDE: generating hypothetical document…")
        try:
            hyde_text = _hypothetical_document(expanded)
            vector = embed_query(hyde_text)
            print(f"[RAG FLOW] HyDE embedding ready (chars={len(hyde_text)})")
        except Exception as exc:
            log.warning("HyDE failed (%s); falling back to direct embedding", exc)
            print(f"[RAG FLOW] HyDE failed ({exc}); using direct query embedding")
            vector = embed_query(expanded)
    else:
        print("[RAG FLOW] HyDE disabled: embedding expanded query")
        vector = embed_query(expanded)

    # Step 3: Retrieve
    print(f"[RAG FLOW] Searching legal_docs (top={s.retrieval_top_k})")
    legal_hits = search(vector, limit=s.retrieval_top_k)
    print(f"[RAG FLOW] legal_docs hits: {len(legal_hits)}")

    user_hits = []
    if doc_id:
        try:
            print(f"[RAG FLOW] Searching user_docs (doc_id={doc_id})")
            user_hits = search_user_docs(vector, doc_id=doc_id, limit=s.retrieval_top_k)
            print(f"[RAG FLOW] user_docs hits: {len(user_hits)}")
        except RuntimeError:
            log.warning("user_docs search failed for doc_id=%s", doc_id)

    # Step 4: Merge + rerank
    candidates = _merge_hits(legal_hits, user_hits, limit=s.retrieval_top_k * 2)
    print(f"[RAG FLOW] Merged candidates: {len(candidates)}")

    if not candidates:
        log.warning("No candidates — Gemini fallback for query=%r", query)
        print("[RAG FLOW] No candidates → Gemini fallback")
        answer, follow_ups = _fallback_answer(query)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[RAG FLOW] End ({elapsed_ms} ms)")
        print("~" * 72 + "\n")
        return answer, False, None, [], follow_ups, intent.domain, intent.label, normalized_query, None

    print(f"[RAG FLOW] Reranking to top {s.top_k}")
    hits = _rerank(query, candidates, top_n=s.top_k)

    top_score = float(hits[0].score)
    print(f"[RAG FLOW] top_score (reranked): {top_score:.3f}")
    log.info("RAG query=%r top_score=%.3f threshold=%.2f intent=%s",
             query, top_score, s.similarity_threshold, intent.domain)

    # Step 5: Refusal gate
    if top_score < -2.0:
        log.info("Score %.3f < -2.0 → Gemini fallback", top_score)
        print("[RAG FLOW] Score below floor → Gemini fallback")
        answer, follow_ups = _fallback_answer(query)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[RAG FLOW] End ({elapsed_ms} ms)")
        print("~" * 72 + "\n")
        return answer, False, top_score, [], follow_ups, intent.domain, intent.label, normalized_query, None

    # Step 6: Exact span extraction from top chunk
    top_span: str | None = None
    try:
        top_chunk_text = (hits[0].payload or {}).get("text", "")
        span_result = extract_span(query, top_chunk_text)
        if span_result:
            top_span = span_result.text
            print(f"[RAG FLOW] Top span (score={span_result.score}): {top_span[:100]}")
    except Exception as exc:
        log.warning("Span extraction failed: %s", exc)

    # Step 7: Generate answer
    print("[RAG FLOW] Generating answer with Gemini…")
    context = _format_context(hits)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    try:
        raw = get_llm().invoke(messages).content
    except Exception as exc:
        if "429" in str(exc) or "Resource exhausted" in str(exc) or "ResourceExhausted" in type(exc).__name__:
            raise RuntimeError(
                "Gemini API rate limit reached (429). Please wait a minute and try again, "
                "or set HYDE_ENABLED=false in your .env to reduce API calls."
            ) from exc
        raise
    answer, follow_ups = _parse_follow_ups(raw)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[RAG FLOW] Answer ready | chars={len(answer)} | follow_ups={len(follow_ups)} | span={'yes' if top_span else 'no'}")
    print(f"[RAG FLOW] End ({elapsed_ms} ms)")
    print("~" * 72 + "\n")

    return (
        answer, False, top_score, _dedupe_sources(hits), follow_ups,
        intent.domain, intent.label, normalized_query, top_span,
    )
