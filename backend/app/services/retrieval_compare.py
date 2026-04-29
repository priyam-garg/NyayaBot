"""
NLP method comparison for legal clause retrieval.

Three retrieval approaches run on the same Qdrant corpus:

  1. BM25 (Okapi BM25)      — purely lexical, term-frequency ranking
  2. Word2Vec averaged       — word-level embeddings (pre-attention / RNN era)
     Trained in-process on the ingested legal corpus (no external download).
  3. MiniLM Transformer      — dense semantic retrieval (current system)

Corpus is fetched from Qdrant once and cached in module-level state.
Word2Vec training happens during first load (~seconds on a small legal corpus).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from gensim.models import Word2Vec

from app.services.embeddings import embed_query
from app.services.qdrant_client import get_qdrant, get_settings

log = logging.getLogger("nyayabot.compare")

# ── Module-level corpus cache (populated on first compare call) ────────────
_corpus: "_Corpus | None" = None


@dataclass
class CompareHit:
    text: str
    source: str
    chunk_index: int
    score: float
    section_number: str = ""
    section_title: str = ""


@dataclass
class MethodResult:
    method: str
    label: str
    description: str
    hits: list[CompareHit]
    latency_ms: int


@dataclass
class _Corpus:
    texts: list[str]
    sources: list[str]
    chunk_indices: list[int]
    section_numbers: list[str]
    section_titles: list[str]
    tokenized: list[list[str]]
    bm25: BM25Okapi
    tfidf: TfidfVectorizer
    tfidf_matrix: object       # sklearn sparse matrix
    word2vec: Word2Vec


# ── Corpus loader ──────────────────────────────────────────────────────────

def _load_corpus() -> _Corpus:
    global _corpus
    if _corpus is not None:
        return _corpus

    log.info("[Compare] Fetching corpus from Qdrant for comparison indexes…")
    t0 = time.perf_counter()

    client = get_qdrant()
    s = get_settings()

    texts, sources, chunk_idxs, sec_nums, sec_titles = [], [], [], [], []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=s.qdrant_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in result:
            p = pt.payload or {}
            text = p.get("text", "")
            if text:
                texts.append(text)
                sources.append(p.get("source", ""))
                chunk_idxs.append(int(p.get("chunk_index", 0)))
                sec_nums.append(p.get("section_number", ""))
                sec_titles.append(p.get("section_title", ""))
        if next_offset is None:
            break
        offset = next_offset

    log.info("[Compare] Loaded %d chunks from Qdrant", len(texts))

    if not texts:
        raise RuntimeError("Corpus is empty — ingest documents first.")

    # ── BM25 ────────────────────────────────────────────────────────────────
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # ── TF-IDF (used internally for Word2Vec corpus quality; not exposed) ──
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(texts)

    # ── Word2Vec — trained on corpus (RNN-era word embedding approach) ─────
    log.info("[Compare] Training Word2Vec on %d documents…", len(tokenized))
    w2v = Word2Vec(
        sentences=tokenized,
        vector_size=100,
        window=5,
        min_count=1,
        workers=2,
        epochs=10,
        seed=42,
    )
    log.info("[Compare] Word2Vec trained | vocab=%d", len(w2v.wv))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    log.info("[Compare] Corpus indexed in %d ms", elapsed_ms)

    _corpus = _Corpus(
        texts=texts,
        sources=sources,
        chunk_indices=chunk_idxs,
        section_numbers=sec_nums,
        section_titles=sec_titles,
        tokenized=tokenized,
        bm25=bm25,
        tfidf=tfidf,
        tfidf_matrix=tfidf_matrix,
        word2vec=w2v,
    )
    return _corpus


def _make_hits(corpus: _Corpus, idxs: np.ndarray, scores: np.ndarray, top_k: int) -> list[CompareHit]:
    hits = []
    for idx in idxs[:top_k]:
        sc = float(scores[idx])
        if sc <= 0:
            continue
        hits.append(CompareHit(
            text=corpus.texts[idx][:500],
            source=corpus.sources[idx],
            chunk_index=corpus.chunk_indices[idx],
            score=round(sc, 4),
            section_number=corpus.section_numbers[idx],
            section_title=corpus.section_titles[idx],
        ))
    return hits


# ── Method 1: BM25 ─────────────────────────────────────────────────────────

def run_bm25(query: str, top_k: int = 5) -> MethodResult:
    t0 = time.perf_counter()
    corpus = _load_corpus()
    qtoks = query.lower().split()
    scores = np.array(corpus.bm25.get_scores(qtoks))
    ranked = np.argsort(scores)[::-1]
    hits = _make_hits(corpus, ranked, scores, top_k)
    return MethodResult(
        method="bm25",
        label="BM25 (Classic NLP)",
        description=(
            "Okapi BM25 — probabilistic term-frequency ranking. "
            "Purely lexical: no word meaning, only occurrence frequency and document length normalisation."
        ),
        hits=hits,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ── Method 2: Word2Vec averaged embeddings ─────────────────────────────────

def _avg_vec(tokens: list[str], model: Word2Vec) -> np.ndarray | None:
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def run_word2vec(query: str, top_k: int = 5) -> MethodResult:
    """
    Averaged Word2Vec embeddings — pre-attention / RNN-era approach.

    Each chunk is represented as the unweighted average of its token vectors,
    trained on the legal corpus itself. Query is embedded the same way.
    Ranking is cosine similarity of averaged vectors.
    """
    t0 = time.perf_counter()
    corpus = _load_corpus()
    qtoks = query.lower().split()
    q_vec = _avg_vec(qtoks, corpus.word2vec)

    if q_vec is None:
        return MethodResult(
            method="word2vec",
            label="Word2Vec (RNN-era)",
            description="Averaged Word2Vec embeddings. No results — query tokens not in vocabulary.",
            hits=[],
            latency_ms=0,
        )

    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    scores = np.zeros(len(corpus.texts))
    for i, tokens in enumerate(corpus.tokenized):
        dv = _avg_vec(tokens, corpus.word2vec)
        if dv is not None:
            d_norm = dv / (np.linalg.norm(dv) + 1e-9)
            scores[i] = float(np.dot(q_norm, d_norm))

    ranked = np.argsort(scores)[::-1]
    hits = _make_hits(corpus, ranked, scores, top_k)
    return MethodResult(
        method="word2vec",
        label="Word2Vec (RNN-era)",
        description=(
            "Averaged Word2Vec embeddings trained on the legal corpus. "
            "Captures basic word semantics without contextual attention — "
            "intermediate between lexical (BM25) and contextual (Transformer) retrieval."
        ),
        hits=hits,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ── Method 3: MiniLM Transformer ──────────────────────────────────────────

def run_transformer(query: str, top_k: int = 5) -> MethodResult:
    """Dense semantic retrieval via paraphrase-multilingual-MiniLM-L12-v2."""
    t0 = time.perf_counter()
    from app.services.qdrant_client import search
    vector = embed_query(query)
    raw_hits = search(vector, limit=top_k)
    hits = [
        CompareHit(
            text=(h.payload or {}).get("text", "")[:500],
            source=(h.payload or {}).get("source", ""),
            chunk_index=int((h.payload or {}).get("chunk_index", 0)),
            score=round(float(h.score), 4),
            section_number=(h.payload or {}).get("section_number", ""),
            section_title=(h.payload or {}).get("section_title", ""),
        )
        for h in raw_hits
    ]
    return MethodResult(
        method="transformer",
        label="MiniLM Transformer (Semantic)",
        description=(
            "paraphrase-multilingual-MiniLM-L12-v2 bi-encoder with Qdrant dense vector search. "
            "Self-attention captures deep contextual meaning — best semantic understanding of the three methods."
        ),
        hits=hits,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


# ── Public entry point ─────────────────────────────────────────────────────

def compare_all(query: str, top_k: int = 5) -> list[MethodResult]:
    """Run BM25, Word2Vec, and MiniLM Transformer in sequence and return all results."""
    return [
        run_bm25(query, top_k),
        run_word2vec(query, top_k),
        run_transformer(query, top_k),
    ]


def invalidate_corpus_cache() -> None:
    """Call after re-ingestion so the comparison corpus is refreshed."""
    global _corpus
    _corpus = None
    log.info("[Compare] Corpus cache invalidated")
