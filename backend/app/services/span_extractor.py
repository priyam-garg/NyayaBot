"""
Exact span extraction.

Given a retrieved chunk and the original query, identifies the single sentence
within that chunk that is most semantically similar to the query.

This is sub-chunk attention-style matching: the same bi-encoder already used
for retrieval scores individual sentences rather than the whole chunk.
No new models required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from app.services.embeddings import embed_query, embed_texts


@dataclass
class SpanResult:
    text: str        # best matching sentence
    score: float     # cosine similarity to query (0–1)
    start_char: int  # character offset within the chunk
    end_char: int


# Sentence boundary: period/exclamation/question/Devanagari danda followed by space
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return list of (sentence, start_char, end_char) within text."""
    parts = _SENT_SPLIT.split(text.strip())
    results: list[tuple[str, int, int]] = []
    search_from = 0
    for part in parts:
        part = part.strip()
        if len(part) < 12:
            search_from += len(part) + 1
            continue
        start = text.find(part, search_from)
        if start == -1:
            start = search_from
        end = start + len(part)
        results.append((part, start, end))
        search_from = end
    return results


def extract_span(query: str, chunk_text: str) -> SpanResult | None:
    """
    Return the sentence within chunk_text most similar to query.

    Returns None when the chunk cannot be meaningfully split (< 2 sentences),
    since the full chunk is already the best answer.

    Vectors are L2-normalised by embed_texts, so dot product == cosine similarity.
    """
    sentences = _split_sentences(chunk_text)
    if len(sentences) < 2:
        return None

    q_vec = np.array(embed_query(query))
    sent_texts = [s[0] for s in sentences]
    sent_vecs = [np.array(v) for v in embed_texts(sent_texts)]

    best_score = -1.0
    best_idx = 0
    for i, sv in enumerate(sent_vecs):
        score = float(np.dot(q_vec, sv))
        if score > best_score:
            best_score = score
            best_idx = i

    sent_text, start, end = sentences[best_idx]
    return SpanResult(
        text=sent_text,
        score=round(best_score, 3),
        start_char=start,
        end_char=end,
    )
