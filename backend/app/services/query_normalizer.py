"""
Query normalization — classical NLP preprocessing step.

Applies before any neural embedding or retrieval:
  1. Unicode normalization
  2. Hinglish token mapping
  3. Legal abbreviation expansion
  4. Lowercase + punctuation stripping
  5. Tokenization + stopword removal
  6. Approximate lemmatization via suffix stripping

No external model downloads required — pure Python + regex.
"""
from __future__ import annotations

import re
import unicodedata

# ── Stopwords (English, pruned for legal query relevance) ──────────────────
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "ought",
    "of", "in", "to", "for", "on", "at", "by", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under", "again",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "own", "same", "than", "too", "very", "just",
    "that", "this", "these", "those", "it", "its",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she",
    "they", "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "all", "each", "few", "more", "most", "other",
    "some", "such", "no", "any", "if",
}

# ── Suffix → replacement (approximate lemmatization) ──────────────────────
_SUFFIX_RULES: list[tuple[str, str]] = [
    (r"ations$", "ate"),
    (r"ation$", "ate"),
    (r"nesses$", ""),
    (r"ness$", ""),
    (r"ments$", ""),
    (r"ment$", ""),
    (r"ings$", ""),
    (r"ing$", ""),
    (r"ies$", "y"),
    (r"ied$", "y"),
    (r"ers$", "er"),
    (r"ed$", ""),
    (r"s$", ""),
]

# ── Legal abbreviations → full form ───────────────────────────────────────
_LEGAL_ABBREV: dict[str, str] = {
    r"\brti\b": "right to information",
    r"\bcpa\b": "consumer protection act",
    r"\bipc\b": "indian penal code",
    r"\bcpc\b": "code of civil procedure",
    r"\bcrpc\b": "code of criminal procedure",
    r"\bit act\b": "information technology act",
    r"\bmva\b": "motor vehicles act",
    r"\bpio\b": "public information officer",
    r"\bcic\b": "central information commission",
    r"\bsic\b": "state information commission",
    r"\bfir\b": "first information report",
    r"\bngo\b": "non governmental organisation",
    r"\bsc\b": "supreme court",
    r"\bhc\b": "high court",
}

# ── Hinglish → English token map ──────────────────────────────────────────
_HINGLISH: dict[str, str] = {
    "kya": "what",
    "kaise": "how",
    "kab": "when",
    "kahan": "where",
    "mera": "my",
    "meri": "my",
    "mere": "my",
    "karna": "do",
    "karo": "do",
    "chahiye": "should",
    "hoga": "will be",
    "hain": "is",
    "hai": "is",
    "nahi": "not",
    "nahin": "not",
    "aur": "and",
    "ya": "or",
    "ke": "of",
    "ka": "of",
    "ki": "of",
    "ko": "to",
    "se": "from",
    "mujhe": "me",
    "humko": "us",
    "unko": "them",
    "adhikar": "rights",
    "kanoon": "law",
    "shikayat": "complaint",
    "nyay": "justice",
    "kourt": "court",
}


def _unicode_norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _lemmatize(word: str) -> str:
    if len(word) <= 3:
        return word
    for pattern, repl in _SUFFIX_RULES:
        new = re.sub(pattern, repl, word)
        if new != word and len(new) >= 3:
            return new
    return word


def normalize_query(query: str) -> tuple[str, str]:
    """
    Normalize a legal query through classical NLP preprocessing.

    Returns:
      normalized_query — lemmatized token string (for BM25 / display)
      expanded_query   — abbreviation-expanded form (for neural embedding)
    """
    # Step 1: unicode
    text = _unicode_norm(query)

    # Step 2: Hinglish token replacement (word-level)
    tokens_raw = text.split()
    tokens_raw = [_HINGLISH.get(t.lower(), t) for t in tokens_raw]
    text = " ".join(tokens_raw)

    # Step 3: legal abbreviation expansion → keep as expanded_query
    expanded = text
    for pattern, replacement in _LEGAL_ABBREV.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)

    # Step 4–7: lowercase, tokenize, stopword-remove, lemmatize → normalized_query
    clean = re.sub(r"[^\w\s]", " ", expanded.lower())
    tokens = [t for t in clean.split() if t not in _STOPWORDS and len(t) > 1]
    lemmatized = [_lemmatize(t) for t in tokens]

    return " ".join(lemmatized), expanded
