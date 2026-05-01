"""
Passage-Level Sentiment & Tone Detection.

Sentiment  — VADER (nltk.sentiment.vader): positive / negative / neutral
Tone       — lexicon pattern matching: technical / procedural / critical /
              informational / punitive
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from nltk.sentiment.vader import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

# ── Tone keyword patterns ─────────────────────────────────────────────────────
_TONE_PATTERNS: list[tuple[str, list[str]]] = [
    ("punitive", [
        r"\bpenalt", r"\bpunish", r"\bimprison", r"\bfine\b", r"\boffence\b",
        r"\boffense\b", r"\bliable\b", r"\bconvict", r"\bsentenc",
    ]),
    ("procedural", [
        r"\bshall\b", r"\bapplicat", r"\bprescribed\b", r"\bprocedure\b",
        r"\bform\b", r"\bappeal\b", r"\bfile\b", r"\bsubmit\b",
        r"\bcomply\b", r"\bdeadline\b", r"\bdays?\b", r"\bnotice\b",
    ]),
    ("technical", [
        r"\bsection\b", r"\bclause\b", r"\bsub-section\b", r"\bact\b",
        r"\bstatute\b", r"\bregulat", r"\bprovision\b", r"\bjurisdiction\b",
        r"\barticle\b", r"\bschedule\b",
    ]),
    ("critical", [
        r"\bviolat", r"\bbreach\b", r"\bfailure\b", r"\bnegligen",
        r"\bwrongful\b", r"\bunlawful\b", r"\billegal\b", r"\bgrievance\b",
        r"\bdispute\b", r"\bdefault\b",
    ]),
    ("informational", [
        r"\bmeans\b", r"\bdefined?\b", r"\bincludes?\b", r"\brefers?\b",
        r"\bexplain", r"\bdescrib", r"\bpurpose\b", r"\bobject",
    ]),
]


@dataclass
class ToneResult:
    sentiment: str          # "positive" | "negative" | "neutral"
    tone: str               # "technical" | "procedural" | "punitive" | "critical" | "informational"
    compound: float         # VADER compound score  -1.0 to +1.0
    pos: float
    neg: float
    neu: float


def analyze_passage(text: str) -> ToneResult:
    """Classify a single passage for sentiment and legal tone."""
    scores = _sia.polarity_scores(text)

    compound = scores["compound"]
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    lower = text.lower()
    tone_scores: dict[str, int] = {}
    for tone_label, patterns in _TONE_PATTERNS:
        hits = sum(1 for p in patterns if re.search(p, lower))
        if hits:
            tone_scores[tone_label] = hits

    tone = max(tone_scores, key=lambda t: tone_scores[t]) if tone_scores else "informational"

    return ToneResult(
        sentiment=sentiment,
        tone=tone,
        compound=round(compound, 4),
        pos=round(scores["pos"], 4),
        neg=round(scores["neg"], 4),
        neu=round(scores["neu"], 4),
    )
