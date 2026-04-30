"""
Legal intent classifier.

Classifies a query into one of 8 legal domains using regex pattern matching —
the classical NLP approach: no training data, no neural model, just expert-curated
lexical rules derived from Indian legal domain knowledge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


DOMAINS = [
    "criminal", "civil", "consumer", "rti",
    "constitutional", "labour", "property", "family", "other",
]

DOMAIN_LABELS: dict[str, str] = {
    "criminal":       "Criminal Law",
    "civil":          "Civil Law",
    "consumer":       "Consumer Protection",
    "rti":            "Right to Information",
    "constitutional": "Constitutional Law",
    "labour":         "Labour & Employment",
    "property":       "Property Law",
    "family":         "Family Law",
    "other":          "General Legal",
}

# (domain, [regex patterns]) — order matters only for tie-breaking
_PATTERNS: list[tuple[str, list[str]]] = [
    ("rti", [
        r"\brti\b", r"right to information", r"information officer",
        r"\bpio\b", r"\bcic\b", r"\bsic\b", r"public information",
        r"information commission", r"first appeal", r"second appeal.*information",
        r"information act", r"30 day", r"thirty day",
    ]),
    ("criminal", [
        r"\bipc\b", r"indian penal code", r"\bcrpc\b", r"criminal procedure",
        r"\bfir\b", r"first information report", r"\bbail\b", r"\barrest\b",
        r"cognizable", r"non.cognizable", r"charge.?sheet", r"\bmurder\b",
        r"\btheft\b", r"\bfraud\b", r"\bassault\b", r"\brape\b",
        r"\bcheating\b", r"\bextortion\b", r"criminal complaint",
        r"\bmagistrate\b", r"sessions court", r"anticipatory bail",
        r"warrant", r"summon", r"police station",
    ]),
    ("consumer", [
        r"\bcpa\b", r"consumer protection", r"consumer forum", r"consumer court",
        r"defective product", r"deficiency.*service", r"service deficiency",
        r"unfair trade", r"overcharg", r"\brefund\b", r"\bwarranty\b",
        r"e.commerce", r"online shopping", r"\bseller\b", r"\bmanufacturer\b",
        r"misleading advertisement", r"consumer dispute",
    ]),
    ("constitutional", [
        r"fundamental right", r"article \d+", r"constitution of india",
        r"writ petition", r"habeas corpus", r"\bmandamus\b", r"\bcertiorari\b",
        r"\bpil\b", r"public interest litigation", r"right to equality",
        r"freedom of speech", r"right to life", r"directive principle",
        r"right to education", r"article 21", r"article 14", r"article 19",
    ]),
    ("labour", [
        r"labour law", r"labor law", r"industrial dispute", r"trade union",
        r"minimum wage", r"provident fund", r"\bepf\b", r"\besi\b",
        r"\bgratuity\b", r"maternity benefit", r"wrongful termination",
        r"dismiss.*employ", r"employ.*dismiss", r"\bretrenchment\b",
        r"workmen compensation", r"employee right", r"factory act",
        r"payment of wages", r"bonus act",
    ]),
    ("property", [
        r"property law", r"transfer of property", r"sale deed", r"\btitle\b",
        r"\bpossession\b", r"\bencroachment\b", r"\blandlord\b", r"\btenant\b",
        r"\beviction\b", r"rent control", r"\blease\b", r"\bmortgage\b",
        r"stamp duty", r"property registration", r"benami", r"adverse possession",
        r"rent agreement", r"society.*flat", r"flat.*society",
    ]),
    ("family", [
        r"\bdivorce\b", r"\bmarriage\b", r"\bmatrimonial\b", r"\balimony\b",
        r"\bmaintenance\b", r"child custody", r"\badoption\b", r"\bguardianship\b",
        r"domestic violence", r"hindu marriage", r"muslim marriage",
        r"personal law", r"\bsuccession\b", r"\binheritance\b",
        r"legal heir", r"\bwill\b.*legal", r"family court",
    ]),
    ("civil", [
        r"\bcpc\b", r"civil procedure", r"civil suit", r"civil court",
        r"\binjunction\b", r"specific performance", r"\bdamages\b",
        r"\bcompensation\b", r"\bcontract\b", r"breach of contract",
        r"\bnegligence\b", r"limitation act", r"\bdecree\b",
        r"execution.*decree", r"civil dispute", r"pecuniary jurisdiction",
    ]),
]


@dataclass
class IntentResult:
    domain: str
    label: str
    confidence: float        # 0.0–1.0
    matched_patterns: list[str] = field(default_factory=list)


def classify_intent(query: str) -> IntentResult:
    """
    Classify the query using regex pattern matching (keyword-based NLP).

    Confidence is proportional to how many patterns matched relative to the
    maximum possible matches for that domain (capped at 1.0).
    """
    q = query.lower()
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}

    for domain, patterns in _PATTERNS:
        hits = [p for p in patterns if re.search(p, q)]
        if hits:
            scores[domain] = len(hits)
            matched[domain] = hits

    if not scores:
        return IntentResult(domain="other", label=DOMAIN_LABELS["other"], confidence=0.0)

    best = max(scores, key=lambda d: scores[d])
    # Confidence: clamp match count to [0, 1] range — 3+ matches = full confidence
    confidence = min(scores[best] / 3.0, 1.0)

    return IntentResult(
        domain=best,
        label=DOMAIN_LABELS[best],
        confidence=round(confidence, 2),
        matched_patterns=matched[best],
    )
