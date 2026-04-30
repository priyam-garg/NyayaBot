import asyncio
import logging

from fastapi import APIRouter, Depends

from app.models.schemas import (
    CompareRequest,
    CompareResponse,
    CompareMethodResult,
    CompareHitOut,
)
from app.services.intent_classifier import classify_intent
from app.services.query_normalizer import normalize_query
from app.services.retrieval_compare import compare_all
from app.services.security import current_user_id

log = logging.getLogger("nyayabot.compare")
router = APIRouter(prefix="/compare", tags=["compare"])


@router.post("", response_model=CompareResponse)
async def compare(
    body: CompareRequest,
    user_id: str = Depends(current_user_id),
) -> CompareResponse:
    normalized, expanded = normalize_query(body.query)
    intent = classify_intent(body.query)

    results = await asyncio.to_thread(compare_all, expanded, body.top_k)

    method_results = [
        CompareMethodResult(
            method=r.method,
            label=r.label,
            description=r.description,
            latency_ms=r.latency_ms,
            hits=[
                CompareHitOut(
                    text=h.text,
                    source=h.source,
                    chunk_index=h.chunk_index,
                    score=h.score,
                    section_number=h.section_number,
                    section_title=h.section_title,
                )
                for h in r.hits
            ],
        )
        for r in results
    ]

    return CompareResponse(
        query=body.query,
        normalized_query=normalized,
        intent_domain=intent.domain,
        intent_label=intent.label,
        intent_confidence=intent.confidence,
        methods=method_results,
    )
