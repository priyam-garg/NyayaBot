import asyncio
import logging
import time
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import ChatRequest, ChatResponse
from app.services.mongo import sessions_col, messages_col
from app.services.rag import run_rag
from app.services.security import current_user_id

log = logging.getLogger("nyayabot.chat")
router = APIRouter(prefix="/chat", tags=["chat"])

_SCENARIO_PREFIX = "[SCENARIO]"


def _extract_rag_query(message: str) -> str:
    """Strip [SCENARIO]{json} prefix from scenario messages, returning only the natural language line."""
    if message.startswith(_SCENARIO_PREFIX):
        lines = message.split("\n", 1)
        if len(lines) > 1:
            return lines[1].strip()
    return message


def _flow_banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"[CHAT FLOW] {title}")
    print("=" * 72)


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, user_id: str = Depends(current_user_id)) -> ChatResponse:
    req_start = time.perf_counter()
    _flow_banner("Incoming /chat request")
    print(f"session_id : {body.session_id}")
    print(f"user_id    : {user_id}")
    print(f"message    : {body.message[:140]}{'...' if len(body.message) > 140 else ''}")

    try:
        sid = ObjectId(body.session_id)
    except (InvalidId, TypeError):
        print("[CHAT FLOW] Invalid session id; returning 400")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id")

    session = await sessions_col().find_one({"_id": sid, "user_id": user_id})
    if not session:
        print("[CHAT FLOW] Session not found for user; returning 404")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    print("[CHAT FLOW] Session validated")

    now = datetime.now(timezone.utc)
    await messages_col().insert_one({
        "session_id": body.session_id,
        "role": "user",
        "content": body.message,
        "created_at": now,
    })
    print("[CHAT FLOW] User message saved to MongoDB")

    doc_id: str | None = session.get("doc_id")
    rag_query = _extract_rag_query(body.message)
    print(f"[CHAT FLOW] RAG query prepared (doc_id={doc_id or 'none'})")

    try:
        print("[CHAT FLOW] Calling RAG pipeline...")
        (
            answer, refused, top_score, sources, follow_ups,
            intent_domain, intent_label, normalized_query, top_span,
        ) = await asyncio.to_thread(run_rag, rag_query, doc_id)
        print("[CHAT FLOW] RAG pipeline finished")
    except RuntimeError as exc:
        print(f"[CHAT FLOW] RAG failed: {exc}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    await messages_col().insert_one({
        "session_id": body.session_id,
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "created_at": datetime.now(timezone.utc),
    })
    print("[CHAT FLOW] Assistant answer saved to MongoDB")

    update = {"updated_at": datetime.now(timezone.utc)}
    if session.get("title") in (None, "", "New chat"):
        if body.message.startswith(_SCENARIO_PREFIX):
            title_text = _extract_rag_query(body.message)
        else:
            title_text = body.message.strip()
        update["title"] = title_text[:60]
    await sessions_col().update_one({"_id": sid}, {"$set": update})

    elapsed_ms = int((time.perf_counter() - req_start) * 1000)
    print("-" * 72)
    print(f"[CHAT FLOW] Completed | refused={refused} | top_score={top_score} | sources={len(sources)} | follow_ups={len(follow_ups)}")
    print(f"[CHAT FLOW] Total backend time: {elapsed_ms} ms")
    print("=" * 72 + "\n")

    return ChatResponse(
        answer=answer,
        refused=refused,
        top_score=top_score,
        sources=sources,
        follow_ups=follow_ups,
        intent_domain=intent_domain,
        intent_label=intent_label,
        normalized_query=normalized_query,
        top_span=top_span,
    )
