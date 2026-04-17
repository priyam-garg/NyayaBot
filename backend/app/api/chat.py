import asyncio
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import ChatRequest, ChatResponse
from app.services.mongo import sessions_col, messages_col
from app.services.rag import run_rag
from app.services.security import current_user_id

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(body: ChatRequest, user_id: str = Depends(current_user_id)) -> ChatResponse:
    try:
        sid = ObjectId(body.session_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id")

    session = await sessions_col().find_one({"_id": sid, "user_id": user_id})
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    now = datetime.now(timezone.utc)
    await messages_col().insert_one({
        "session_id": body.session_id,
        "role": "user",
        "content": body.message,
        "created_at": now,
    })

    answer, refused, top_score, sources = await asyncio.to_thread(run_rag, body.message)

    await messages_col().insert_one({
        "session_id": body.session_id,
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "created_at": datetime.now(timezone.utc),
    })

    update = {"updated_at": datetime.now(timezone.utc)}
    if session.get("title") in (None, "", "New chat"):
        update["title"] = body.message.strip()[:60]
    await sessions_col().update_one({"_id": sid}, {"$set": update})

    return ChatResponse(answer=answer, refused=refused, top_score=top_score, sources=sources)
