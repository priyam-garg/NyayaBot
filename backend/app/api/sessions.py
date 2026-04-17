from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import SessionCreate, SessionOut, MessageOut
from app.services.mongo import sessions_col, messages_col
from app.services.security import current_user_id

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid id")


def _session_out(doc) -> SessionOut:
    return SessionOut(
        id=str(doc["_id"]),
        title=doc.get("title") or "New chat",
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(body: SessionCreate, user_id: str = Depends(current_user_id)) -> SessionOut:
    now = datetime.now(timezone.utc)
    doc = {
        "user_id": user_id,
        "title": (body.title or "New chat").strip()[:120],
        "created_at": now,
        "updated_at": now,
    }
    result = await sessions_col().insert_one(doc)
    doc["_id"] = result.inserted_id
    return _session_out(doc)


@router.get("", response_model=list[SessionOut])
async def list_sessions(user_id: str = Depends(current_user_id)) -> list[SessionOut]:
    cursor = sessions_col().find({"user_id": user_id}).sort("updated_at", -1)
    return [_session_out(doc) async for doc in cursor]


@router.get("/{session_id}/messages", response_model=list[MessageOut])
async def get_messages(session_id: str, user_id: str = Depends(current_user_id)) -> list[MessageOut]:
    sid = _oid(session_id)
    session = await sessions_col().find_one({"_id": sid, "user_id": user_id})
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    cursor = messages_col().find({"session_id": session_id}).sort("created_at", 1)
    return [
        MessageOut(
            id=str(doc["_id"]),
            session_id=doc["session_id"],
            role=doc["role"],
            content=doc["content"],
            created_at=doc["created_at"],
        )
        async for doc in cursor
    ]
