import asyncio
import logging
import uuid
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import get_settings
from app.models.schemas import DocumentOut, UploadResponse
from app.services.document_processor import process_upload
from app.services.mongo import documents_col, sessions_col
from app.services.security import current_user_id

log = logging.getLogger("nyayabot.documents")
router = APIRouter(prefix="/documents", tags=["documents"])


def _oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session id")


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(current_user_id),
) -> UploadResponse:
    s = get_settings()

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted.",
        )

    chunks_read = []
    total = 0
    while True:
        chunk = await file.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > s.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds the {s.max_upload_bytes // (1024 * 1024)} MB limit.",
            )
        chunks_read.append(chunk)
    pdf_bytes = b"".join(chunks_read)

    sid = _oid(session_id)
    session = await sessions_col().find_one({"_id": sid, "user_id": user_id})
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if session.get("doc_id"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This session already has a document attached. Start a new session to upload another.",
        )

    doc_id = str(uuid.uuid4())

    try:
        display_name, chunk_count = await asyncio.to_thread(
            process_upload,
            pdf_bytes,
            file.filename or "document.pdf",
            user_id,
            session_id,
            doc_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    now = datetime.now(timezone.utc)
    await documents_col().insert_one({
        "_id": doc_id,
        "user_id": user_id,
        "session_id": session_id,
        "doc_id": doc_id,
        "filename": file.filename or "document.pdf",
        "display_name": display_name,
        "chunk_count": chunk_count,
        "uploaded_at": now,
    })
    await sessions_col().update_one({"_id": sid}, {"$set": {"doc_id": doc_id, "updated_at": now}})

    log.info("Uploaded doc_id=%s (%d chunks) to session=%s", doc_id, chunk_count, session_id)
    return UploadResponse(doc_id=doc_id, display_name=display_name, chunk_count=chunk_count)


@router.get("/{session_id}", response_model=DocumentOut | None)
async def get_document(
    session_id: str,
    user_id: str = Depends(current_user_id),
) -> DocumentOut | None:
    sid = _oid(session_id)
    session = await sessions_col().find_one({"_id": sid, "user_id": user_id})
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    doc_id = session.get("doc_id")
    if not doc_id:
        return None

    doc = await documents_col().find_one({"doc_id": doc_id})
    if not doc:
        return None

    return DocumentOut(
        doc_id=doc["doc_id"],
        session_id=doc["session_id"],
        filename=doc["filename"],
        display_name=doc["display_name"],
        chunk_count=doc["chunk_count"],
        uploaded_at=doc["uploaded_at"],
    )
