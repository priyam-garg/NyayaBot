import hashlib
import io
import logging
import uuid

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.services.embeddings import embed_texts
from app.services.qdrant_client import upsert_user_doc_chunks

log = logging.getLogger("nyayabot.docproc")

_BATCH_SIZE = 64


def _pdf_display_name(filename: str, reader: PdfReader) -> str:
    try:
        meta = reader.metadata
        if meta and meta.title and meta.title.strip():
            return meta.title.strip()
    except Exception:
        pass
    stem = filename.rsplit(".", 1)[0]
    return stem.replace("_", " ").replace("-", " ").title()


def _extract_text(reader: PdfReader) -> str:
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            log.warning("Page extract failed: %s", e)
    return "\n\n".join(pages)


def _point_id(doc_id: str, chunk_index: int) -> str:
    h = hashlib.sha1(f"{doc_id}:{chunk_index}".encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


def process_upload(
    pdf_bytes: bytes,
    filename: str,
    user_id: str,
    session_id: str,
    doc_id: str,
) -> tuple[str, int]:
    """
    Parse, chunk, embed, and upsert a user-uploaded PDF into user_docs.

    Returns (display_name, chunk_count).
    Raises ValueError if PDF has no extractable text.
    """
    s = get_settings()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    display_name = _pdf_display_name(filename, reader)
    text = _extract_text(reader)

    if not text.strip():
        raise ValueError("PDF contains no extractable text (possibly a scanned image).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.upload_chunk_size,
        chunk_overlap=s.upload_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]

    if not chunks:
        raise ValueError("No text chunks could be produced from the PDF.")

    for start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[start: start + _BATCH_SIZE]
        vectors = embed_texts(batch)
        points = [
            qmodels.PointStruct(
                id=_point_id(doc_id, start + i),
                vector=vec,
                payload={
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "session_id": session_id,
                    "source": filename,
                    "text": chunk,
                    "chunk_index": start + i,
                },
            )
            for i, (chunk, vec) in enumerate(zip(batch, vectors))
        ]
        upsert_user_doc_chunks(points)
        log.info("Upserted %d chunks (doc_id=%s, batch_start=%d)", len(points), doc_id, start)

    return display_name, len(chunks)
