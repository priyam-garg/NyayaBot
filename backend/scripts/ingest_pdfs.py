"""One-shot PDF ingestion: data/ingest/*.pdf -> chunks -> Qdrant.

Run from the backend/ directory:
    python -m scripts.ingest_pdfs

Re-running is safe — points are upserted with deterministic ids per
(source, chunk_index), so repeat runs overwrite prior content rather
than duplicating it.
"""
from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client.http import models as qmodels

# Allow running as `python scripts/ingest_pdfs.py` from backend/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.embeddings import embed_texts
from app.services.qdrant_client import ensure_collection, get_qdrant

INGEST_DIR = ROOT / "data" / "ingest"
BATCH_SIZE = 128


def pdf_display_name(pdf_path: Path, reader: PdfReader) -> str:
    """Return a human-readable document title: PDF metadata > clean filename."""
    try:
        meta = reader.metadata
        if meta and meta.title and meta.title.strip():
            return meta.title.strip()
    except Exception:
        pass
    # Clean up filename: strip extension, replace separators, title-case
    stem = pdf_path.stem
    name = stem.replace("_", " ").replace("-", " ")
    return name.title()


def extract_text(pdf_path: Path, reader: PdfReader) -> str:
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            print(f"  warn: page extract failed in {pdf_path.name}: {e}")
    return "\n\n".join(pages)


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def point_id(source: str, chunk_index: int) -> str:
    h = hashlib.sha1(f"{source}:{chunk_index}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(h[:32]))


def upsert_batch(points: list[qmodels.PointStruct]) -> None:
    if not points:
        return
    get_qdrant().upsert(
        collection_name=get_settings().qdrant_collection,
        points=points,
    )


def main() -> int:
    if not INGEST_DIR.exists():
        print(f"error: {INGEST_DIR} does not exist")
        return 1

    pdfs = sorted(INGEST_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs found in {INGEST_DIR}")
        return 1

    ensure_collection()

    total_chunks = 0
    for pdf in pdfs:
        print(f"reading {pdf.name}")
        reader = PdfReader(str(pdf))
        display_name = pdf_display_name(pdf, reader)
        print(f"  title: {display_name}")
        text = extract_text(pdf, reader)
        if not text.strip():
            print(f"  skip: empty text")
            continue

        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks")

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            vectors = embed_texts(batch)
            points = [
                qmodels.PointStruct(
                    id=point_id(pdf.name, start + i),
                    vector=vec,
                    payload={
                        "source": pdf.name,
                        "document_title": display_name,
                        "chunk_index": start + i,
                        "text": chunk,
                    },
                )
                for i, (chunk, vec) in enumerate(zip(batch, vectors))
            ]
            upsert_batch(points)
            total_chunks += len(points)

    count = get_qdrant().count(
        collection_name=get_settings().qdrant_collection, exact=True
    ).count
    print(f"\ndone. upserted {total_chunks} chunks. collection now has {count} points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
