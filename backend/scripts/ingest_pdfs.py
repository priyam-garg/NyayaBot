"""One-shot PDF ingestion: data/ingest/*.pdf -> chunks -> Qdrant.

Run from the backend/ directory:
    python -m scripts.ingest_pdfs

Re-running is safe — points are upserted with deterministic ids per
(source, chunk_index), so repeat runs overwrite prior content rather
than duplicating it.

Each chunk now carries section metadata (section_number, section_title,
parent_act) extracted via regex from the text, enabling section-level
retrieval and display.
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client.http import models as qmodels

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.embeddings import embed_texts
from app.services.qdrant_client import ensure_collection, get_qdrant

INGEST_DIR = ROOT / "data" / "ingest"
BATCH_SIZE = 128

# ── Section header patterns (Indian legal docs) ────────────────────────────
_SECTION_PATTERNS = [
    # "Section 7(1) — Filing fee"
    re.compile(
        r"^(?:Section|Sec\.?)\s+(\d+[A-Z]?(?:\(\d+[A-Z]?\))?)\s*[.:\-—]?\s*(.{0,80})",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "7. Short title and commencement"
    re.compile(
        r"^(\d{1,3}[A-Z]?)\.\s+([A-Z][^\n.]{5,70})",
        re.MULTILINE,
    ),
    # "CHAPTER III — Rights of Information Seekers"
    re.compile(
        r"^(CHAPTER\s+[IVXLCDM\d]+)\s*[.:\-—]?\s*(.{0,80})",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Known act names matched against filename / PDF title
_ACT_NAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"rti|right.to.information", re.I), "Right to Information Act, 2005"),
    (re.compile(r"consumer.protection|cpa", re.I), "Consumer Protection Act, 2019"),
    (re.compile(r"ipc|indian.penal.code", re.I), "Indian Penal Code, 1860"),
    (re.compile(r"crpc|criminal.procedure", re.I), "Code of Criminal Procedure, 1973"),
    (re.compile(r"cpc|civil.procedure", re.I), "Code of Civil Procedure, 1908"),
    (re.compile(r"it.act|information.technology", re.I), "Information Technology Act, 2000"),
    (re.compile(r"motor.vehicle|mva", re.I), "Motor Vehicles Act, 1988"),
    (re.compile(r"labour|labor|industrial.dispute", re.I), "Industrial Disputes Act, 1947"),
    (re.compile(r"transfer.of.property|topa", re.I), "Transfer of Property Act, 1882"),
]


def _banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"[INGEST] {title}")
    print("=" * 78)


def _line(message: str) -> None:
    print(f"[INGEST] {message}")


def pdf_display_name(pdf_path: Path, reader: PdfReader) -> str:
    try:
        meta = reader.metadata
        if meta and meta.title and meta.title.strip():
            return meta.title.strip()
    except Exception:
        pass
    stem = pdf_path.stem
    return stem.replace("_", " ").replace("-", " ").title()


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


def detect_parent_act(name: str) -> str:
    for pattern, act_name in _ACT_NAMES:
        if pattern.search(name):
            return act_name
    return ""


def extract_section_info(chunk: str) -> tuple[str, str]:
    """
    Extract (section_number, section_title) from the first matching header
    found within a chunk. Returns ("", "") when no header is detected.
    """
    for pat in _SECTION_PATTERNS:
        m = pat.search(chunk)
        if m:
            sec_num = m.group(1).strip()
            sec_title = m.group(2).strip()[:80] if m.lastindex >= 2 else ""
            # Discard noisy titles (all caps headings that are likely page headers)
            if sec_title and sec_title == sec_title.upper() and len(sec_title) > 40:
                sec_title = ""
            return sec_num, sec_title
    return "", ""


def upsert_batch(points: list[qmodels.PointStruct]) -> None:
    if not points:
        return
    get_qdrant().upsert(
        collection_name=get_settings().qdrant_collection,
        points=points,
    )


def main() -> int:
    run_start = time.perf_counter()
    _banner("PDF ingestion started")
    _line(f"Source directory : {INGEST_DIR}")
    _line(f"Batch size       : {BATCH_SIZE}")

    if not INGEST_DIR.exists():
        _line(f"ERROR: ingest directory does not exist -> {INGEST_DIR}")
        return 1

    pdfs = sorted(INGEST_DIR.glob("*.pdf"))
    if not pdfs:
        _line(f"No PDFs found in {INGEST_DIR}")
        return 1

    _line(f"Found {len(pdfs)} PDF(s)")
    _line("Ensuring Qdrant collection exists (will retry if cluster is hibernated)...")

    _RETRY_DELAYS = [10, 20, 30]
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, 1):
        if delay:
            _line(f"Qdrant timeout — retrying in {delay}s (attempt {attempt}/{len(_RETRY_DELAYS)+1})…")
            time.sleep(delay)
        try:
            ensure_collection()
            break
        except Exception as exc:
            msg = str(exc)
            if "10060" in msg or "ConnectTimeout" in msg or "timed out" in msg.lower():
                if attempt <= len(_RETRY_DELAYS):
                    continue
            _line(f"ERROR connecting to Qdrant: {exc}")
            return 1

    _line("Qdrant collection ready")

    total_chunks = 0
    for index, pdf in enumerate(pdfs, 1):
        file_start = time.perf_counter()
        _banner(f"Processing file {index}/{len(pdfs)}: {pdf.name}")

        reader = PdfReader(str(pdf))
        display_name = pdf_display_name(pdf, reader)
        parent_act = detect_parent_act(pdf.name) or detect_parent_act(display_name)
        _line(f"Document title : {display_name}")
        _line(f"Parent act     : {parent_act or 'unknown'}")

        text = extract_text(pdf, reader)
        if not text.strip():
            _line("SKIP: extracted text is empty")
            continue

        chunks = chunk_text(text)
        _line(f"Chunks generated: {len(chunks)}")

        sec_found = 0
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start: start + BATCH_SIZE]
            batch_no = (start // BATCH_SIZE) + 1
            _line(f"Batch {batch_no}: embedding+upsert indexes {start}–{start + len(batch) - 1}")

            vectors = embed_texts(batch)
            points = []
            for i, (chunk, vec) in enumerate(zip(batch, vectors)):
                sec_num, sec_title = extract_section_info(chunk)
                if sec_num:
                    sec_found += 1
                points.append(qmodels.PointStruct(
                    id=point_id(pdf.name, start + i),
                    vector=vec,
                    payload={
                        "source": pdf.name,
                        "document_title": display_name,
                        "chunk_index": start + i,
                        "text": chunk,
                        "section_number": sec_num,
                        "section_title": sec_title,
                        "parent_act": parent_act,
                    },
                ))

            upsert_batch(points)
            total_chunks += len(points)
            _line(f"Batch {batch_no}: upserted {len(points)} points | sections found so far: {sec_found}")

        file_elapsed_ms = int((time.perf_counter() - file_start) * 1000)
        _line(f"Completed {pdf.name} in {file_elapsed_ms} ms | sections detected: {sec_found}/{len(chunks)}")

    count = get_qdrant().count(
        collection_name=get_settings().qdrant_collection, exact=True
    ).count
    total_elapsed_ms = int((time.perf_counter() - run_start) * 1000)
    _banner("Ingestion finished")
    _line(f"Total chunks upserted  : {total_chunks}")
    _line(f"Collection point count : {count}")
    _line(f"Total runtime          : {total_elapsed_ms} ms")
    _line("Tip: run this again any time — existing points are upserted (no duplicates).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
