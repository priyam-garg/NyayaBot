# NyayaBot

Legal-document Q&A assistant. RAG over ingested PDFs, Gemini Flash 2.0 answers, similarity-threshold refusal for out-of-scope queries, per-user chat history.

## Stack

| Layer           | Tool                                                      |
|-----------------|-----------------------------------------------------------|
| Framework       | LangChain                                                 |
| LLM             | Gemini Flash 2.0                                          |
| Embeddings      | Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector Store    | Qdrant Cloud                                              |
| PDF Ingestion   | pypdf                                                     |
| DB (all data)   | MongoDB — `users`, `sessions`, `messages`                 |
| Frontend        | React (Vite) + axios + react-router-dom                   |
| Backend         | FastAPI + uvicorn                                         |
| Auth            | JWT + bcrypt                                              |

## Prerequisites

- Python 3.10+
- Node 18+
- A Qdrant Cloud cluster (URL + API key)
- A Google AI Studio API key for Gemini
- MongoDB — local install or Atlas connection string

## Backend setup

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# edit .env with real GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, MONGODB_URI, JWT_SECRET
```

### Ingest PDFs (run once, or whenever documents change)

Drop legal PDFs into `backend/data/ingest/`, then:

```bash
cd backend
python -m scripts.ingest_pdfs
```

Re-running is safe — point ids are deterministic per `(source, chunk_index)`, so repeat runs upsert rather than duplicate.

### Run the API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Health check: <http://localhost:8000/health>. OpenAPI docs: <http://localhost:8000/docs>.

## Frontend setup

```bash
cd frontend
npm install
cp .env.example .env  # VITE_API_URL defaults to http://localhost:8000
npm run dev
```

Open <http://localhost:5173>.

## End-to-end verification

1. Mongo and Qdrant Cloud reachable; `.env` populated on both sides.
2. At least one PDF ingested (`python -m scripts.ingest_pdfs` prints collection count > 0).
3. Backend and frontend running.
4. Sign up at `/signup` → redirects into chat.
5. Ask an in-scope question → grounded answer from Gemini.
6. Ask `what is photosynthesis?` → refusal (`refused: true` on the response).
7. Sign out, sign back in from another tab → sidebar shows prior sessions; clicking one reloads the conversation.
8. Confirm the `messages` collection in MongoDB contains both user and assistant rows per exchange.

## Tuning refusal

The threshold is `SIMILARITY_THRESHOLD` in `.env` (default 0.6, cosine). Raise it if refusals let too much through; lower if in-scope questions are being refused. Embeddings are L2-normalized on both ingest and query, so cosine scores are directly comparable to this fixed threshold.
