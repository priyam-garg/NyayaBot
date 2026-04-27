# NyayaBot: Algorithm & Principles

## System Overview

NyayaBot is a **Retrieval-Augmented Generation (RAG)** system that grounds legal document Q&A in ingested PDF content, reducing hallucinations and ensuring answers are sourced from actual documents.

---

## Complete Pipeline: Question to Answer

### **Phase 1: Data Ingestion (Offline)**

**Trigger:** Run `python -m scripts.ingest_pdfs` once during setup.

#### Step 1.1: PDF Loading
- **Input:** Legal PDF files in `backend/data/ingest/`
- **Technology:** `pypdf` library
- **Process:** Extract text and metadata from each PDF

#### Step 1.2: Semantic Chunking
- **Input:** Raw PDF text
- **Technology:** LangChain's RecursiveCharacterTextSplitter
- **Process:** 
  - Split documents into overlapping chunks (e.g., 1000 chars with 200-char overlap)
  - Preserve context by keeping chunks small but overlapped
  - Maintain source document reference for each chunk
- **Output:** Structured chunks with metadata

#### Step 1.3: Embedding Generation
- **Input:** Each text chunk
- **Technology:** `Sentence Transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Process:**
  - Convert each chunk to a 384-dimensional dense vector
  - L2-normalize embeddings for cosine similarity
  - Preserve semantic meaning in vector space
- **Output:** Dense vector representations

#### Step 1.4: Vector Storage
- **Input:** Chunk + embedding + metadata
- **Technology:** `Qdrant Cloud` (vector database)
- **Process:**
  - Store vectors with:
    - Embedding vector (384 dims)
    - Original text chunk
    - Source document name
    - Chunk index (deterministic point IDs)
  - Point IDs are deterministic: `hash(source, chunk_index)` → safe re-runs
  - Indexed for fast approximate nearest neighbor search (HNSW algorithm)
- **Output:** Queryable vector index

#### Step 1.5: User & Session Setup (MongoDB)
- **Technology:** MongoDB (`users`, `sessions`, `messages` collections)
- **Process:**
  - Store user accounts (email, password hash)
  - Initialize session records on login
  - Ready to store conversation history
- **Output:** User/session records in DB

---

### **Phase 2: Chat at Runtime (Online)**

#### Step 2.1: User Authentication
- **Input:** Email + password (login) or new account (signup)
- **Technology:** JWT + bcrypt
- **Process:**
  - Hash password with bcrypt (salt rounds: typically 10)
  - Store user in MongoDB
  - Generate JWT token on successful auth
  - Token includes user ID, expires at (e.g., 24h)
- **Output:** JWT token; user session created

#### Step 2.2: Question Submission
- **Input:** User types question in React frontend; clicks send
- **Technology:** React (Vite) + axios
- **Process:**
  - Frontend captures question text
  - Attach JWT token to HTTP header: `Authorization: Bearer <token>`
  - POST to `/api/chat/` endpoint
- **Output:** Question transmitted to backend

#### Step 2.3: Server-Side Question Reception & Storage
- **Input:** HTTP request with question
- **Technology:** FastAPI + uvicorn + MongoDB
- **Process:**
  - FastAPI endpoint receives POST
  - Validate JWT token (check signature, expiry, user ID)
  - Extract user ID from token
  - Create new message record in MongoDB: `{ user_id, role: "user", content: question, timestamp }`
  - Session ID attached to all messages in this conversation
- **Output:** Question persisted; user identified

#### Step 2.4: Question-to-Embedding Conversion
- **Input:** User's question text
- **Technology:** Sentence Transformers (same model as ingestion)
- **Process:**
  - Use identical model: `paraphrase-multilingual-MiniLM-L12-v2`
  - Convert question to 384-dim dense vector
  - L2-normalize (critical for cosine similarity)
  - Result: single embedding vector in semantic space
- **Output:** Query vector Q

---

### **Phase 3: Semantic Retrieval (Core RAG)**

#### Step 3.1: Vector Similarity Search
- **Input:** Query vector Q (from step 2.4)
- **Technology:** Qdrant Cloud with HNSW index
- **Process:**
  - Qdrant searches index for top-K nearest neighbors (K typically 5-10)
  - Distance metric: **Cosine similarity**
  - Scoring formula: `similarity = (embedding_1 · embedding_2) / (||embedding_1|| × ||embedding_2||)`
  - Normalized embeddings → cosine score = dot product
  - Returns ranked list: `[(chunk_text, source, similarity_score), ...]`
- **Output:** Top-K retrieved chunks with similarity scores

#### Step 3.2: Similarity Threshold Filtering (Refusal Mechanism)
- **Input:** Top-K chunks with similarity scores
- **Configuration:** `SIMILARITY_THRESHOLD` from `.env` (default: 0.6)
- **Process:**
  - **Core Principle:** If best match score < threshold → question is out-of-scope
  - Compare max(similarity_scores) against threshold
  - If max_score >= threshold: proceed to LLM
  - If max_score < threshold: return refusal response
  - Example:
    - Question: "What is photosynthesis?" (legal doc knowledge base)
    - Max similarity: 0.45 (too low for legal docs)
    - 0.45 < 0.6 → **Refuse** with message like: "This question is outside the scope of our legal documents."
- **Output:** Either proceed with retrieved chunks OR issue refusal

#### Step 3.3: Context Building (Prompt Engineering)
- **Input:** Passed chunks (if threshold satisfied)
- **Technology:** LangChain prompt templates
- **Process:**
  - Rank retrieved chunks by similarity (highest first)
  - Construct prompt:
    ```
    Context from documents:
    [Chunk 1]
    [Chunk 2]
    [Chunk 3]
    ...
    
    User Question: {question}
    
    Answer based ONLY on the context above:
    ```
  - Include source attribution hints in prompt
  - Set temperature (e.g., 0.3) for low hallucination
- **Output:** Full prompt ready for LLM

---

### **Phase 4: LLM Response Generation**

#### Step 4.1: LLM API Call
- **Input:** Constructed prompt + context
- **Technology:** Google Gemini Flash 2.0 (via Google AI SDK)
- **Process:**
  - Call Gemini API with:
    - System prompt: "You are a legal assistant. Answer only from provided documents."
    - User message: Full prompt from step 3.3
    - Temperature: 0.3 (low randomness, grounded answers)
    - Max tokens: 500-1000
  - Gemini generates response token-by-token
  - Stop conditions: max tokens reached or natural end
- **Output:** Generated answer text

#### Step 4.2: Answer Post-Processing
- **Input:** Raw Gemini output
- **Process:**
  - Trim trailing whitespace
  - Add source attribution: "Based on documents: [source names]"
  - Format for JSON response
- **Output:** Final answer string

---

### **Phase 5: Persistence & Response**

#### Step 5.1: Store Assistant Response
- **Input:** Answer text
- **Technology:** MongoDB + FastAPI
- **Process:**
  - Create message record: `{ user_id, session_id, role: "assistant", content: answer, timestamp, sources: [...], similarity_score: max_sim }`
  - Index on `(user_id, session_id)` for quick retrieval
- **Output:** Conversation persisted

#### Step 5.2: Return to Frontend
- **Input:** Answer + metadata
- **Technology:** FastAPI JSON response
- **Process:**
  - HTTP 200 response body:
    ```json
    {
      "content": "The answer is...",
      "sources": ["document_name_1.pdf"],
      "similarity_score": 0.87,
      "refused": false,
      "timestamp": "2026-04-26T10:30:00Z"
    }
    ```
  - Frontend receives and renders in chat UI
- **Output:** User sees answer in React Chat component

#### Step 5.3: Session Persistence
- **Input:** User navigates away or logs out
- **Technology:** MongoDB + React Context
- **Process:**
  - Sessions stored with session_id and user_id
  - On next login: fetch all sessions for user
  - Click session → reload all messages in that session
  - Sidebar shows conversation history
- **Output:** User can resume prior conversations

---

## Key Principles

### **1. Retrieval-Augmented Generation (RAG)**
- **Why:** Reduces hallucinations by grounding answers in real documents
- **How:** Always retrieve relevant chunks BEFORE calling LLM
- **Benefit:** Answers are verifiable and traceable to sources

### **2. Semantic Search via Dense Embeddings**
- **Why:** Keyword search misses synonyms and context; embeddings understand meaning
- **How:** Use Sentence Transformers to convert text → 384-dim vectors
- **Benefit:** "Legal liability" and "responsibility" match semantically even if keywords differ

### **3. Cosine Similarity for Ranking**
- **Why:** Normalized embeddings + cosine distance = interpretable similarity (0 to 1)
- **How:** L2-normalize all embeddings; compute dot product
- **Benefit:** Threshold (e.g., 0.6) works consistently across all queries

### **4. Similarity Threshold Refusal**
- **Why:** Prevents confident-sounding but wrong answers on out-of-scope topics
- **How:** If best match < threshold → refuse instead of answering
- **Tuning:** Raise threshold if hallucinations slip through; lower if valid questions refused
- **Example:** "What's the weather?" has low similarity to legal docs → refused

### **5. Deterministic Vector IDs**
- **Why:** Safe to re-ingest PDFs without duplicates
- **How:** Point ID = hash(source_name, chunk_index) — always same for same chunk
- **Benefit:** Upserts instead of inserts; idempotent ingestion

### **6. Per-User Session History**
- **Why:** Maintains context and allows users to resume conversations
- **How:** MongoDB stores user_id + session_id on every message
- **Benefit:** Multi-device sync; conversation privacy

### **7. JWT Authentication**
- **Why:** Stateless auth; scales; secure
- **How:** Token includes user ID, signed with secret
- **Benefit:** No session table needed; token travels with every request

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     OFFLINE (Ingestion)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PDFs ──pypdf──> Text ──LangChain──> Chunks                │
│                                           │                  │
│                                           │ Sentence         │
│                                           │ Transformers     │
│                                           │                  │
│                                      Embeddings              │
│                                           │                  │
│                                           └──> Qdrant Cloud  │
│                                                  (HNSW Index) │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     ONLINE (Chat Runtime)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Question (React)                                      │
│         │                                                    │
│         └──> FastAPI Backend                                │
│              │                                               │
│              ├─> JWT Validation                             │
│              │                                               │
│              ├─> Embed Question (Sentence Transformers)    │
│              │                                               │
│              ├─> Semantic Search (Qdrant)                   │
│              │                                               │
│              ├─> Similarity Threshold Check (config)        │
│              │                                               │
│              ├─ If PASS: Build Prompt + Context             │
│              │   │                                           │
│              │   └──> Call Gemini Flash 2.0                 │
│              │        │                                      │
│              │        └──> Receive Answer                   │
│              │             │                                 │
│              ├─ If FAIL: Return Refusal (Refused: true)    │
│              │                                               │
│              ├─> Store in MongoDB (messages collection)     │
│              │                                               │
│              └──> Send JSON response to React UI            │
│                   (content, sources, similarity_score)      │
│                                                              │
│  React Chat Component                                       │
│         │                                                    │
│         └──> Render answer + sources in sidebar             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack Summary Table

| **Stage**              | **Technologies Used**              | **Purpose**                          |
|------------------------|------------------------------------|--------------------------------------|
| PDF Ingestion          | pypdf, LangChain                   | Extract & chunk documents            |
| Embedding Generation   | Sentence Transformers              | Semantic vector representation       |
| Vector Storage         | Qdrant Cloud (HNSW)                | Fast approximate nearest neighbor    |
| Question Encoding      | Sentence Transformers              | Query embedding (same model)         |
| Similarity Search      | Qdrant (Cosine distance)           | Retrieve relevant chunks             |
| Threshold Filtering    | Custom logic (config-driven)       | Refusal mechanism                    |
| LLM Generation         | Gemini Flash 2.0                   | Answer generation & reasoning        |
| Database               | MongoDB                            | User, session, message storage       |
| Auth                   | JWT + bcrypt                       | Secure authentication                |
| Backend Framework      | FastAPI + uvicorn                  | HTTP API & server                    |
| Frontend Framework     | React (Vite) + axios               | User interface & API calls           |

---

## Performance Characteristics

- **Embedding Time:** ~10-50ms per question (Sentence Transformers)
- **Vector Search Time:** ~100-200ms for top-10 (Qdrant with HNSW index)
- **LLM Latency:** ~1-3 seconds (Gemini Flash 2.0, streaming)
- **Total E2E Latency:** ~2-5 seconds (most time spent on LLM)
- **Storage:** ~1.5 KB per chunk in Qdrant + embeddings (~380KB per 100 chunks)

---

## Security & Privacy

1. **Password Storage:** Bcrypt with salt (one-way hashing)
2. **Auth Tokens:** JWT signed with backend secret; includes expiry
3. **Database Access:** MongoDB connection string in `.env` (never in code)
4. **API Keys:** Gemini, Qdrant keys in `.env` (never committed)
5. **User Isolation:** Every query filtered by user_id in MongoDB

---

## Handling Edge Cases

| **Edge Case**                    | **Handling**                                          |
|---------------------------------|-------------------------------------------------------|
| Out-of-scope question           | Similarity < threshold → refuse with explanation     |
| Empty/gibberish input           | Embeddings still computed; likely low similarity      |
| Very long question              | Truncate to model max tokens (handled by LangChain)  |
| Duplicate PDF ingestion         | Deterministic point IDs → safe upsert (no duplicates)|
| Session timeout                 | JWT expiry enforced; re-login required                |
| Qdrant unavailable              | FastAPI returns 503 (can add retry logic)             |
| No matching chunks              | Max similarity < threshold → refusal                  |

---

## Why This Architecture Works for Legal Q&A

1. **Accuracy:** RAG + threshold prevents confident hallucinations
2. **Interpretability:** Sources are traceable and verifiable
3. **Scalability:** Vector DB + semantic search handle 1000s of documents
4. **Multi-tenant:** Per-user sessions & MongoDB isolation
5. **Reduced Costs:** Gemini Flash 2.0 is fast & cheap; not every call needs expensive models
6. **Compliance:** Answers are grounded; audit trail in MongoDB

