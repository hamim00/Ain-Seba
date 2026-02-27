# AinSeba (আইনসেবা) — Bangladesh Legal Aid RAG Assistant

> **Phase 1-5: Data Ingestion -> Vector Store -> RAG Chain -> Bilingual -> FastAPI Backend**

AinSeba ("Law Service" in Bangla) is a RAG-based legal assistant that helps 170+ million Bangladeshis understand their legal rights by answering questions grounded in actual Bangladesh legislation with section-level citations.

---

## Project Structure (Phase 1-5)

```
ainseba/
├── README.md
├── requirements.txt                   # Phase 1
├── requirements-phase2.txt            # Phase 2
├── requirements-phase3.txt            # Phase 3
├── requirements-phase4.txt            # Phase 4
├── requirements-phase5.txt            # Phase 5 (includes all)
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/                           # Original law PDFs
│   └── processed/                     # Chunks, reports, test results
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Central configuration
│   │
│   ├── vectorstore/                   # Phase 2: Vector Store
│   │   ├── chroma_store.py, embeddings.py, populate.py
│   │
│   ├── retrieval/                     # Phase 2: Retrieval
│   │   ├── retriever.py, reranker.py
│   │
│   ├── prompts/                       # Phase 3: Prompt Templates
│   │   └── templates.py
│   │
│   ├── chain/                         # Phase 3: RAG Chain
│   │   ├── rag_chain.py, memory.py, builder.py
│   │
│   ├── language/                      # Phase 4: Bilingual
│   │   ├── detector.py, translator.py, bilingual.py
│   │
│   ├── models/                        # ⭐ Phase 5: Pydantic Schemas
│   │   └── schemas.py                 # Request/Response models
│   │
│   └── api/                           # ⭐ Phase 5: FastAPI Backend
│       ├── app.py                     # FastAPI app with all endpoints
│       └── rate_limiter.py            # In-memory sliding window rate limiter
│
├── scripts/
│   ├── run_pipeline.py                # Phase 1 CLI
│   ├── run_vectorstore.py             # Phase 2 CLI
│   ├── run_chain.py                   # Phase 3 CLI
│   ├── run_bilingual.py               # Phase 4 CLI
│   ├── test_retrieval.py              # Phase 2 retrieval tests
│   └── ingestion/                     # Phase 1 modules
│
├── tests/
│   ├── test_ingestion.py              # 31 Phase 1 tests
│   ├── test_vectorstore.py            # 27 Phase 2 tests
│   ├── test_chain.py                  # 31 Phase 3 tests
│   ├── test_bilingual.py              # 46 Phase 4 tests
│   └── test_api.py                    # ⭐ 38 Phase 5 tests
│
├── chroma_db/                         # ChromaDB persistent storage
└── docs/
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+ (recommended: 3.11 or 3.12)
- pip (Python package manager)
- Git
- **OpenAI API key** (required for Phase 2 — embeddings)

### Step 1: Clone & Navigate

```bash
unzip ainseba-phase2.zip
cd ainseba
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Activate:
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
# All phases (recommended):
pip install -r requirements-phase5.txt
```

> **Note:** Phase 5 adds FastAPI and uvicorn.
> First install may take a few minutes due to PyTorch and sentence-transformers.

### Step 4: Setup Environment

```bash
cp .env.example .env
```

**Edit `.env` and add your OpenAI API key:**
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 5: Verify Installation

```bash
# Run all tests (173 total: 31 + 27 + 31 + 46 + 38)
pytest tests/ -v
```

---

## Phase 1: Data Ingestion (Quick Reference)

```bash
# Test pipeline with sample PDF (no downloads needed)
python scripts/run_pipeline.py --sample

# Check which real PDFs you need
python scripts/run_pipeline.py --check

# Process all downloaded PDFs
python scripts/run_pipeline.py --all

# Process a single PDF
python scripts/run_pipeline.py --file data/raw/penal_code_1860.pdf
```

---

## Phase 2: Vector Store & Retrieval

### What Phase 2 Does

```
Phase 1 JSON Chunks → OpenAI Embeddings → ChromaDB Storage → Similarity Search → Cross-Encoder Reranking
```

1. **Embedding Generation** — Converts text chunks to vectors using `text-embedding-3-small`
2. **ChromaDB Storage** — Stores embeddings with metadata in persistent ChromaDB
3. **Similarity Search** — Finds relevant chunks via cosine similarity
4. **Metadata Filtering** — Filter by act, category, year range
5. **Cross-Encoder Reranking** — Re-scores top candidates with `ms-marco-MiniLM-L-6-v2` for precision

### Running Phase 2

#### Step 1: Populate the Vector Store

```bash
# Populate with sample data (quick test)
python scripts/run_vectorstore.py --populate-sample

# Populate with all processed chunks
python scripts/run_vectorstore.py --populate

# Check what's stored
python scripts/run_vectorstore.py --stats
```

**Expected output (sample):**
```
[1/2] Generating embeddings for 10 chunks...
  Batch 1/1 (10 texts)
  Embedding dimension: 1536
  Total tokens used: ~800

[2/2] Storing in ChromaDB...
  Upserted batch: 10/10

✅ Vector store population complete!
  Total documents in store: 10
  Acts: sample_workers_2024
  Tokens used: ~800
```

#### Step 2: Test Retrieval Quality

```bash
# Run all 15 sample queries
python scripts/test_retrieval.py

# Test a single query
python scripts/run_vectorstore.py --query "What is the penalty for theft?"

# Interactive mode (type queries, see results)
python scripts/run_vectorstore.py --interactive

# Without reranker (faster, less precise)
python scripts/run_vectorstore.py --query "working hours" --no-reranker
```

**Expected retrieval output:**
```
--- Result 1 ---
  Citation:   Labour Act 2006, Chapter III, Section 42 (Maximum working hours)
  Similarity: 0.8923
  Rerank:     2.4567
  Text:       No worker shall be required or allowed to work...
```

#### Step 3: Reset & Re-populate (if needed)

```bash
python scripts/run_vectorstore.py --reset
python scripts/run_vectorstore.py --populate
```

### Retrieval Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                              │
│                                                                    │
│  User Query                                                        │
│      │                                                             │
│      ▼                                                             │
│  ┌─────────────────────┐                                          │
│  │  EmbeddingGenerator │   OpenAI text-embedding-3-small          │
│  │  (embed query)      │   → 1536-dim vector                     │
│  └──────────┬──────────┘                                          │
│             │                                                      │
│             ▼                                                      │
│  ┌─────────────────────┐   ┌────────────────────────────────┐    │
│  │   ChromaDB Query    │──→│  Optional Metadata Filters     │    │
│  │   (cosine search)   │   │  • act_id = "labour_act_2006"  │    │
│  │   top_k = 10        │   │  • category = "Employment"     │    │
│  └──────────┬──────────┘   │  • year >= 2000                │    │
│             │               └────────────────────────────────┘    │
│             ▼                                                      │
│  ┌─────────────────────┐                                          │
│  │  Cross-Encoder      │   ms-marco-MiniLM-L-6-v2                │
│  │  Reranker           │   Scores each (query, doc) pair         │
│  │  top_n = 5          │   → More precise ranking                │
│  └──────────┬──────────┘                                          │
│             │                                                      │
│             ▼                                                      │
│  ┌─────────────────────┐                                          │
│  │  RetrievalResult[]  │   Each result contains:                  │
│  │  • text             │   • similarity_score (0-1)               │
│  │  • citation         │   • rerank_score                         │
│  │  • metadata         │   • act, chapter, section info           │
│  └─────────────────────┘                                          │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **EmbeddingGenerator** | `src/vectorstore/embeddings.py` | OpenAI embedding wrapper with batching & retries |
| **ChromaStore** | `src/vectorstore/chroma_store.py` | ChromaDB wrapper with metadata filtering |
| **Populate** | `src/vectorstore/populate.py` | Loads Phase 1 JSON → embeds → stores |
| **LegalRetriever** | `src/retrieval/retriever.py` | Full pipeline: embed → search → rerank |
| **CrossEncoderReranker** | `src/retrieval/reranker.py` | Cross-encoder reranking for precision |

### Metadata Filtering Examples

```python
from src.retrieval.retriever import LegalRetriever

# Search within a specific act
results = retriever.retrieve("working hours", act_id="labour_act_2006")

# Search by legal category
results = retriever.retrieve("punishment", category="Criminal Law")

# Search modern laws only
results = retriever.retrieve("digital crimes", year_min=2000)

# Direct section lookup (no embedding needed)
results = retriever.search_by_section("labour_act_2006", section_number="42")
```

### Configuration

All Phase 2 settings are in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHROMA_PERSIST_DIR` | `chroma_db` | ChromaDB storage directory |
| `CHROMA_COLLECTION_NAME` | `ainseba_laws` | Collection name |
| `RETRIEVAL_TOP_K` | `10` | Candidates from vector search |
| `RERANK_TOP_N` | `5` | Final results after reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |

---

## Phase 3: RAG Chain & LLM Integration

### What Phase 3 Does

```
User Question -> Retrieve Context (Phase 2) -> Format Prompt -> GPT-4o-mini -> Citation-Grounded Answer
                                              + Conversation Memory (k=5 exchanges)
```

1. **Prompt Engineering** — Carefully designed system prompt that enforces citation, disclaimers, and context-only answers
2. **LangChain LCEL Chain** — Clean pipeline: SystemMessage + formatted context + history -> LLM -> parsed answer
3. **Conversation Memory** — Sliding window (last 5 exchanges) enabling follow-up questions
4. **Source Tracking** — Every response carries full source metadata (act, section, similarity scores)
5. **Streaming** — Token-by-token streaming for real-time responses

### Running Phase 3

#### Quick Start

```bash
# Make sure Phase 2 vector store is populated
python scripts/run_vectorstore.py --populate-sample

# Ask a question
python scripts/run_chain.py --query "What are the maximum working hours per day?"

# Streaming output
python scripts/run_chain.py --stream "My employer hasn't paid me for 3 months"

# Interactive chat (with conversation memory)
python scripts/run_chain.py --interactive

# Preview retrieved context only (no LLM call, no cost)
python scripts/run_chain.py --context "working hours"

# Run the full Q&A test suite (5 queries: direct, situational, follow-up, out-of-scope)
python scripts/run_chain.py --test
```

#### Interactive Mode Commands

```
You: What is the penalty for theft?          # Ask a question
You: What section covers that?               # Follow-up (uses memory)
You: /clear                                  # Reset conversation memory
You: /history                                # View conversation history
You: /sources                                # View sources from last answer
You: /quit                                   # Exit
```

#### Expected Output

```
Q: What are the maximum working hours per day?

╭─────────────── AinSeba Response ───────────────╮
│ According to Section 4 of the Sample Workers   │
│ Protection Act 2024, no worker shall be        │
│ required or allowed to work in an              │
│ establishment for more than eight hours in     │
│ any day (Section 4(1)). Additionally, no       │
│ worker shall work more than forty-eight hours  │
│ in any week (Section 4(2)).                    │
│                                                │
│ If a worker is required to work beyond these   │
│ prescribed hours, the employer must pay        │
│ overtime at the rate of twice the ordinary     │
│ rate of wages (Section 4(3)).                  │
│                                                │
│ **References:**                                │
│ - Sample Workers Protection Act 2024,          │
│   Section 4 (Maximum working hours)            │
│                                                │
│ *Disclaimer: This information is for           │
│ educational purposes only and does not         │
│ constitute legal advice.*                      │
╰────────────────────────────────────────────────╯

Sources (3):
  [1] Sample Workers Protection Act 2024, Section 4 (sim=0.912)
  [2] Sample Workers Protection Act 2024, Section 5 (sim=0.847)
  [3] Sample Workers Protection Act 2024, Section 3 (sim=0.801)
```

### Phase 3 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG CHAIN (Phase 3)                          │
│                                                                     │
│  User Question                                                      │
│      │                                                              │
│      ├──────────────────────────────┐                               │
│      │                              │                               │
│      ▼                              ▼                               │
│  ┌─────────────────┐    ┌──────────────────────┐                   │
│  │ LegalRetriever  │    │ ConversationMemory   │                   │
│  │ (Phase 2)       │    │ (sliding window k=5) │                   │
│  │ embed -> search │    │ last 5 Q&A pairs     │                   │
│  │    -> rerank    │    └──────────┬───────────┘                   │
│  └────────┬────────┘               │                               │
│           │                        │                               │
│           ▼                        ▼                               │
│  ┌─────────────────────────────────────────────┐                   │
│  │          Prompt Assembly                     │                   │
│  │  ┌─────────────────────────────────────┐    │                   │
│  │  │ SYSTEM: AinSeba legal assistant     │    │                   │
│  │  │ - Answer ONLY from context          │    │                   │
│  │  │ - Always cite Section numbers       │    │                   │
│  │  │ - Include disclaimer                │    │                   │
│  │  └─────────────────────────────────────┘    │                   │
│  │  ┌─────────────────────────────────────┐    │                   │
│  │  │ USER: Context + History + Question  │    │                   │
│  │  └─────────────────────────────────────┘    │                   │
│  └────────────────────┬────────────────────────┘                   │
│                       │                                             │
│                       ▼                                             │
│              ┌─────────────────┐                                   │
│              │   GPT-4o-mini   │                                   │
│              │  (LangChain)    │                                   │
│              │  temp=0.1       │                                   │
│              └────────┬────────┘                                   │
│                       │                                             │
│                       ▼                                             │
│              ┌─────────────────┐                                   │
│              │   RAGResponse   │                                   │
│              │  • answer       │                                   │
│              │  • sources[]    │  -> Update memory                 │
│              │  • citations    │                                   │
│              └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **System Prompt** | `src/prompts/templates.py` | Legal assistant role, citation rules, guardrails |
| **User Prompt** | `src/prompts/templates.py` | Context + history + question formatting |
| **ConversationMemory** | `src/chain/memory.py` | Sliding window (k=5), multi-session, export/import |
| **LegalRAGChain** | `src/chain/rag_chain.py` | Full pipeline: retrieve -> format -> LLM -> response |
| **Chain Builder** | `src/chain/builder.py` | Factory wiring all components together |
| **CLI Runner** | `scripts/run_chain.py` | Query, stream, interactive, test modes |

### System Prompt Design

The system prompt enforces these critical behaviors:

| Rule | Why |
|------|-----|
| Answer ONLY from provided context | Prevents hallucination of non-existent laws |
| Always cite Section numbers | Enables users to verify answers |
| Include legal disclaimer | Legal protection — this is not legal advice |
| Handle unknowns gracefully | "I don't have information on this" vs. making up answers |
| Simple language | Target audience is general citizens, not lawyers |

### Configuration (Phase 3)

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | LLM for answer generation |
| `LLM_TEMPERATURE` | `0.1` | Low = more factual, less creative |
| `LLM_MAX_TOKENS` | `1500` | Max response length |
| `CONVERSATION_MEMORY_K` | `5` | Exchanges to remember (5 pairs = 10 messages) |

---

## Phase 4: Bilingual Support (Bangla + English)

### What Phase 4 Does

```
User Query (any language) -> Detect Language -> Translate if needed -> RAG Chain -> Translate response -> Answer in user's language
```

1. **Language Detection** — Multi-strategy detector using Unicode analysis + Banglish pattern matching
2. **Query Translation** — Bangla/Banglish queries translated to English for retrieval (embeddings work better in English)
3. **Response Translation** — English answers translated back to Bangla when user queries in Bangla
4. **Banglish Support** — Detects Bangla written in Latin script (very common in BD)
5. **Language Override** — Force response in any language regardless of detection

### Supported Languages

| Language | Input | Output | Example |
|----------|-------|--------|---------|
| English | Yes | Yes | "What is the penalty for theft?" |
| Bangla | Yes | Yes | "চুরির শাস্তি কী?" |
| Banglish | Yes | Bangla | "churi r shasti ki?" |
| Mixed | Yes | Auto | "Section 42 অনুযায়ী working hours কত?" |

### Running Phase 4

```bash
# Detect language (no API key needed)
python scripts/run_bilingual.py --detect "আমার মালিক বেতন দেয় না"
python scripts/run_bilingual.py --detect "amar malik betan dey nai"

# Query in Bangla
python scripts/run_bilingual.py --query "শ্রমিকের সর্বোচ্চ কর্মঘণ্টা কত?"

# Query in Banglish
python scripts/run_bilingual.py --query "amar malik betan dey nai, ki korbo?"

# Force Bangla response for English query
python scripts/run_bilingual.py --lang bn --query "What is the penalty for theft?"

# Interactive bilingual chat
python scripts/run_bilingual.py --interactive

# Run bilingual test suite (detection + pipeline)
python scripts/run_bilingual.py --test
```

### Bilingual Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   BILINGUAL PIPELINE (Phase 4)                    │
│                                                                    │
│  User Query (Bangla/English/Banglish)                             │
│      │                                                             │
│      ▼                                                             │
│  ┌──────────────────┐                                             │
│  │ Language Detector │  Unicode analysis + Banglish markers       │
│  │ -> en / bn /      │  Confidence score + metadata               │
│  │    banglish       │                                             │
│  └────────┬─────────┘                                             │
│           │                                                        │
│     ┌─────┴──────┐                                                │
│     │ Needs      │                                                │
│     │ translation?│                                               │
│     └─────┬──────┘                                                │
│     Yes   │   No                                                  │
│     ▼     ▼                                                       │
│  ┌────────────────┐   ┌─────────────────┐                        │
│  │ GPT-4o-mini    │   │ Pass through    │                        │
│  │ Translate to   │   │ (already        │                        │
│  │ English        │   │  English)       │                        │
│  └────────┬───────┘   └────────┬────────┘                        │
│           └────────┬───────────┘                                  │
│                    ▼                                               │
│           ┌────────────────┐                                      │
│           │  RAG Chain     │  Phase 3 pipeline                    │
│           │  (English)     │  retrieve -> LLM -> answer           │
│           └────────┬───────┘                                      │
│                    │                                               │
│              ┌─────┴──────┐                                       │
│              │ Response in │                                      │
│              │ Bangla?     │                                      │
│              └─────┬──────┘                                       │
│              Yes   │   No                                         │
│              ▼     ▼                                              │
│  ┌────────────────┐   ┌─────────────────┐                        │
│  │ GPT-4o-mini    │   │ Return English  │                        │
│  │ Translate to   │   │ answer as-is    │                        │
│  │ Bangla         │   └────────┬────────┘                        │
│  └────────┬───────┘            │                                  │
│           └────────┬───────────┘                                  │
│                    ▼                                               │
│           ┌────────────────┐                                      │
│           │ BilingualResponse │                                   │
│           │ • answer (user's language)                            │
│           │ • answer_english (always)                             │
│           │ • detected_language                                   │
│           │ • was_translated                                      │
│           │ • sources[]                                           │
│           └────────────────┘                                      │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Language Detector** | `src/language/detector.py` | Unicode + Banglish pattern detection |
| **Query Translator** | `src/language/translator.py` | GPT-4o-mini translation (query + response) |
| **Bilingual Chain** | `src/language/bilingual.py` | Wraps RAG chain with language support |
| **CLI Runner** | `scripts/run_bilingual.py` | Detect, query, interactive, test modes |

### Configuration (Phase 4)

| Setting | Default | Description |
|---------|---------|-------------|
| `TRANSLATION_MODEL` | `gpt-4o-mini` | Model for translation |
| `DEFAULT_RESPONSE_LANGUAGE` | `auto` | "auto" (detect), "en", or "bn" |

---

## Phase 5: FastAPI Backend

### What Phase 5 Does

Wraps the entire RAG pipeline in a production-ready REST API with auto-generated Swagger documentation.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Submit a legal question (bilingual) |
| `POST` | `/api/query/stream` | Streaming response via SSE |
| `GET` | `/api/health` | Health check + vector store status |
| `GET` | `/api/sources` | List available law documents |
| `POST` | `/api/feedback` | User feedback (1-5 rating) |
| `GET` | `/api/session/{id}` | Conversation history for a session |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/redoc` | ReDoc documentation |

### Running the API

```bash
# Start the server
uvicorn src.api.app:app --reload --port 8000

# Or with custom host/port
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Example API Calls

```bash
# Health check
curl http://localhost:8000/api/health

# Ask a question
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the penalty for theft?"}'

# Ask in Bangla
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "চুরির শাস্তি কী?", "language": "bn"}'

# With filters
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "working hours", "act_id": "labour_act_2006"}'

# Streaming (SSE)
curl -N http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What are worker rights?"}'

# Submit feedback
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "answer": "test", "rating": 5, "comment": "Great!"}'

# List available laws
curl http://localhost:8000/api/sources
```

### API Response Format

```json
{
  "answer": "According to Section 379 of The Penal Code 1860...",
  "answer_english": "According to Section 379...",
  "sources": [
    {
      "citation": "The Penal Code 1860, Section 379 (Punishment for theft)",
      "act_name": "The Penal Code 1860",
      "section_number": "379",
      "similarity_score": 0.91,
      "rerank_score": 3.2
    }
  ],
  "detected_language": "en",
  "response_language": "en",
  "was_translated": false,
  "session_id": "default",
  "retrieval_count": 5,
  "timestamp": "2026-02-18T03:00:00"
}
```

### Features

- **Input Validation** — Pydantic models validate question length (3-2000 chars), language codes, rating ranges
- **Rate Limiting** — 30 requests/minute per IP (configurable)
- **CORS** — Enabled for frontend connection
- **Error Handling** — Structured error responses with status codes
- **SSE Streaming** — Real-time token-by-token response delivery
- **Swagger UI** — Interactive API docs at `/docs`

### Configuration (Phase 5)

| Setting | Default | Description |
|---------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `API_RATE_LIMIT` | `30` | Max requests per window |
| `API_RATE_WINDOW` | `60` | Rate limit window in seconds |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |

---

## Key Design Decisions

### Phase 1
1. **Section-Level Chunking** — Parses legal structure (Part -> Chapter -> Section) so each chunk is a coherent legal concept
2. **Rich Metadata** — Every chunk carries act_name, chapter, section_number, category, year
3. **Token-Based Sizing** — Uses tiktoken for accurate chunk sizing matching the embedding model
4. **Quality Scoring** — Automated reports catch issues early

### Phase 2
5. **Two-Stage Retrieval** — Fast vector search (top 10) -> precise cross-encoder reranking (top 5)
6. **Metadata Filtering** — ChromaDB's native metadata filters enable scoping searches
7. **Persistent Storage** — ChromaDB persists to disk
8. **Batch Embedding** — Efficient API usage with configurable batch sizes and retry logic
9. **Citation Generation** — Each result auto-generates a citation string

### Phase 3
10. **Strict Grounding** — System prompt enforces answers ONLY from retrieved context
11. **LangChain LCEL** — Clean composable pipeline
12. **Sliding Window Memory** — k=5 enables follow-up questions without unbounded token growth
13. **Low Temperature (0.1)** — Keeps answers factual and deterministic
14. **Source Tracking** — Every response carries full source metadata

### Phase 4
15. **Translate-then-Retrieve** — Bangla queries are translated to English before embedding/retrieval, because English embeddings perform significantly better than multilingual ones for legal text
16. **Multi-Strategy Detection** — Unicode script analysis (fast, reliable) + Banglish word matching (catches Latin-script Bangla) + langdetect fallback
17. **Banglish as First-Class** — Recognizes that many Bangladeshis type Bangla in Latin script, not just in Bangla Unicode
18. **Section References Preserved** — Translation keeps "Section 42" in English even in Bangla responses, since these are the official legal identifiers
19. **Zero New Dependencies** — Reuses langdetect (Phase 1) and OpenAI (Phase 2), keeping the stack lean

### Phase 5
20. **Pydantic Validation** — All requests validated with strict types, ranges, and regex patterns before hitting the chain
21. **Sliding Window Rate Limiter** — Per-IP rate limiting (30 req/min default) protects the API from abuse without external dependencies
22. **SSE Streaming** — Token-by-token streaming via Server-Sent Events for real-time response display
23. **Lazy Chain Loading** — Bilingual chain initialized on first request, not at startup, for faster server boot
24. **Auto-Generated Docs** — Swagger UI at `/docs` and ReDoc at `/redoc` with full schema documentation

---

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
# Create .env file
cp .env.example .env
# Edit and add your key:
# OPENAI_API_KEY=sk-your-key-here
```

### "No chunks found"
Run Phase 1 first: `python scripts/run_pipeline.py --sample`

### Cross-encoder model download fails
The reranker downloads on first use (~80MB). If it fails:
```bash
# Use without reranker
python scripts/run_vectorstore.py --query "test" --no-reranker
```

### ChromaDB errors
```bash
# Reset and re-populate
python scripts/run_vectorstore.py --reset
python scripts/run_vectorstore.py --populate
```

---

## What's Next (Phase 6)

Phase 6 will add a Streamlit frontend:
1. Chat interface with message history display
2. Language toggle (Bangla/English)
3. Source citations as expandable sections
4. Example questions sidebar for new users
5. Feedback mechanism (thumbs up/down)

---

## Tech Stack

| Tool | Purpose | Phase |
|------|---------|-------|
| PyMuPDF | PDF text extraction | 1 |
| tiktoken | Token counting | 1 |
| rich | Terminal output | 1-4 |
| pytest | Testing | 1-5 |
| **OpenAI** | Embeddings + LLM + Translation | **2, 3, 4** |
| **ChromaDB** | Vector database (persistent) | **2** |
| **sentence-transformers** | Cross-encoder reranker | **2** |
| **LangChain** | LCEL RAG chain, LLM integration | **3** |
| **langdetect** | Language detection fallback | **1, 4** |
| **FastAPI** | REST API framework | **5** |
| **uvicorn** | ASGI server | **5** |
| **Pydantic** | Request/response validation | **5** |

---

**Built by Mahmudul Hasan** | East West University, CSE
