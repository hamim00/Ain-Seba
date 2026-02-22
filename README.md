# AinSeba (আইনসেবা) — Bangladesh Legal Aid RAG Assistant

> **Phase 1 + Phase 2: Data Ingestion → Vector Store & Retrieval**

AinSeba ("Law Service" in Bangla) is a RAG-based legal assistant that helps 170+ million Bangladeshis understand their legal rights by answering questions grounded in actual Bangladesh legislation with section-level citations.

---

## Project Structure (Phase 1 + 2)

```
ainseba/
├── README.md
├── requirements.txt                   # Phase 1 dependencies
├── requirements-phase2.txt            # Phase 2 dependencies (includes Phase 1)
├── .env.example                       # Environment config (API keys, settings)
├── .gitignore
│
├── data/
│   ├── raw/                           # Original law PDFs
│   └── processed/                     # Phase 1 output (JSON chunks, CSV, reports)
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Central configuration & law registry
│   │
│   ├── vectorstore/                   # ⭐ Phase 2: Vector Store
│   │   ├── __init__.py
│   │   ├── chroma_store.py            # ChromaDB wrapper (persistent storage)
│   │   ├── embeddings.py              # OpenAI text-embedding-3-small wrapper
│   │   └── populate.py                # Loads Phase 1 chunks → embeds → stores
│   │
│   └── retrieval/                     # ⭐ Phase 2: Retrieval
│       ├── __init__.py
│       ├── retriever.py               # Full retrieval pipeline (embed → search → rerank)
│       └── reranker.py                # Cross-encoder reranker (ms-marco-MiniLM)
│
├── scripts/
│   ├── __init__.py
│   ├── run_pipeline.py                # Phase 1 CLI (PDF → chunks)
│   ├── run_vectorstore.py             # ⭐ Phase 2 CLI (embed → store → query)
│   ├── test_retrieval.py              # ⭐ Retrieval quality tests (15 sample queries)
│   └── ingestion/                     # Phase 1 modules
│       ├── pdf_extractor.py
│       ├── text_cleaner.py
│       ├── chunker.py
│       ├── pipeline.py
│       ├── quality_report.py
│       ├── create_sample_pdf.py
│       └── download_laws.py
│
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py              # 31 Phase 1 tests
│   └── test_vectorstore.py            # ⭐ 27 Phase 2 tests
│
├── chroma_db/                         # ⭐ ChromaDB persistent storage (auto-created)
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
# Phase 1 only:
pip install -r requirements.txt

# Phase 1 + Phase 2 (recommended):
pip install -r requirements-phase2.txt
```

> **Note:** Phase 2 installs PyTorch (CPU), sentence-transformers, and chromadb.
> First install may take a few minutes.

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
# Run all tests (58 total: 31 Phase 1 + 27 Phase 2)
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

## Key Design Decisions

### Phase 1
1. **Section-Level Chunking** — Parses legal structure (Part → Chapter → Section) so each chunk is a coherent legal concept
2. **Rich Metadata** — Every chunk carries act_name, chapter, section_number, category, year
3. **Token-Based Sizing** — Uses tiktoken for accurate chunk sizing matching the embedding model
4. **Quality Scoring** — Automated reports catch issues early

### Phase 2
5. **Two-Stage Retrieval** — Fast vector search (top 10) → precise cross-encoder reranking (top 5). This balances speed and accuracy.
6. **Metadata Filtering** — ChromaDB's native metadata filters enable scoping searches to specific acts, categories, or year ranges before vector similarity
7. **Persistent Storage** — ChromaDB persists to disk so you don't need to re-embed after restarts
8. **Batch Embedding** — Efficient API usage with configurable batch sizes and retry logic
9. **Citation Generation** — Each result auto-generates a citation string: "Labour Act 2006, Chapter III, Section 42 (Maximum working hours)"

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

## What's Next (Phase 3)

Phase 3 will connect retrieval to GPT-4o-mini with a crafted prompt to generate citation-grounded answers:
1. Design system prompt with legal assistant role
2. Build LangChain RAG chain
3. Implement conversation memory
4. Add source document tracking
5. Test with diverse query types

---

## Tech Stack

| Tool | Purpose | Phase |
|------|---------|-------|
| PyMuPDF | PDF text extraction | 1 |
| tiktoken | Token counting | 1 |
| rich | Terminal output | 1, 2 |
| pytest | Testing | 1, 2 |
| **OpenAI** | `text-embedding-3-small` embeddings | **2** |
| **ChromaDB** | Vector database (persistent) | **2** |
| **sentence-transformers** | Cross-encoder reranker | **2** |

---

**Built by Mahmudul Hasan** | East West University, CSE
