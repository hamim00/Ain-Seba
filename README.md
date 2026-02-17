# AinSeba (আইনসেবা) — Bangladesh Legal Aid RAG Assistant

> **Phase 1: Data Collection & Ingestion Pipeline**

AinSeba ("Law Service" in Bangla) is a RAG-based legal assistant that helps 170+ million Bangladeshis understand their legal rights by answering questions grounded in actual Bangladesh legislation with section-level citations.

---

## Phase 1 Overview

Phase 1 builds the **data ingestion pipeline** — the foundation that processes raw Bangladesh law PDFs into clean, metadata-rich chunks ready for embedding and retrieval in later phases.

### What Phase 1 Does

```
PDF Files → Text Extraction → Cleaning → Metadata-Aware Chunking → JSON/CSV Export → Quality Report
```

1. **PDF Text Extraction** — Uses PyMuPDF (fitz) to extract text from law PDFs
2. **Text Cleaning** — Removes noise (headers, footers, page numbers, watermarks, encoding artifacts)
3. **Metadata-Aware Chunking** — Splits documents respecting legal structure (Part → Chapter → Section) with rich metadata
4. **Export** — Saves chunks as JSON (for Phase 2 embedding) and CSV (for inspection)
5. **Quality Report** — Generates automated quality assessment with scoring

---

## Project Structure (Phase 1)

```
ainseba/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment config template
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── raw/                           # Original law PDFs go here
│   │   └── .gitkeep_readme
│   └── processed/                     # Pipeline output (JSON, CSV, reports)
│       └── .gitkeep_readme
│
├── src/
│   ├── __init__.py
│   └── config.py                      # Central configuration & law registry
│
├── scripts/
│   ├── __init__.py
│   ├── run_pipeline.py                # CLI entry point (main runner)
│   └── ingestion/
│       ├── __init__.py
│       ├── pdf_extractor.py           # PDF → raw text (PyMuPDF)
│       ├── text_cleaner.py            # Noise removal & normalization
│       ├── chunker.py                 # Metadata-aware document chunking
│       ├── quality_report.py          # Chunk quality assessment
│       ├── pipeline.py                # Orchestrator (ties everything together)
│       ├── create_sample_pdf.py       # Test PDF generator
│       └── download_laws.py           # PDF download guide & checker
│
├── tests/
│   ├── __init__.py
│   └── test_ingestion.py             # Unit + integration tests
│
└── docs/                              # Documentation
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+ (recommended: 3.11 or 3.12)
- pip (Python package manager)
- Git

### Step 1: Clone & Navigate

```bash
# If starting fresh
mkdir ainseba && cd ainseba

# Or if you have the zip
unzip ainseba-phase1.zip
cd ainseba
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env if you need to change any settings (defaults work fine)
```

### Step 5: Verify Installation

```bash
# Run tests to verify everything works
pytest tests/test_ingestion.py -v
```

Expected output:
```
tests/test_ingestion.py::TestTextCleaner::test_removes_page_numbers PASSED
tests/test_ingestion.py::TestTextCleaner::test_removes_gazette_headers PASSED
tests/test_ingestion.py::TestLegalStructureParser::test_finds_sections PASSED
tests/test_ingestion.py::TestMetadataAwareChunker::test_chunks_simple_document PASSED
... (all tests should pass)
```

---

## Running the Pipeline

### Quick Test (No PDFs needed!)

Test the entire pipeline with a generated sample PDF:

```bash
python scripts/run_pipeline.py --sample
```

**Expected output:**
```
🧪 Running pipeline with SAMPLE PDF...

✓ Sample PDF created: data/raw/sample_workers_protection_act_2024.pdf
  Pages: 4
  Size: 12.3 KB

[1/5] Extracting text from PDF...
  Extracted 4 pages, 8,234 characters
[2/5] Cleaning extracted text...
  Cleaned: 8,234 → 7,891 characters (4.2% noise removed)
[3/5] Chunking with metadata extraction...
  Found 22 structural markers
  Created 17 chunks (avg 87 tokens/chunk)
[4/5] Exporting processed data...
  JSON saved: data/processed/sample_workers_2024_chunks.json
  CSV saved: data/processed/sample_workers_2024_chunks.csv
  Cleaned text saved: data/processed/sample_workers_2024_cleaned.txt
[5/5] Generating quality report...

✅ Sample pipeline run COMPLETE!
📊 Quality Score: XX/100
📦 Total Chunks: 17
🔤 Total Tokens: ~1,500
```

### Process Real Law PDFs

#### Step 1: Download PDFs

```bash
# See download instructions and check what you have
python scripts/run_pipeline.py --check
```

This shows which PDFs you need and where to save them. Download PDFs from [bdlaws.minlaw.gov.bd](http://bdlaws.minlaw.gov.bd) using the browser Print-to-PDF method.

#### Step 2: Process All Available PDFs

```bash
# Process everything that's been downloaded
python scripts/run_pipeline.py --all

# Or process only high-priority laws first
python scripts/run_pipeline.py --priority P0
```

#### Step 3: Process a Single PDF

```bash
python scripts/run_pipeline.py --file data/raw/bangladesh_labour_act_2006.pdf
```

#### Step 4: View Quality Reports

```bash
python scripts/run_pipeline.py --report
```

---

## Understanding the Output

After running the pipeline, `data/processed/` will contain:

| File | Purpose |
|------|---------|
| `{act_id}_chunks.json` | **Primary output** — chunks with full metadata (used in Phase 2) |
| `{act_id}_chunks.csv` | CSV version for spreadsheet inspection |
| `{act_id}_cleaned.txt` | Full cleaned text (for debugging) |
| `{act_id}_quality_report.json` | Quality assessment report |
| `all_chunks_combined.json` | All chunks from all laws in one file |
| `all_chunks_combined.csv` | Combined CSV |
| `combined_quality_report.json` | Overall quality report |
| `pipeline_run_log.json` | Pipeline execution log |
| `pipeline.log` | Detailed execution log |

### Chunk JSON Format

Each chunk in the JSON looks like:

```json
{
  "chunk_id": "labour_act_2006_a1b2c3d4e5f6",
  "text": "Section 42. Maximum working hours\n(1) No worker shall be required or allowed to work in an establishment for more than eight hours in any day...",
  "token_count": 342,
  "chunk_index": 15,
  "act_name": "Bangladesh Labour Act 2006",
  "act_id": "labour_act_2006",
  "part": "Part II: Employment Conditions",
  "chapter": "Chapter III: Working Hours",
  "section_number": "42",
  "section_title": "Maximum working hours",
  "category": "Employment",
  "year": 2006,
  "language": "english",
  "page_numbers": []
}
```

### Quality Report

The quality report scores chunks on:

- **Token Distribution** — Are chunks within the 100-800 token target range?
- **Metadata Completeness** — Do chunks have act_name, section_number, etc.?
- **Content Issues** — Empty chunks, very short/long chunks, header-only chunks?
- **Quality Score** — 0-100 overall score with A-F grade

---

## Architecture (Phase 1)

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA INGESTION PIPELINE                     │
│                                                                │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐  │
│  │  PDF Files   │───→│ PDFExtractor │───→│  Raw Text       │  │
│  │  (data/raw/) │    │  (PyMuPDF)   │    │  (per page)     │  │
│  └─────────────┘    └──────────────┘    └───────┬─────────┘  │
│                                                   │            │
│                                          ┌────────▼────────┐  │
│                                          │  TextCleaner    │  │
│                                          │  - Remove noise │  │
│                                          │  - Fix encoding │  │
│                                          │  - Normalize    │  │
│                                          └────────┬────────┘  │
│                                                   │            │
│                                          ┌────────▼────────┐  │
│                                          │ StructureParser │  │
│                                          │ - Find Parts    │  │
│                                          │ - Find Chapters │  │
│                                          │ - Find Sections │  │
│                                          └────────┬────────┘  │
│                                                   │            │
│                                          ┌────────▼────────┐  │
│                                          │ MetadataAware   │  │
│                                          │ Chunker         │  │
│                                          │ - Section-level │  │
│                                          │ - Token limits  │  │
│                                          │ - Overlap       │  │
│                                          └────────┬────────┘  │
│                                                   │            │
│  ┌─────────────────────────────────────────────────┤           │
│  │                                                 │           │
│  ▼                          ▼                      ▼           │
│  ┌──────────┐  ┌────────────────┐  ┌───────────────────────┐  │
│  │  JSON    │  │  CSV           │  │  Quality Report       │  │
│  │  Chunks  │  │  (inspection)  │  │  (score + analysis)   │  │
│  └──────────┘  └────────────────┘  └───────────────────────┘  │
│                                                                │
│  Output: data/processed/                                       │
└──────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | File | Responsibility |
|-----------|------|----------------|
| **Config** | `src/config.py` | Law registry, paths, chunking params |
| **PDFExtractor** | `scripts/ingestion/pdf_extractor.py` | PyMuPDF text extraction, page-by-page |
| **TextCleaner** | `scripts/ingestion/text_cleaner.py` | Noise removal, encoding fixes, normalization |
| **LegalStructureParser** | `scripts/ingestion/chunker.py` | Regex-based Part/Chapter/Section detection |
| **MetadataAwareChunker** | `scripts/ingestion/chunker.py` | Token-bounded chunking with metadata |
| **QualityReporter** | `scripts/ingestion/quality_report.py` | Automated quality scoring |
| **IngestionPipeline** | `scripts/ingestion/pipeline.py` | Orchestrates all stages |
| **CLI Runner** | `scripts/run_pipeline.py` | Command-line interface |

---

## Key Design Decisions

1. **Section-Level Chunking** — Instead of naive fixed-size splitting, we parse legal structure (Part → Chapter → Section) so each chunk represents a coherent legal concept. This dramatically improves retrieval quality in Phase 2.

2. **Rich Metadata** — Every chunk carries `act_name`, `chapter`, `section_number`, `section_title`, `category`, and `year`. This enables metadata filtering during retrieval (e.g., "search only in Labour Act, Chapter III").

3. **Token-Based Sizing** — Chunks are sized by tokens (not characters) using tiktoken, matching the tokenizer used by the embedding model (text-embedding-3-small) for accurate sizing.

4. **Overlap** — 100-token overlap between chunks ensures no information is lost at chunk boundaries.

5. **Quality Scoring** — Automated reports catch issues early (too-short chunks, missing metadata, encoding problems) before embedding.

---

## Troubleshooting

### "No structural markers found"
The cleaner or PDF extraction may have altered section markers. Check:
```bash
# View the cleaned text to inspect
cat data/processed/{act_id}_cleaned.txt | head -100
```

### Very low quality score
- Check the quality report for specific issues
- The PDF might be scanned (image-only) — PyMuPDF can't extract text from images
- Try a different PDF source or use OCR

### Tests failing
```bash
# Run with verbose output
pytest tests/test_ingestion.py -v --tb=long
```

### Import errors
Make sure you're in the project root and virtual environment is activated:
```bash
cd ainseba
source venv/bin/activate  # Linux/Mac
```

---

## What's Next (Phase 2)

Phase 2 will take the JSON chunks produced here and:
1. Generate embeddings using `text-embedding-3-small`
2. Store in ChromaDB with metadata
3. Build retrieval functions with metadata filtering
4. Add a cross-encoder reranker
5. Test retrieval quality

The primary input for Phase 2 is `data/processed/all_chunks_combined.json`.

---

## Tech Stack (Phase 1)

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Runtime |
| PyMuPDF | 1.24.14 | PDF text extraction |
| tiktoken | 0.7.0 | Token counting |
| pandas | 2.2.3 | CSV export |
| rich | 13.9.4 | Terminal output |
| pytest | 8.3.3 | Testing |

---

**Built by Mahmudul Hasan** | East West University, CSE
