#!/usr/bin/env python3
"""
AinSeba - Phase 2 Runner (Vector Store & Retrieval)
CLI entry point for Phase 2 operations.

Usage:
    python scripts/run_vectorstore.py --populate         # Embed & store all chunks
    python scripts/run_vectorstore.py --populate-sample   # Embed & store sample chunks
    python scripts/run_vectorstore.py --stats             # Show vector store stats
    python scripts/run_vectorstore.py --query "question"  # Quick retrieval test
    python scripts/run_vectorstore.py --test              # Run full retrieval tests
    python scripts/run_vectorstore.py --interactive       # Interactive query mode
    python scripts/run_vectorstore.py --reset             # Reset the vector store
"""

import sys
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OPENAI_API_KEY,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
    RERANKER_MODEL,
    PROCESSED_DATA_DIR,
)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_populate(sample_only: bool = False):
    """Embed chunks and store in ChromaDB."""
    from src.vectorstore.populate import load_chunks, populate_vectorstore

    if sample_only:
        # Make sure sample data exists
        sample_json = PROCESSED_DATA_DIR / "sample_workers_2024_chunks.json"
        if not sample_json.exists():
            print("Sample chunks not found. Running Phase 1 sample pipeline first...")
            from scripts.ingestion.create_sample_pdf import create_sample_pdf, SAMPLE_LAW_CONFIG
            from scripts.ingestion.pipeline import IngestionPipeline
            create_sample_pdf()
            pipeline = IngestionPipeline()
            pipeline.process_single_pdf(
                PROCESSED_DATA_DIR.parent / "raw" / "sample_workers_protection_act_2024.pdf",
                law_config=SAMPLE_LAW_CONFIG,
            )

        chunks = load_chunks(source_path=sample_json)
    else:
        chunks = load_chunks()

    if not chunks:
        print("❌ No chunks found. Run Phase 1 pipeline first.")
        sys.exit(1)

    populate_vectorstore(chunks)


def cmd_stats():
    """Show vector store statistics."""
    from src.vectorstore.populate import show_stats
    show_stats()


def cmd_query(query: str, no_reranker: bool = False):
    """Run a single query."""
    from scripts.test_retrieval import build_retriever
    from src.retrieval.retriever import print_results

    retriever = build_retriever(use_reranker=not no_reranker)
    results = retriever.retrieve(query=query)
    print_results(results)


def cmd_test(no_reranker: bool = False):
    """Run full retrieval quality tests."""
    from scripts.test_retrieval import build_retriever, run_test_queries

    retriever = build_retriever(use_reranker=not no_reranker)
    run_test_queries(retriever)


def cmd_interactive(no_reranker: bool = False):
    """Interactive query mode."""
    from scripts.test_retrieval import build_retriever, interactive_mode

    retriever = build_retriever(use_reranker=not no_reranker)
    interactive_mode(retriever)


def cmd_reset():
    """Reset the vector store."""
    confirm = input("⚠ This will DELETE all stored embeddings. Continue? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    from src.vectorstore.chroma_store import ChromaStore

    store = ChromaStore(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    store.reset_collection()
    print("✅ Vector store reset complete.")


def main():
    parser = argparse.ArgumentParser(
        description="AinSeba — Phase 2: Vector Store & Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_vectorstore.py --populate-sample    Embed sample data
  python scripts/run_vectorstore.py --populate           Embed all processed chunks
  python scripts/run_vectorstore.py --stats              Show collection stats
  python scripts/run_vectorstore.py --query "penalty for theft"
  python scripts/run_vectorstore.py --test               Run retrieval tests
  python scripts/run_vectorstore.py --interactive        Interactive mode
  python scripts/run_vectorstore.py --reset              Reset everything
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--populate", action="store_true", help="Embed & store all chunks")
    group.add_argument("--populate-sample", action="store_true", help="Embed & store sample chunks only")
    group.add_argument("--stats", action="store_true", help="Show vector store statistics")
    group.add_argument("--query", type=str, help="Run a single query")
    group.add_argument("--test", action="store_true", help="Run retrieval quality tests")
    group.add_argument("--interactive", action="store_true", help="Interactive query mode")
    group.add_argument("--reset", action="store_true", help="Reset the vector store")

    parser.add_argument("--no-reranker", action="store_true", help="Skip cross-encoder reranking")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    args = parser.parse_args()
    setup_logging(args.log_level)

    # Check API key for operations that need it
    needs_api_key = args.populate or args.populate_sample or args.query or args.test or args.interactive
    if needs_api_key and not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set.")
        print("   Create a .env file with: OPENAI_API_KEY=sk-your-key-here")
        print("   Or: export OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    # Route to command
    if args.populate:
        cmd_populate(sample_only=False)
    elif args.populate_sample:
        cmd_populate(sample_only=True)
    elif args.stats:
        cmd_stats()
    elif args.query:
        cmd_query(args.query, no_reranker=args.no_reranker)
    elif args.test:
        cmd_test(no_reranker=args.no_reranker)
    elif args.interactive:
        cmd_interactive(no_reranker=args.no_reranker)
    elif args.reset:
        cmd_reset()


if __name__ == "__main__":
    main()
