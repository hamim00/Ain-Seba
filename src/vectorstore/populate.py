"""
AinSeba - Vector Store Population Script
Loads Phase 1 processed chunks, generates embeddings, and stores in ChromaDB.

Usage:
    python -m src.vectorstore.populate                  # All chunks
    python -m src.vectorstore.populate --act labour_act_2006   # Single act
    python -m src.vectorstore.populate --reset           # Reset and re-populate
    python -m src.vectorstore.populate --stats           # Show store statistics
"""

import json
import sys
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    OPENAI_API_KEY,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from src.vectorstore.embeddings import EmbeddingGenerator
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def load_chunks(source_path: Path = None, act_id: str = None) -> list[dict]:
    """
    Load processed chunks from Phase 1 JSON files.

    Args:
        source_path: Specific JSON file to load. If None, loads combined file.
        act_id: If specified, load only this act's chunks file.

    Returns:
        List of chunk dictionaries.
    """
    if source_path:
        path = Path(source_path)
    elif act_id:
        path = PROCESSED_DATA_DIR / f"{act_id}_chunks.json"
    else:
        # Try combined file first, then individual files
        combined = PROCESSED_DATA_DIR / "all_chunks_combined.json"
        if combined.exists():
            path = combined
        else:
            # Collect from individual files
            all_chunks = []
            for json_file in sorted(PROCESSED_DATA_DIR.glob("*_chunks.json")):
                if "combined" not in json_file.name and "quality" not in json_file.name:
                    with open(json_file, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                        all_chunks.extend(chunks)
                        logger.info(f"  Loaded {len(chunks)} chunks from {json_file.name}")
            return all_chunks

    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {path.name}")
    return chunks


def populate_vectorstore(
    chunks: list[dict],
    api_key: str = OPENAI_API_KEY,
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = 100,
) -> dict:
    """
    Embed chunks and store in ChromaDB.

    Args:
        chunks: List of chunk dictionaries from Phase 1.
        api_key: OpenAI API key.
        persist_dir: ChromaDB persistence directory.
        collection_name: ChromaDB collection name.
        embedding_model: OpenAI embedding model name.
        batch_size: Batch size for embedding API calls.

    Returns:
        Summary dict with stats.
    """
    logger.info(f"\n{'='*60}")
    logger.info("AinSeba — Vector Store Population")
    logger.info(f"{'='*60}")
    logger.info(f"Chunks to embed: {len(chunks)}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"ChromaDB dir: {persist_dir}")
    logger.info(f"Collection: {collection_name}")

    # Step 1: Initialize components
    embedder = EmbeddingGenerator(api_key=api_key, model=embedding_model)
    store = ChromaStore(persist_dir=persist_dir, collection_name=collection_name)

    # Step 2: Prepare data
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
        for c in chunks
    ]

    # Step 3: Generate embeddings
    logger.info(f"\n[1/2] Generating embeddings for {len(texts)} chunks...")
    embeddings = embedder.embed_texts(texts, batch_size=batch_size)
    logger.info(f"  Embedding dimension: {len(embeddings[0])}")
    logger.info(f"  Total tokens used: {embedder.total_tokens_used:,}")

    # Step 4: Store in ChromaDB
    logger.info(f"\n[2/2] Storing in ChromaDB...")
    added = store.add_chunks(
        chunk_ids=chunk_ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Step 5: Verify
    stats = store.get_stats()

    summary = {
        "chunks_processed": len(chunks),
        "chunks_stored": added,
        "embedding_model": embedding_model,
        "embedding_dimension": len(embeddings[0]),
        "total_tokens_used": embedder.total_tokens_used,
        "collection_stats": stats,
    }

    logger.info(f"\n{'='*60}")
    logger.info("✅ Vector store population complete!")
    logger.info(f"  Total documents in store: {stats['total_documents']}")
    logger.info(f"  Acts: {', '.join(stats['acts'])}")
    logger.info(f"  Categories: {', '.join(stats['categories'])}")
    logger.info(f"  Tokens used: {embedder.total_tokens_used:,}")
    logger.info(f"{'='*60}")

    return summary


def show_stats(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
):
    """Print vector store statistics."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
    except ImportError:
        console = None

    store = ChromaStore(persist_dir=persist_dir, collection_name=collection_name)
    stats = store.get_stats()

    if console:
        console.print(Panel(
            f"[bold]Vector Store Statistics[/bold]",
            border_style="blue",
        ))

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Collection", stats["collection_name"])
        table.add_row("Total Documents", str(stats["total_documents"]))
        table.add_row("Acts", ", ".join(stats["acts"]) or "None")
        table.add_row("Categories", ", ".join(stats["categories"]) or "None")
        table.add_row("Storage Path", stats.get("persist_dir", "N/A"))
        console.print(table)
    else:
        print(f"Collection: {stats['collection_name']}")
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Acts: {', '.join(stats['acts'])}")


def main():
    parser = argparse.ArgumentParser(description="AinSeba — Populate Vector Store")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", default=True, help="Populate with all chunks (default)")
    group.add_argument("--act", type=str, help="Populate with a specific act's chunks")
    group.add_argument("--file", type=str, help="Populate from a specific JSON file")
    group.add_argument("--stats", action="store_true", help="Show store statistics")
    group.add_argument("--reset", action="store_true", help="Reset collection and re-populate")

    parser.add_argument("--batch-size", type=int, default=100, help="Embedding batch size")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.stats:
        show_stats()
        return

    if args.reset:
        store = ChromaStore(
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        store.reset_collection()
        logger.info("Collection reset. Re-populating...")

    # Load chunks
    if args.file:
        chunks = load_chunks(source_path=args.file)
    elif args.act:
        chunks = load_chunks(act_id=args.act)
    else:
        chunks = load_chunks()

    if not chunks:
        logger.error("No chunks found. Run Phase 1 pipeline first.")
        sys.exit(1)

    # Populate
    populate_vectorstore(chunks, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
