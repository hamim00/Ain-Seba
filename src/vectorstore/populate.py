"""
AinSeba - Vector Store Population Script
Loads Phase 1 processed chunks, generates embeddings, and stores in ChromaDB.

Usage:
    python -m src.vectorstore.populate                       # All chunks (default)
    python -m src.vectorstore.populate --act labour_act_2006  # Single act
    python -m src.vectorstore.populate --file path/to.json    # Specific file
    python -m src.vectorstore.populate --reset                # Reset and re-populate
    python -m src.vectorstore.populate --stats                # Show store statistics
"""

import json
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

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


# ---------------------------------------------------------------------
# Safety splitter to avoid embedding context-length errors (8192 tokens)
# ---------------------------------------------------------------------
def split_large_text(text: str, max_chars: int = 12000) -> List[str]:
    """
    Split a long text into smaller parts, attempting paragraph-based splits first.
    This is a character-based proxy to keep token count comfortably below 8192.

    max_chars=12000 is a conservative default for legal text.
    """
    if not text:
        return []

    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    buf = ""

    # Prefer splitting on double-newlines (paragraphs)
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(buf) + len(para) + 2 <= max_chars:
            buf = (buf + "\n\n" + para) if buf else para
            continue

        # flush buffer
        if buf:
            parts.append(buf.strip())
            buf = ""

        # if the paragraph itself fits, start new buffer
        if len(para) <= max_chars:
            buf = para
        else:
            # hard cut a giant paragraph
            for i in range(0, len(para), max_chars):
                chunk = para[i : i + max_chars].strip()
                if chunk:
                    parts.append(chunk)

    if buf:
        parts.append(buf.strip())

    # Final cleanup
    return [p for p in parts if p]


def normalize_and_split_chunks(
    chunks: List[Dict],
    max_chars: int = 12000
) -> Tuple[List[str], List[str], List[Dict], int]:
    """
    Convert incoming Phase-1 chunks into safe (id, text, metadata) arrays,
    splitting any oversized chunk into subchunks.

    Returns:
        chunk_ids, texts, metadatas, num_splits
    """
    chunk_ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict] = []

    num_splits = 0

    for c in chunks:
        base_id = c.get("chunk_id")
        text = c.get("text", "")

        if not base_id:
            # Fallback if chunk_id missing
            base_id = f"chunk_{len(chunk_ids)}"

        # metadata = everything except chunk_id + text
        meta = {k: v for k, v in c.items() if k not in ("text", "chunk_id")}

        parts = split_large_text(text, max_chars=max_chars)

        if len(parts) <= 1:
            chunk_ids.append(base_id)
            texts.append(parts[0] if parts else "")
            metadatas.append(meta)
        else:
            num_splits += 1
            for idx, part in enumerate(parts):
                chunk_ids.append(f"{base_id}__{idx}")
                texts.append(part)
                metadatas.append({**meta, "subchunk": idx, "subchunk_total": len(parts)})

    # Remove any accidental empties
    filtered = [(i, t, m) for i, t, m in zip(chunk_ids, texts, metadatas) if t and t.strip()]
    chunk_ids = [x[0] for x in filtered]
    texts = [x[1] for x in filtered]
    metadatas = [x[2] for x in filtered]

    return chunk_ids, texts, metadatas, num_splits


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_chunks(source_path: Path = None, act_id: str = None) -> List[Dict]:
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
        combined = PROCESSED_DATA_DIR / "all_chunks_combined.json"
        if combined.exists():
            path = combined
        else:
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


# ---------------------------------------------------------------------
# Populate
# ---------------------------------------------------------------------
def populate_vectorstore(
    chunks: List[Dict],
    api_key: str = OPENAI_API_KEY,
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = 100,
    max_chars_per_embedding: int = 12000,
) -> Dict:
    """
    Embed chunks and store in ChromaDB.

    Key improvement: splits oversized texts to prevent OpenAI embedding context errors.

    Returns:
        Summary dict with stats.
    """
    logger.info(f"\n{'='*60}")
    logger.info("AinSeba — Vector Store Population")
    logger.info(f"{'='*60}")
    logger.info(f"Raw chunks loaded: {len(chunks)}")
    logger.info(f"Embedding model: {embedding_model}")
    logger.info(f"ChromaDB dir: {persist_dir}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Max chars per embedding input: {max_chars_per_embedding}")

    # Step 1: Initialize components
    embedder = EmbeddingGenerator(api_key=api_key, model=embedding_model)
    store = ChromaStore(persist_dir=persist_dir, collection_name=collection_name)

    # Step 2: Prepare + split data safely
    chunk_ids, texts, metadatas, num_splits = normalize_and_split_chunks(
        chunks, max_chars=max_chars_per_embedding
    )

    logger.info(f"Prepared texts to embed: {len(texts)} (split source chunks: {num_splits})")

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
        "raw_chunks_loaded": len(chunks),
        "texts_embedded": len(texts),
        "source_chunks_split": num_splits,
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


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AinSeba — Populate Vector Store")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", default=True, help="Populate with all chunks (default)")
    group.add_argument("--act", type=str, help="Populate with a specific act's chunks")
    group.add_argument("--file", type=str, help="Populate from a specific JSON file")
    group.add_argument("--stats", action="store_true", help="Show store statistics")
    group.add_argument("--reset", action="store_true", help="Reset collection and re-populate")

    parser.add_argument("--batch-size", type=int, default=100, help="Embedding batch size")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max characters per embedding input")
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
    populate_vectorstore(
        chunks,
        batch_size=args.batch_size,
        max_chars_per_embedding=args.max_chars,
    )


if __name__ == "__main__":
    main()