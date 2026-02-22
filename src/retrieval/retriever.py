"""
AinSeba - Legal Document Retriever
Combines vector similarity search, metadata filtering, and cross-encoder reranking
into a single retrieval pipeline.

This is the main interface for searching Bangladesh law documents.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with all scores and metadata."""
    chunk_id: str
    text: str
    act_name: str
    act_id: str
    section_number: str
    section_title: str
    chapter: str
    category: str
    year: int
    similarity_score: float      # Cosine similarity (0-1)
    rerank_score: float = 0.0    # Cross-encoder score
    distance: float = 0.0        # Cosine distance
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Generate a human-readable citation string."""
        parts = [self.act_name]
        if self.chapter:
            parts.append(self.chapter)
        if self.section_number:
            parts.append(f"Section {self.section_number}")
            if self.section_title:
                parts.append(f"({self.section_title})")
        return ", ".join(parts)


class LegalRetriever:
    """
    Full retrieval pipeline for AinSeba.
    
    Pipeline:
    1. Embed the query using OpenAI text-embedding-3-small
    2. Vector search in ChromaDB (top_k candidates) with optional metadata filters
    3. Cross-encoder reranking for precision (top_n final results)
    
    Supports metadata filtering by:
    - act_id / act_name — search within a specific law
    - category — search by legal category (Employment, Criminal, etc.)
    - year — search by year range
    - section_number — search for a specific section
    """

    def __init__(
        self,
        chroma_store,
        embedding_generator,
        reranker=None,
        top_k: int = 10,
        rerank_top_n: int = 5,
    ):
        """
        Args:
            chroma_store: ChromaStore instance.
            embedding_generator: EmbeddingGenerator instance.
            reranker: Optional CrossEncoderReranker instance.
            top_k: Number of candidates from vector search.
            rerank_top_n: Number of final results after reranking.
        """
        self.store = chroma_store
        self.embedder = embedding_generator
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        act_id: Optional[str] = None,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        use_reranker: bool = True,
    ) -> list[RetrievalResult]:
        """
        Full retrieval pipeline: embed → search → (optional) rerank.

        Args:
            query: Natural language legal question.
            top_k: Override default top_k for vector search.
            rerank_top_n: Override default top_n for reranking.
            act_id: Filter by specific act (e.g., "labour_act_2006").
            category: Filter by category (e.g., "Employment").
            year_min: Filter by minimum year.
            year_max: Filter by maximum year.
            use_reranker: Whether to apply cross-encoder reranking.

        Returns:
            List of RetrievalResult objects, ranked by relevance.
        """
        top_k = top_k or self.top_k
        rerank_top_n = rerank_top_n or self.rerank_top_n

        logger.info(f"Retrieving for: '{query[:80]}...' (top_k={top_k})")

        # Step 1: Build metadata filter
        where_filter = self._build_where_filter(
            act_id=act_id,
            category=category,
            year_min=year_min,
            year_max=year_max,
        )

        if where_filter:
            logger.info(f"  Metadata filter: {where_filter}")

        # Step 2: Vector search
        raw_results = self.store.query_with_text(
            query_text=query,
            embedding_fn=self.embedder.embed_query,
            top_k=top_k,
            where=where_filter,
        )

        logger.info(f"  Vector search returned {len(raw_results)} results")

        if not raw_results:
            logger.warning("  No results found.")
            return []

        # Step 3: Rerank (if enabled and reranker is available)
        if use_reranker and self.reranker:
            logger.info(f"  Reranking with cross-encoder...")
            raw_results = self.reranker.rerank(
                query=query,
                results=raw_results,
                top_n=rerank_top_n,
            )
            logger.info(f"  Reranked to {len(raw_results)} results")

        # Step 4: Convert to RetrievalResult objects
        results = []
        for r in raw_results:
            meta = r.get("metadata", {})
            results.append(RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                act_name=meta.get("act_name", "Unknown"),
                act_id=meta.get("act_id", ""),
                section_number=str(meta.get("section_number", "")),
                section_title=meta.get("section_title", ""),
                chapter=meta.get("chapter", ""),
                category=meta.get("category", ""),
                year=int(meta.get("year", 0)),
                similarity_score=r.get("similarity_score", 0.0),
                rerank_score=r.get("rerank_score", 0.0),
                distance=r.get("distance", 0.0),
                metadata=meta,
            ))

        return results

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        **filter_kwargs,
    ) -> list[RetrievalResult]:
        """
        Simple similarity search WITHOUT reranking.
        Faster but less precise. Good for quick lookups.
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            use_reranker=False,
            **filter_kwargs,
        )

    def search_by_section(
        self,
        act_id: str,
        section_number: str,
    ) -> list[RetrievalResult]:
        """
        Direct lookup of a specific section (no embedding needed).
        Uses ChromaDB metadata filtering.
        """
        results = self.store.collection.get(
            where={
                "$and": [
                    {"act_id": act_id},
                    {"section_number": section_number},
                ]
            },
            include=["documents", "metadatas"],
        )

        retrieval_results = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            retrieval_results.append(RetrievalResult(
                chunk_id=doc_id,
                text=results["documents"][i],
                act_name=meta.get("act_name", "Unknown"),
                act_id=meta.get("act_id", ""),
                section_number=str(meta.get("section_number", "")),
                section_title=meta.get("section_title", ""),
                chapter=meta.get("chapter", ""),
                category=meta.get("category", ""),
                year=int(meta.get("year", 0)),
                similarity_score=1.0,  # Direct match
                metadata=meta,
            ))

        return retrieval_results

    def _build_where_filter(
        self,
        act_id: Optional[str] = None,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> Optional[dict]:
        """Build a ChromaDB where filter from optional parameters."""
        conditions = []

        if act_id:
            conditions.append({"act_id": act_id})
        if category:
            conditions.append({"category": category})
        if year_min:
            conditions.append({"year": {"$gte": year_min}})
        if year_max:
            conditions.append({"year": {"$lte": year_max}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}


def format_results_table(results: list[RetrievalResult]) -> str:
    """Format retrieval results as a readable string (for CLI/logging)."""
    if not results:
        return "No results found."

    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"Found {len(results)} results:")
    lines.append(f"{'='*80}")

    for i, r in enumerate(results, 1):
        lines.append(f"\n--- Result {i} ---")
        lines.append(f"  Citation:   {r.citation}")
        lines.append(f"  Similarity: {r.similarity_score:.4f}")
        if r.rerank_score:
            lines.append(f"  Rerank:     {r.rerank_score:.4f}")
        lines.append(f"  Act:        {r.act_name} ({r.year})")
        lines.append(f"  Category:   {r.category}")
        lines.append(f"  Text:       {r.text[:200]}...")

    return "\n".join(lines)


def print_results(results: list[RetrievalResult]) -> None:
    """Pretty-print retrieval results using rich."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")

        for i, r in enumerate(results, 1):
            # Score bar
            sim_bar = "█" * int(r.similarity_score * 20)
            rerank_info = f" | Rerank: {r.rerank_score:.3f}" if r.rerank_score else ""

            console.print(Panel(
                f"[bold]{r.citation}[/bold]\n\n"
                f"{r.text[:300]}{'...' if len(r.text) > 300 else ''}\n\n"
                f"[dim]Similarity: {r.similarity_score:.4f} {sim_bar}{rerank_info}[/dim]",
                title=f"Result {i}",
                border_style="green" if r.similarity_score > 0.7 else "yellow",
            ))

    except ImportError:
        print(format_results_table(results))
