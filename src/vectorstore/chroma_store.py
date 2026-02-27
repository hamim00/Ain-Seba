"""
AinSeba - ChromaDB Vector Store
Manages the ChromaDB collection for storing and querying law document embeddings.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Metadata fields that ChromaDB can filter on
FILTERABLE_METADATA_FIELDS = [
    "act_name",
    "act_id",
    "part",
    "chapter",
    "section_number",
    "section_title",
    "category",
    "year",
    "language",
]


class ChromaStore:
    """
    ChromaDB wrapper for AinSeba law document embeddings.

    Features:
    - Persistent storage (survives restarts)
    - Metadata-based filtering (by act, chapter, section, category, year)
    - Batch upsert with deduplication via chunk_id
    - Collection statistics and inspection utilities

    NOTE:
    - We provide embeddings ourselves (OpenAI).
    - Therefore we explicitly set embedding_function=None to prevent Chroma
      from loading its DefaultEmbeddingFunction (which pulls onnxruntime).
    """

    def __init__(
        self,
        persist_dir: str | Path = "chroma_db",
        collection_name: str = "ainseba_laws",
    ):
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_dir: Directory for ChromaDB data persistence.
            collection_name: Name of the collection to use.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(f"Initializing ChromaDB at: {self.persist_dir}")

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # ✅ Critical fix: prevent default embedding function from loading (onnxruntime)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )

        logger.info(
            f"Collection '{collection_name}' ready. "
            f"Current document count: {self.collection.count()}"
        )

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        batch_size: int = 500,
    ) -> int:
        """
        Add chunks with embeddings and metadata to ChromaDB.
        """
        if not chunk_ids:
            logger.warning("No chunks to add.")
            return 0

        assert len(chunk_ids) == len(texts) == len(embeddings) == len(metadatas), (
            "All input lists must have the same length."
        )

        clean_metadatas = [self._clean_metadata(m) for m in metadatas]

        total_added = 0

        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_metadatas = clean_metadatas[i : i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )

            total_added += len(batch_ids)
            logger.info(f"  Upserted batch: {total_added}/{len(chunk_ids)}")

        logger.info(
            f"Added {total_added} chunks to collection '{self.collection_name}'. "
            f"Total count: {self.collection.count()}"
        )
        return total_added

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> dict:
        """
        Query the vector store for similar chunks.
        """
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count() or top_k),
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            query_kwargs["where"] = where
        if where_document:
            query_kwargs["where_document"] = where_document

        results = self.collection.query(**query_kwargs)
        return results

    def query_with_text(
        self,
        query_text: str,
        embedding_fn,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        High-level query: embed text, search, return formatted results.
        """
        query_embedding = embedding_fn(query_text)
        raw_results = self.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )
        return self.format_results(raw_results)

    def format_results(self, raw_results: dict) -> list[dict]:
        """Public formatter for raw ChromaDB query results."""
        return self._format_results(raw_results)

    def get_stats(self) -> dict:
        """Get statistics about the collection."""
        count = self.collection.count()

        if count == 0:
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "acts": [],
                "categories": [],
            }

        sample_size = min(count, 1000)
        sample = self.collection.get(
            limit=sample_size,
            include=["metadatas"],
        )

        acts = set()
        categories = set()
        years = set()

        for meta in sample["metadatas"]:
            if meta.get("act_id"):
                acts.add(meta["act_id"])
            if meta.get("category"):
                categories.add(meta["category"])
            if meta.get("year"):
                years.add(meta["year"])

        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "acts": sorted(acts),
            "categories": sorted(categories),
            "years": sorted(years),
            "persist_dir": str(self.persist_dir),
        }

    def get_chunks_by_act(self, act_id: str, limit: int = 100) -> list[dict]:
        """Retrieve all chunks for a specific act."""
        results = self.collection.get(
            where={"act_id": act_id},
            limit=limit,
            include=["documents", "metadatas"],
        )

        chunks = []
        for i, doc_id in enumerate(results["ids"]):
            chunks.append(
                {
                    "chunk_id": doc_id,
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )

        return chunks

    def delete_act(self, act_id: str) -> int:
        """Delete all chunks for a specific act (useful for re-ingestion)."""
        results = self.collection.get(
            where={"act_id": act_id},
            include=[],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for act_id='{act_id}'")
            return len(results["ids"])

        return 0

    def reset_collection(self) -> None:
        """Delete and recreate the collection (WARNING: destroys all data)."""
        logger.warning(f"Resetting collection '{self.collection_name}'!")
        self.client.delete_collection(self.collection_name)

        # ✅ Critical fix here too
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset complete.")

    def _clean_metadata(self, metadata: dict) -> dict:
        """
        Clean metadata for ChromaDB compatibility.
        ChromaDB only supports: str, int, float, bool.
        """
        clean = {}
        for key, value in metadata.items():
            if key in FILTERABLE_METADATA_FIELDS:
                if isinstance(value, (str, int, float, bool)):
                    clean[key] = value
                elif isinstance(value, list):
                    clean[key] = ",".join(str(v) for v in value)
                elif value is not None:
                    clean[key] = str(value)
        return clean

    def _format_results(self, raw_results: dict) -> list[dict]:
        """Format raw ChromaDB results into clean dicts."""
        if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
            return []

        results = []
        for i in range(len(raw_results["ids"][0])):
            distance = raw_results["distances"][0][i]
            similarity = 1 - distance

            results.append(
                {
                    "chunk_id": raw_results["ids"][0][i],
                    "text": raw_results["documents"][0][i],
                    "metadata": raw_results["metadatas"][0][i],
                    "distance": round(distance, 4),
                    "similarity_score": round(similarity, 4),
                }
            )

        return results