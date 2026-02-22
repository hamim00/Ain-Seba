"""
AinSeba - Unit Tests for Phase 2 (Vector Store & Retrieval)

Run with: pytest tests/test_vectorstore.py -v

NOTE: Tests that require OpenAI API are marked with @pytest.mark.api
      and are skipped by default. To run them:
      OPENAI_API_KEY=sk-your-key pytest tests/test_vectorstore.py -v -m api
"""

import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever import LegalRetriever, RetrievalResult


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # On Windows, ChromaDB holds file locks — ignore cleanup errors
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def chroma_store(temp_chroma_dir):
    """Create a fresh ChromaStore for testing."""
    store = ChromaStore(
        persist_dir=temp_chroma_dir,
        collection_name="test_collection",
    )
    yield store
    # Close the client to release file locks on Windows
    del store.collection
    del store.client


@pytest.fixture
def sample_chunks():
    """Sample chunk data for testing."""
    return [
        {
            "chunk_id": "test_001",
            "text": "Section 1. Maximum working hours. No worker shall be required to work more than eight hours in any day.",
            "token_count": 25,
            "act_name": "Labour Act 2006",
            "act_id": "labour_act_2006",
            "chapter": "Chapter III",
            "section_number": "1",
            "section_title": "Maximum working hours",
            "category": "Employment",
            "year": 2006,
            "language": "english",
        },
        {
            "chunk_id": "test_002",
            "text": "Section 2. Payment of wages. Every employer shall pay wages before the seventh working day.",
            "token_count": 20,
            "act_name": "Labour Act 2006",
            "act_id": "labour_act_2006",
            "chapter": "Chapter IV",
            "section_number": "2",
            "section_title": "Payment of wages",
            "category": "Employment",
            "year": 2006,
            "language": "english",
        },
        {
            "chunk_id": "test_003",
            "text": "Section 42. Penalty for theft. Whoever commits theft shall be punished with imprisonment up to three years.",
            "token_count": 22,
            "act_name": "Penal Code 1860",
            "act_id": "penal_code_1860",
            "chapter": "Chapter XVII",
            "section_number": "42",
            "section_title": "Penalty for theft",
            "category": "Criminal Law",
            "year": 1860,
            "language": "english",
        },
        {
            "chunk_id": "test_004",
            "text": "Section 10. Consumer complaint mechanism. Any consumer may file a complaint against defective goods.",
            "token_count": 18,
            "act_name": "Consumer Rights Act 2009",
            "act_id": "consumer_rights_2009",
            "chapter": "Chapter V",
            "section_number": "10",
            "section_title": "Consumer complaint mechanism",
            "category": "Consumer Rights",
            "year": 2009,
            "language": "english",
        },
        {
            "chunk_id": "test_005",
            "text": "Section 5. Cyber offenses. Unauthorized access to computer systems shall be punishable with imprisonment.",
            "token_count": 17,
            "act_name": "Cyber Security Act 2023",
            "act_id": "cyber_security_2023",
            "chapter": "Chapter III",
            "section_number": "5",
            "section_title": "Cyber offenses",
            "category": "Cyber Law",
            "year": 2023,
            "language": "english",
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Fake embeddings (small dimension for testing)."""
    import random
    random.seed(42)
    # 5 embeddings of dimension 128 (small for tests)
    return [[random.random() for _ in range(128)] for _ in range(5)]


@pytest.fixture
def populated_store(chroma_store, sample_chunks, sample_embeddings):
    """ChromaStore populated with sample data."""
    chunk_ids = [c["chunk_id"] for c in sample_chunks]
    texts = [c["text"] for c in sample_chunks]
    metadatas = [
        {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
        for c in sample_chunks
    ]

    chroma_store.add_chunks(
        chunk_ids=chunk_ids,
        texts=texts,
        embeddings=sample_embeddings,
        metadatas=metadatas,
    )
    return chroma_store


# ============================================
# Test: ChromaStore
# ============================================

class TestChromaStore:
    """Tests for the ChromaDB wrapper."""

    def test_create_empty_store(self, chroma_store):
        assert chroma_store.collection.count() == 0

    def test_add_chunks(self, chroma_store, sample_chunks, sample_embeddings):
        chunk_ids = [c["chunk_id"] for c in sample_chunks]
        texts = [c["text"] for c in sample_chunks]
        metadatas = [
            {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
            for c in sample_chunks
        ]

        added = chroma_store.add_chunks(
            chunk_ids=chunk_ids,
            texts=texts,
            embeddings=sample_embeddings,
            metadatas=metadatas,
        )
        assert added == 5
        assert chroma_store.collection.count() == 5

    def test_upsert_deduplication(self, populated_store, sample_chunks, sample_embeddings):
        """Upserting same IDs should not create duplicates."""
        chunk_ids = [c["chunk_id"] for c in sample_chunks]
        texts = [c["text"] for c in sample_chunks]
        metadatas = [
            {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
            for c in sample_chunks
        ]

        populated_store.add_chunks(
            chunk_ids=chunk_ids,
            texts=texts,
            embeddings=sample_embeddings,
            metadatas=metadatas,
        )
        # Count should remain 5, not 10
        assert populated_store.collection.count() == 5

    def test_query_returns_results(self, populated_store, sample_embeddings):
        results = populated_store.query(
            query_embedding=sample_embeddings[0],
            top_k=3,
        )
        assert len(results["ids"][0]) <= 3
        assert len(results["documents"][0]) <= 3
        assert len(results["metadatas"][0]) <= 3

    def test_query_with_metadata_filter(self, populated_store, sample_embeddings):
        results = populated_store.query(
            query_embedding=sample_embeddings[0],
            top_k=10,
            where={"act_id": "labour_act_2006"},
        )
        # Should only return Labour Act chunks
        for meta in results["metadatas"][0]:
            assert meta["act_id"] == "labour_act_2006"

    def test_query_with_category_filter(self, populated_store, sample_embeddings):
        results = populated_store.query(
            query_embedding=sample_embeddings[0],
            top_k=10,
            where={"category": "Criminal Law"},
        )
        for meta in results["metadatas"][0]:
            assert meta["category"] == "Criminal Law"

    def test_get_stats(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["total_documents"] == 5
        assert "labour_act_2006" in stats["acts"]
        assert "penal_code_1860" in stats["acts"]
        assert "Employment" in stats["categories"]

    def test_get_chunks_by_act(self, populated_store):
        chunks = populated_store.get_chunks_by_act("labour_act_2006")
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk["metadata"]["act_id"] == "labour_act_2006"

    def test_delete_act(self, populated_store):
        deleted = populated_store.delete_act("penal_code_1860")
        assert deleted == 1
        assert populated_store.collection.count() == 4

    def test_reset_collection(self, populated_store):
        populated_store.reset_collection()
        assert populated_store.collection.count() == 0

    def test_format_results(self, populated_store, sample_embeddings):
        raw = populated_store.query(
            query_embedding=sample_embeddings[0],
            top_k=3,
        )
        formatted = populated_store._format_results(raw)
        assert len(formatted) <= 3
        for r in formatted:
            assert "chunk_id" in r
            assert "text" in r
            assert "metadata" in r
            assert "similarity_score" in r
            assert 0 <= r["similarity_score"] <= 1

    def test_empty_query_results(self, chroma_store):
        """Query on empty store should return empty results."""
        dummy_embedding = [0.0] * 128
        results = chroma_store.query(
            query_embedding=dummy_embedding,
            top_k=5,
        )
        assert len(results["ids"][0]) == 0


# ============================================
# Test: LegalRetriever (with mocked embedding)
# ============================================

class TestLegalRetriever:
    """Tests for the LegalRetriever with mocked components."""

    def test_retrieve_returns_results(self, populated_store, sample_embeddings):
        # Mock the embedding generator
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = sample_embeddings[0]

        retriever = LegalRetriever(
            chroma_store=populated_store,
            embedding_generator=mock_embedder,
            reranker=None,
            top_k=5,
        )

        results = retriever.retrieve(query="working hours")
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_with_act_filter(self, populated_store, sample_embeddings):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = sample_embeddings[0]

        retriever = LegalRetriever(
            chroma_store=populated_store,
            embedding_generator=mock_embedder,
            reranker=None,
        )

        results = retriever.retrieve(
            query="wages",
            act_id="labour_act_2006",
        )
        for r in results:
            assert r.act_id == "labour_act_2006"

    def test_retrieve_with_category_filter(self, populated_store, sample_embeddings):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = sample_embeddings[0]

        retriever = LegalRetriever(
            chroma_store=populated_store,
            embedding_generator=mock_embedder,
            reranker=None,
        )

        results = retriever.retrieve(
            query="punishment",
            category="Criminal Law",
        )
        for r in results:
            assert r.category == "Criminal Law"

    def test_similarity_search_no_reranker(self, populated_store, sample_embeddings):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = sample_embeddings[0]

        retriever = LegalRetriever(
            chroma_store=populated_store,
            embedding_generator=mock_embedder,
            reranker=None,
        )

        results = retriever.similarity_search(query="working hours", top_k=3)
        assert len(results) <= 3

    def test_citation_format(self, populated_store, sample_embeddings):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = sample_embeddings[0]

        retriever = LegalRetriever(
            chroma_store=populated_store,
            embedding_generator=mock_embedder,
            reranker=None,
        )

        results = retriever.retrieve(query="test")
        if results:
            citation = results[0].citation
            assert results[0].act_name in citation

    def test_search_by_section(self, populated_store):
        retriever = LegalRetriever(
            chroma_store=populated_store,
            embedding_generator=MagicMock(),
            reranker=None,
        )

        results = retriever.search_by_section(
            act_id="labour_act_2006",
            section_number="1",
        )
        assert len(results) == 1
        assert results[0].section_number == "1"

    def test_build_where_filter_single(self):
        retriever = LegalRetriever(
            chroma_store=MagicMock(),
            embedding_generator=MagicMock(),
        )

        filter_ = retriever._build_where_filter(act_id="test_act")
        assert filter_ == {"act_id": "test_act"}

    def test_build_where_filter_multiple(self):
        retriever = LegalRetriever(
            chroma_store=MagicMock(),
            embedding_generator=MagicMock(),
        )

        filter_ = retriever._build_where_filter(
            act_id="test_act",
            category="Employment",
        )
        assert "$and" in filter_
        assert len(filter_["$and"]) == 2

    def test_build_where_filter_none(self):
        retriever = LegalRetriever(
            chroma_store=MagicMock(),
            embedding_generator=MagicMock(),
        )

        filter_ = retriever._build_where_filter()
        assert filter_ is None


# ============================================
# Test: RetrievalResult
# ============================================

class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_citation_full(self):
        r = RetrievalResult(
            chunk_id="test",
            text="test text",
            act_name="Labour Act 2006",
            act_id="labour_act",
            section_number="42",
            section_title="Working Hours",
            chapter="Chapter III",
            category="Employment",
            year=2006,
            similarity_score=0.85,
        )
        citation = r.citation
        assert "Labour Act 2006" in citation
        assert "Section 42" in citation
        assert "Working Hours" in citation
        assert "Chapter III" in citation

    def test_citation_minimal(self):
        r = RetrievalResult(
            chunk_id="test",
            text="test",
            act_name="Test Act",
            act_id="test",
            section_number="",
            section_title="",
            chapter="",
            category="",
            year=0,
            similarity_score=0.5,
        )
        assert r.citation == "Test Act"


# ============================================
# Test: Reranker (mocked)
# ============================================

class TestReranker:
    """Tests for the CrossEncoderReranker (with mocked model)."""

    def test_rerank_sorts_by_score(self):
        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "test"

        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        reranker._model = mock_model

        results = [
            {"text": "doc A", "chunk_id": "a"},
            {"text": "doc B", "chunk_id": "b"},
            {"text": "doc C", "chunk_id": "c"},
        ]

        reranked = reranker.rerank("query", results)
        assert reranked[0]["chunk_id"] == "b"  # Highest score
        assert reranked[1]["chunk_id"] == "c"
        assert reranked[2]["chunk_id"] == "a"  # Lowest score

    def test_rerank_top_n(self):
        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "test"

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        reranker._model = mock_model

        results = [
            {"text": "doc A", "chunk_id": "a"},
            {"text": "doc B", "chunk_id": "b"},
            {"text": "doc C", "chunk_id": "c"},
        ]

        reranked = reranker.rerank("query", results, top_n=2)
        assert len(reranked) == 2

    def test_rerank_threshold(self):
        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "test"

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        reranker._model = mock_model

        results = [
            {"text": "doc A", "chunk_id": "a"},
            {"text": "doc B", "chunk_id": "b"},
            {"text": "doc C", "chunk_id": "c"},
        ]

        reranked = reranker.rerank("query", results, score_threshold=0.4)
        assert len(reranked) == 2  # Only scores >= 0.4

    def test_rerank_empty_input(self):
        from src.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "test"

        assert reranker.rerank("query", []) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
