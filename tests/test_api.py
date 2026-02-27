"""
AinSeba - Unit Tests for Phase 5 (FastAPI Backend)

Run with: pytest tests/test_api.py -v

All tests use mocked chains and TestClient — no API key needed.
"""

import sys
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.rate_limiter import RateLimiter
from src.models.schemas import (
    QueryRequest,
    QueryResponse,
    SourceInfo,
    HealthResponse,
    SourceListResponse,
    FeedbackRequest,
    FeedbackResponse,
    ErrorResponse,
)


# ============================================
# Test: Pydantic Models
# ============================================

class TestQueryRequest:
    """Tests for QueryRequest validation."""

    def test_valid_request(self):
        req = QueryRequest(question="What is the penalty for theft?")
        assert req.question == "What is the penalty for theft?"
        assert req.language is None
        assert req.use_reranker is True

    def test_minimal_request(self):
        req = QueryRequest(question="abc")
        assert len(req.question) == 3

    def test_too_short_question(self):
        with pytest.raises(Exception):
            QueryRequest(question="ab")

    def test_with_all_fields(self):
        req = QueryRequest(
            question="Test question",
            session_id="s123",
            language="bn",
            act_id="labour_act_2006",
            category="Employment",
            use_reranker=False,
        )
        assert req.session_id == "s123"
        assert req.language == "bn"
        assert req.act_id == "labour_act_2006"
        assert req.use_reranker is False

    def test_invalid_language(self):
        with pytest.raises(Exception):
            QueryRequest(question="Test", language="fr")

    def test_valid_languages(self):
        for lang in ["en", "bn", "auto"]:
            req = QueryRequest(question="Test", language=lang)
            assert req.language == lang


class TestQueryResponse:
    """Tests for QueryResponse model."""

    def test_basic_response(self):
        resp = QueryResponse(
            answer="According to Section 42...",
            sources=[],
        )
        assert resp.answer == "According to Section 42..."
        assert resp.detected_language == "en"
        assert resp.was_translated is False

    def test_response_with_sources(self):
        resp = QueryResponse(
            answer="Test",
            sources=[
                SourceInfo(citation="Labour Act, Section 42", similarity_score=0.89),
                SourceInfo(citation="Penal Code, Section 379", similarity_score=0.75),
            ],
        )
        assert len(resp.sources) == 2
        assert resp.sources[0].similarity_score == 0.89

    def test_bilingual_response(self):
        resp = QueryResponse(
            answer="Bangla answer",
            answer_english="English answer",
            detected_language="bn",
            response_language="bn",
            was_translated=True,
        )
        assert resp.was_translated is True
        assert resp.detected_language == "bn"

    def test_has_timestamp(self):
        resp = QueryResponse(answer="Test")
        assert resp.timestamp is not None
        assert len(resp.timestamp) > 0


class TestSourceInfo:
    """Tests for SourceInfo model."""

    def test_source_info(self):
        src = SourceInfo(
            citation="Labour Act, Section 42",
            act_name="Labour Act 2006",
            section_number="42",
            similarity_score=0.89,
        )
        assert src.citation == "Labour Act, Section 42"
        assert src.similarity_score == 0.89

    def test_default_values(self):
        src = SourceInfo(citation="Test")
        assert src.act_name == ""
        assert src.similarity_score == 0.0


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response(self):
        resp = HealthResponse(
            status="ok",
            vector_store_documents=100,
            vector_store_acts=["labour_act_2006"],
            model="gpt-4o-mini",
        )
        assert resp.status == "ok"
        assert resp.vector_store_documents == 100


class TestFeedbackRequest:
    """Tests for FeedbackRequest validation."""

    def test_valid_feedback(self):
        fb = FeedbackRequest(
            query="Test question",
            answer="Test answer",
            rating=5,
            comment="Great!",
        )
        assert fb.rating == 5

    def test_rating_range(self):
        with pytest.raises(Exception):
            FeedbackRequest(query="q", answer="a", rating=0)

        with pytest.raises(Exception):
            FeedbackRequest(query="q", answer="a", rating=6)

    def test_valid_ratings(self):
        for r in [1, 2, 3, 4, 5]:
            fb = FeedbackRequest(query="q", answer="a", rating=r)
            assert fb.rating == r


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_error_response(self):
        err = ErrorResponse(error="Something went wrong", status_code=500)
        assert err.error == "Something went wrong"
        assert err.status_code == 500


# ============================================
# Test: Rate Limiter
# ============================================

class TestRateLimiter:
    """Tests for the rate limiter."""

    def test_allows_within_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("client1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False

    def test_separate_clients(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False
        assert limiter.is_allowed("client2") is True

    def test_remaining_count(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.remaining("client1") == 5
        limiter.is_allowed("client1")
        assert limiter.remaining("client1") == 4

    def test_reset_client(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False
        limiter.reset("client1")
        assert limiter.is_allowed("client1") is True

    def test_reset_all(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.is_allowed("c1")
        limiter.is_allowed("c2")
        limiter.reset()
        assert limiter.is_allowed("c1") is True
        assert limiter.is_allowed("c2") is True

    def test_window_expiry(self):
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False
        time.sleep(1.1)  # Wait for window to expire
        assert limiter.is_allowed("client1") is True


# ============================================
# Test: FastAPI Endpoints (TestClient)
# ============================================

class TestAPIEndpoints:
    """Tests for FastAPI endpoints using TestClient with mocked chain."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Set up TestClient with mocked chain."""
        from fastapi.testclient import TestClient

        # Mock the bilingual chain
        mock_bilingual = MagicMock()

        # Mock query response
        mock_response = MagicMock()
        mock_response.answer = "According to Section 42, max 8 hours."
        mock_response.answer_english = "According to Section 42, max 8 hours."
        mock_response.sources = [
            {
                "citation": "Labour Act, Section 42",
                "act_name": "Labour Act 2006",
                "act_id": "labour_act_2006",
                "section_number": "42",
                "section_title": "Working hours",
                "chapter": "Chapter III",
                "similarity_score": 0.89,
                "rerank_score": 2.5,
            }
        ]
        mock_response.query_original = "working hours"
        mock_response.query_english = "working hours"
        mock_response.detected_language = "en"
        mock_response.response_language = "en"
        mock_response.was_translated = False
        mock_response.session_id = "default"
        mock_response.model = "gpt-4o-mini"
        mock_bilingual.query.return_value = mock_response

        # Mock RAG chain with retriever and store
        mock_rag = MagicMock()
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "total_documents": 100,
            "acts": ["labour_act_2006"],
        }
        mock_rag.retriever.store = mock_store
        mock_rag.get_conversation_history.return_value = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        mock_bilingual.rag_chain = mock_rag

        # Patch the chain loader and API key
        import src.api.app as app_module
        app_module._bilingual_chain = mock_bilingual
        app_module._rate_limiter.reset()

        # Mock the API key check
        self._api_key_patch = patch.object(app_module, 'OPENAI_API_KEY', 'test-key')
        self._api_key_patch.start()

        self.client = TestClient(app_module.app)
        self.mock_bilingual = mock_bilingual

    def teardown_method(self):
        self._api_key_patch.stop()

    # --- Health ---
    def test_health_check(self):
        resp = self.client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["vector_store_documents"] == 100
        assert "labour_act_2006" in data["vector_store_acts"]

    # --- Sources ---
    def test_list_sources(self):
        resp = self.client.get("/api/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 100
        assert len(data["acts"]) > 0
        assert len(data["categories"]) > 0
        # Check that acts have expected fields
        act = data["acts"][0]
        assert "id" in act
        assert "name" in act
        assert "category" in act

    # --- Query ---
    def test_query_basic(self):
        resp = self.client.post("/api/query", json={
            "question": "What are the working hours?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["sources"]) > 0
        assert data["sources"][0]["citation"] == "Labour Act, Section 42"

    def test_query_with_language(self):
        resp = self.client.post("/api/query", json={
            "question": "What is the penalty for theft?",
            "language": "bn",
        })
        assert resp.status_code == 200

    def test_query_with_filters(self):
        resp = self.client.post("/api/query", json={
            "question": "Working hours",
            "act_id": "labour_act_2006",
            "category": "Employment",
        })
        assert resp.status_code == 200
        self.mock_bilingual.query.assert_called_once()

    def test_query_with_session(self):
        resp = self.client.post("/api/query", json={
            "question": "Follow up question",
            "session_id": "session123",
        })
        assert resp.status_code == 200
        call_kwargs = self.mock_bilingual.query.call_args[1]
        assert call_kwargs["session_id"] == "session123"

    def test_query_validation_too_short(self):
        resp = self.client.post("/api/query", json={
            "question": "ab",
        })
        assert resp.status_code == 422  # Validation error

    def test_query_validation_invalid_language(self):
        resp = self.client.post("/api/query", json={
            "question": "Test question",
            "language": "xyz",
        })
        assert resp.status_code == 422

    # --- Session ---
    def test_get_session(self):
        resp = self.client.get("/api/session/test_session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "test_session"
        assert len(data["messages"]) == 2

    # --- Feedback ---
    def test_submit_feedback(self):
        resp = self.client.post("/api/feedback", json={
            "query": "Test question",
            "answer": "Test answer",
            "rating": 5,
            "comment": "Great answer!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "received"

    def test_feedback_invalid_rating(self):
        resp = self.client.post("/api/feedback", json={
            "query": "Test",
            "answer": "Test",
            "rating": 0,
        })
        assert resp.status_code == 422

    # --- Rate Limiting ---
    def test_rate_limiting(self):
        import src.api.app as app_module
        app_module._rate_limiter = RateLimiter(max_requests=2, window_seconds=60)

        # First 2 should work
        resp1 = self.client.post("/api/query", json={"question": "Test one"})
        resp2 = self.client.post("/api/query", json={"question": "Test two"})
        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # 3rd should be rate limited
        resp3 = self.client.post("/api/query", json={"question": "Test three"})
        assert resp3.status_code == 429

    # --- Swagger Docs ---
    def test_swagger_docs_available(self):
        resp = self.client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema(self):
        resp = self.client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "AinSeba API"
        assert "/api/query" in schema["paths"]
        assert "/api/health" in schema["paths"]
        assert "/api/sources" in schema["paths"]
        assert "/api/feedback" in schema["paths"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
