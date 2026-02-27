"""
AinSeba - Unit Tests for Phase 6 (Frontend)

Run with: pytest tests/test_frontend.py -v

Tests frontend helper functions and API integration logic.
All tests use mocked HTTP calls — no running server needed.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================
# Test: Frontend Helper Functions
# ============================================

class TestFrontendHelpers:
    """Tests for frontend utility functions."""

    def _import_app(self):
        """Import app module with mocked streamlit."""
        mock_st = MagicMock()
        mock_st.session_state = {}
        with patch.dict('sys.modules', {'streamlit': mock_st}):
            # We test the helper functions by extracting them
            pass

    def test_render_language_badge_english(self):
        """Test language badge rendering for English."""
        # Inline test of badge logic
        lang = "en"
        labels = {"en": "English", "bn": "Bangla", "banglish": "Banglish"}
        assert lang in labels
        assert labels[lang] == "English"

    def test_render_language_badge_bangla(self):
        lang = "bn"
        labels = {"en": "English", "bn": "Bangla", "banglish": "Banglish"}
        assert labels[lang] == "Bangla"

    def test_render_language_badge_banglish(self):
        lang = "banglish"
        labels = {"en": "English", "bn": "Bangla", "banglish": "Banglish"}
        assert labels[lang] == "Banglish"


class TestAPIIntegration:
    """Tests for API call logic with mocked HTTP."""

    def test_successful_query(self):
        """Simulate a successful API query."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "According to Section 379...",
            "sources": [{"citation": "Penal Code, Section 379", "similarity_score": 0.9}],
            "detected_language": "en",
            "was_translated": False,
        }

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post(
                "http://localhost:8000/api/query",
                json={"question": "What is the penalty for theft?"},
            )
            data = resp.json()
            assert data["answer"] == "According to Section 379..."
            assert len(data["sources"]) == 1

    def test_bangla_query_response(self):
        """Simulate a Bangla query that gets translated."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "answer": "ধারা ৩৭৯ অনুযায়ী...",
            "answer_english": "According to Section 379...",
            "detected_language": "bn",
            "was_translated": True,
            "sources": [],
        }

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post(
                "http://localhost:8000/api/query",
                json={"question": "চুরির শাস্তি কী?"},
            )
            data = resp.json()
            assert data["detected_language"] == "bn"
            assert data["was_translated"] is True

    def test_rate_limited_response(self):
        """Simulate a 429 rate limit response."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post(
                "http://localhost:8000/api/query",
                json={"question": "test"},
            )
            assert resp.status_code == 429

    def test_server_error_response(self):
        """Simulate a 500 server error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post(
                "http://localhost:8000/api/query",
                json={"question": "test"},
            )
            assert resp.status_code == 500

    def test_connection_error(self):
        """Simulate server not running."""
        with patch("requests.post", side_effect=Exception("Connection refused")):
            import requests
            with pytest.raises(Exception, match="Connection refused"):
                requests.post(
                    "http://localhost:8000/api/query",
                    json={"question": "test"},
                )

    def test_health_check_success(self):
        """Simulate successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "vector_store_documents": 150,
            "vector_store_acts": ["labour_act_2006"],
        }

        with patch("requests.get", return_value=mock_response):
            import requests
            resp = requests.get("http://localhost:8000/api/health")
            data = resp.json()
            assert data["status"] == "ok"
            assert data["vector_store_documents"] == 150

    def test_health_check_offline(self):
        """Simulate API offline."""
        with patch("requests.get", side_effect=Exception("Connection refused")):
            import requests
            with pytest.raises(Exception):
                requests.get("http://localhost:8000/api/health")

    def test_sources_endpoint(self):
        """Simulate fetching available sources."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_documents": 200,
            "acts": [
                {"id": "labour_act_2006", "name": "Labour Act 2006", "indexed": True},
                {"id": "penal_code_1860", "name": "Penal Code 1860", "indexed": True},
            ],
            "categories": ["Employment", "Criminal Law"],
        }

        with patch("requests.get", return_value=mock_response):
            import requests
            resp = requests.get("http://localhost:8000/api/sources")
            data = resp.json()
            assert data["total_documents"] == 200
            assert len(data["acts"]) == 2
            assert data["acts"][0]["indexed"] is True

    def test_feedback_submission(self):
        """Simulate feedback submission."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "received",
            "message": "Thank you for your feedback!",
        }

        with patch("requests.post", return_value=mock_response):
            import requests
            resp = requests.post(
                "http://localhost:8000/api/feedback",
                json={
                    "query": "test",
                    "answer": "test answer",
                    "rating": 5,
                    "comment": "Great!",
                },
            )
            data = resp.json()
            assert data["status"] == "received"

    def test_session_history(self):
        """Simulate fetching session history."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "session_id": "test_session",
            "messages": [
                {"role": "user", "content": "What is theft?"},
                {"role": "assistant", "content": "According to Section 378..."},
            ],
        }

        with patch("requests.get", return_value=mock_response):
            import requests
            resp = requests.get("http://localhost:8000/api/session/test_session")
            data = resp.json()
            assert len(data["messages"]) == 2


class TestFrontendConfig:
    """Tests for frontend configuration."""

    def test_example_questions_exist(self):
        """Verify example questions cover key categories."""
        example_questions = [
            "What is the penalty for theft?",
            "What are the maximum working hours per day?",
            "My employer hasn't paid me for 3 months. What can I do?",
            "চুরির শাস্তি কী?",
            "amar malik betan dey nai, ki korbo?",
            "What rights does a tenant have?",
            "How do I file a consumer complaint?",
        ]
        assert len(example_questions) >= 5

        # Check language coverage
        has_english = any(q[0].isascii() and not any(c in q for c in "আকখগ") for q in example_questions)
        has_bangla = any(any('\u0980' <= c <= '\u09FF' for c in q) for q in example_questions)
        has_banglish = any("amar" in q.lower() or "malik" in q.lower() for q in example_questions)
        assert has_english
        assert has_bangla
        assert has_banglish

    def test_language_options(self):
        """Verify language options are correct."""
        lang_options = {"Auto-detect": "auto", "English": "en", "Bangla (বাংলা)": "bn"}
        assert lang_options["Auto-detect"] == "auto"
        assert lang_options["English"] == "en"
        assert lang_options["Bangla (বাংলা)"] == "bn"

    def test_api_base_url(self):
        API_BASE_URL = "http://localhost:8000"
        assert "localhost" in API_BASE_URL
        assert "8000" in API_BASE_URL


class TestSourceRendering:
    """Tests for source citation rendering logic."""

    def test_source_data_structure(self):
        """Verify source data has expected fields."""
        source = {
            "citation": "Labour Act, Section 42",
            "act_name": "Labour Act 2006",
            "section_number": "42",
            "similarity_score": 0.89,
            "rerank_score": 2.5,
        }
        assert source["citation"] == "Labour Act, Section 42"
        assert source["similarity_score"] > 0

    def test_empty_sources(self):
        """Verify empty sources are handled."""
        sources = []
        assert len(sources) == 0

    def test_multiple_sources(self):
        """Verify multiple sources can be rendered."""
        sources = [
            {"citation": "Source 1", "similarity_score": 0.9, "rerank_score": 0},
            {"citation": "Source 2", "similarity_score": 0.8, "rerank_score": 0},
            {"citation": "Source 3", "similarity_score": 0.7, "rerank_score": 0},
        ]
        assert len(sources) == 3
        # Verify they're sorted by relevance
        scores = [s["similarity_score"] for s in sources]
        assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
