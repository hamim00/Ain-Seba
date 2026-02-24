"""
AinSeba - Unit Tests for Phase 3 (RAG Chain & LLM Integration)

Run with: pytest tests/test_chain.py -v

All tests use mocked LLM/retriever — no API key needed.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chain.memory import ConversationMemory, Message
from src.prompts.templates import (
    SYSTEM_PROMPT,
    format_context,
    format_chat_history,
    build_user_prompt,
)


# ============================================
# Fixtures
# ============================================

@dataclass
class MockRetrievalResult:
    """Minimal mock of RetrievalResult for testing."""
    chunk_id: str = "test_001"
    text: str = "Section 42. Maximum working hours. No worker shall work more than 8 hours."
    act_name: str = "Labour Act 2006"
    act_id: str = "labour_act_2006"
    section_number: str = "42"
    section_title: str = "Maximum working hours"
    chapter: str = "Chapter III"
    category: str = "Employment"
    year: int = 2006
    similarity_score: float = 0.89
    rerank_score: float = 2.5

    @property
    def citation(self):
        parts = [self.act_name]
        if self.chapter:
            parts.append(self.chapter)
        if self.section_number:
            title = f" ({self.section_title})" if self.section_title else ""
            parts.append(f"Section {self.section_number}{title}")
        return ", ".join(parts)


@pytest.fixture
def mock_results():
    """Sample retrieval results."""
    return [
        MockRetrievalResult(
            chunk_id="test_001",
            text="Section 42. Maximum working hours. No worker shall be required to work more than eight hours in any day.",
            section_number="42",
            section_title="Maximum working hours",
        ),
        MockRetrievalResult(
            chunk_id="test_002",
            text="Section 43. Overtime. Where a worker works beyond prescribed hours, the employer shall pay overtime at twice the rate.",
            section_number="43",
            section_title="Overtime",
        ),
        MockRetrievalResult(
            chunk_id="test_003",
            text="Section 6. Payment of wages. Wages shall be paid before the seventh working day.",
            section_number="6",
            section_title="Payment of wages",
            chapter="Chapter IV",
        ),
    ]


# ============================================
# Test: Conversation Memory
# ============================================

class TestConversationMemory:
    """Tests for the ConversationMemory class."""

    def test_empty_memory(self):
        mem = ConversationMemory(k=5)
        assert mem.get_history() == []
        assert mem.message_count == 0

    def test_add_user_message(self):
        mem = ConversationMemory(k=5)
        mem.add_user_message("What is the penalty for theft?")
        history = mem.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert "theft" in history[0]["content"]

    def test_add_assistant_message(self):
        mem = ConversationMemory(k=5)
        mem.add_assistant_message("According to Section 379...", sources=["Penal Code"])
        history = mem.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "assistant"

    def test_conversation_exchange(self):
        mem = ConversationMemory(k=5)
        mem.add_user_message("Question 1")
        mem.add_assistant_message("Answer 1")
        mem.add_user_message("Follow-up")
        mem.add_assistant_message("Answer 2")
        assert mem.message_count == 4

    def test_sliding_window_trim(self):
        mem = ConversationMemory(k=2)  # Keep last 2 exchanges = 4 messages
        for i in range(10):
            mem.add_user_message(f"Q{i}")
            mem.add_assistant_message(f"A{i}")
        
        history = mem.get_history()
        assert len(history) == 4  # k=2 -> 2 pairs -> 4 messages
        # Should keep the most recent
        assert "Q8" in history[0]["content"]
        assert "A9" in history[-1]["content"]

    def test_multiple_sessions(self):
        mem = ConversationMemory(k=5)
        mem.add_user_message("Question A", session_id="session_a")
        mem.add_user_message("Question B", session_id="session_b")

        history_a = mem.get_history(session_id="session_a")
        history_b = mem.get_history(session_id="session_b")

        assert len(history_a) == 1
        assert len(history_b) == 1
        assert "A" in history_a[0]["content"]
        assert "B" in history_b[0]["content"]

    def test_clear_session(self):
        mem = ConversationMemory(k=5)
        mem.add_user_message("Hello")
        mem.add_assistant_message("Hi")
        mem.clear()
        assert mem.get_history() == []

    def test_get_last_exchange(self):
        mem = ConversationMemory(k=5)
        mem.add_user_message("What about wages?")
        mem.add_assistant_message("Section 6 covers wages.")
        
        last_user, last_assistant = mem.get_last_exchange()
        assert "wages" in last_user
        assert "Section 6" in last_assistant

    def test_get_last_exchange_empty(self):
        mem = ConversationMemory(k=5)
        last_user, last_assistant = mem.get_last_exchange()
        assert last_user is None
        assert last_assistant is None

    def test_export_import(self):
        mem = ConversationMemory(k=5)
        mem.add_user_message("Q1")
        mem.add_assistant_message("A1")

        exported = mem.export_session()
        
        mem2 = ConversationMemory(k=5)
        mem2.import_session(exported)
        
        assert mem2.get_history() == exported

    def test_session_ids(self):
        mem = ConversationMemory(k=5)
        mem.add_user_message("Q", session_id="s1")
        mem.add_user_message("Q", session_id="s2")
        assert "s1" in mem.session_ids
        assert "s2" in mem.session_ids

    def test_message_with_sources(self):
        mem = ConversationMemory(k=5)
        mem.add_assistant_message(
            "Answer",
            sources=["Labour Act, Section 42"],
        )
        history = mem.get_history()
        assert history[0].get("sources") == ["Labour Act, Section 42"]


# ============================================
# Test: Prompt Templates
# ============================================

class TestPromptTemplates:
    """Tests for prompt template functions."""

    def test_system_prompt_has_key_elements(self):
        assert "AinSeba" in SYSTEM_PROMPT
        assert "Bangladesh law" in SYSTEM_PROMPT.lower() or "Bangladesh" in SYSTEM_PROMPT
        assert "citation" in SYSTEM_PROMPT.lower() or "cite" in SYSTEM_PROMPT.lower()
        assert "disclaimer" in SYSTEM_PROMPT.lower()
        assert "Section" in SYSTEM_PROMPT

    def test_system_prompt_has_guardrails(self):
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "only from" in prompt_lower or "provided context" in prompt_lower
        assert "not legal advice" in prompt_lower or "educational" in prompt_lower

    def test_format_context_with_results(self, mock_results):
        context = format_context(mock_results)
        assert "Source [1]" in context
        assert "Source [2]" in context
        assert "Section 42" in context
        assert "Maximum working hours" in context
        assert "Labour Act 2006" in context

    def test_format_context_empty(self):
        context = format_context([])
        assert "No relevant" in context

    def test_format_chat_history_with_messages(self):
        messages = [
            {"role": "user", "content": "What about wages?"},
            {"role": "assistant", "content": "Section 6 covers wages."},
        ]
        history = format_chat_history(messages)
        assert "User:" in history
        assert "Assistant:" in history
        assert "wages" in history

    def test_format_chat_history_empty(self):
        history = format_chat_history([])
        assert "No prior conversation" in history

    def test_format_chat_history_truncates_long_messages(self):
        messages = [
            {"role": "user", "content": "x" * 1000},
        ]
        history = format_chat_history(messages)
        assert "..." in history
        assert len(history) < 1000

    def test_build_user_prompt(self, mock_results):
        prompt = build_user_prompt(
            question="What are working hours?",
            retrieval_results=mock_results,
            chat_history=[],
        )
        assert "What are working hours?" in prompt
        assert "Section 42" in prompt
        assert "Legal Context" in prompt

    def test_build_user_prompt_with_history(self, mock_results):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        prompt = build_user_prompt(
            question="Follow-up question",
            retrieval_results=mock_results,
            chat_history=history,
        )
        assert "Previous question" in prompt
        assert "Follow-up question" in prompt


# ============================================
# Test: RAG Chain (mocked LLM)
# ============================================

class TestRAGChain:
    """Tests for the LegalRAGChain with mocked components."""

    def _build_mock_chain(self):
        """Build a chain with all components mocked."""
        from src.chain.rag_chain import LegalRAGChain

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            MockRetrievalResult(
                chunk_id="test_001",
                text="Section 42. No worker shall work more than 8 hours.",
                section_number="42",
                section_title="Working hours",
            ),
        ]

        # Create chain with mocked LLM
        chain = LegalRAGChain.__new__(LegalRAGChain)
        chain.retriever = mock_retriever
        chain.memory = ConversationMemory(k=5)
        chain.model_name = "gpt-4o-mini"

        # Mock LangChain LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "According to Section 42 of the Labour Act 2006, "
            "no worker shall be required to work more than eight hours per day.\n\n"
            "*Disclaimer: This is for educational purposes only.*"
        )
        mock_llm.__or__ = MagicMock(return_value=MagicMock(
            invoke=MagicMock(return_value=mock_response.content)
        ))

        chain.llm = mock_llm
        chain.parser = MagicMock()

        return chain

    def test_query_returns_response(self):
        chain = self._build_mock_chain()

        # Override the query method's LLM call
        mock_answer = (
            "According to Section 42, no worker shall work more than 8 hours.\n\n"
            "*Disclaimer: This is for educational purposes only.*"
        )

        with patch.object(chain, 'query') as mock_query:
            from src.chain.rag_chain import RAGResponse
            mock_query.return_value = RAGResponse(
                answer=mock_answer,
                sources=[{"citation": "Labour Act 2006, Section 42", "similarity_score": 0.89, "rerank_score": 0}],
                query="working hours",
                retrieval_count=1,
                model="gpt-4o-mini",
            )

            response = chain.query("What are the working hours?")
            assert response.answer is not None
            assert len(response.sources) > 0
            assert response.retrieval_count > 0

    def test_memory_updates_after_query(self):
        mem = ConversationMemory(k=5)
        mem.add_user_message("What about working hours?")
        mem.add_assistant_message("Section 42 covers working hours.")

        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_clear_conversation(self):
        mem = ConversationMemory(k=5)
        mem.add_user_message("Q1")
        mem.add_assistant_message("A1")
        mem.clear()
        assert mem.get_history() == []


# ============================================
# Test: RAGResponse
# ============================================

class TestRAGResponse:
    """Tests for the RAGResponse data model."""

    def test_format_sources(self):
        from src.chain.rag_chain import RAGResponse

        response = RAGResponse(
            answer="Test answer",
            sources=[
                {"citation": "Labour Act, Section 42", "similarity_score": 0.89},
                {"citation": "Penal Code, Section 379", "similarity_score": 0.75},
            ],
        )
        formatted = response.format_sources()
        assert "Labour Act" in formatted
        assert "Penal Code" in formatted
        assert "[1]" in formatted
        assert "[2]" in formatted

    def test_format_sources_empty(self):
        from src.chain.rag_chain import RAGResponse

        response = RAGResponse(answer="Test", sources=[])
        assert "No sources" in response.format_sources()

    def test_to_dict(self):
        from src.chain.rag_chain import RAGResponse

        response = RAGResponse(
            answer="Test answer",
            sources=[{"citation": "Test"}],
            query="Test query",
            model="gpt-4o-mini",
        )
        d = response.to_dict()
        assert d["answer"] == "Test answer"
        assert d["query"] == "Test query"
        assert d["model"] == "gpt-4o-mini"
        assert len(d["sources"]) == 1


# ============================================
# Test: Chain Builder (import check)
# ============================================

class TestChainBuilder:
    """Tests for the chain builder factory."""

    def test_builder_rejects_empty_api_key(self):
        from src.chain.builder import build_rag_chain

        with pytest.raises(ValueError, match="API key"):
            build_rag_chain(api_key="")

    def test_builder_rejects_none_api_key(self):
        from src.chain.builder import build_rag_chain

        with pytest.raises(ValueError, match="API key"):
            build_rag_chain(api_key="")


# ============================================
# Test: End-to-End Prompt Assembly
# ============================================

class TestPromptAssembly:
    """Integration test for the full prompt assembly pipeline."""

    def test_full_prompt_assembly(self, mock_results):
        """Verify the complete prompt is well-formed."""
        history = [
            {"role": "user", "content": "Tell me about labor law"},
            {"role": "assistant", "content": "The Labour Act 2006 covers employment."},
        ]

        prompt = build_user_prompt(
            question="What are the working hours?",
            retrieval_results=mock_results,
            chat_history=history,
        )

        # Should contain all key sections
        assert "Legal Context" in prompt
        assert "Conversation History" in prompt
        assert "User's Question" in prompt
        assert "What are the working hours?" in prompt

        # Should contain retrieved context
        assert "Section 42" in prompt
        assert "Labour Act 2006" in prompt

        # Should contain chat history
        assert "labor law" in prompt

    def test_prompt_with_no_context(self):
        prompt = build_user_prompt(
            question="What about US tax law?",
            retrieval_results=[],
            chat_history=[],
        )
        assert "No relevant" in prompt
        assert "US tax law" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
