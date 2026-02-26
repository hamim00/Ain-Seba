"""
AinSeba - Unit Tests for Phase 4 (Bilingual Support)

Run with: pytest tests/test_bilingual.py -v

All tests use mocked APIs — no API key needed.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.detector import (
    detect_language,
    Language,
    DetectionResult,
    _detect_banglish,
    BANGLISH_MARKERS,
)


# ============================================
# Test: Language Detection
# ============================================

class TestLanguageDetection:
    """Tests for the language detector."""

    # --- Pure English ---
    def test_detects_english(self):
        result = detect_language("What is the penalty for theft?")
        assert result.language == Language.ENGLISH
        assert result.confidence > 0.5

    def test_detects_english_legal_question(self):
        result = detect_language("What are the maximum working hours per day?")
        assert result.language == Language.ENGLISH

    def test_detects_english_short(self):
        result = detect_language("hello")
        assert result.language == Language.ENGLISH

    # --- Pure Bangla ---
    def test_detects_bangla(self):
        result = detect_language("চুরির শাস্তি কী?")
        assert result.language == Language.BANGLA
        assert result.has_bangla_script is True
        assert result.confidence > 0.7

    def test_detects_bangla_long(self):
        result = detect_language("আমার মালিক তিন মাস ধরে বেতন দেয়নি। আমি কী করতে পারি?")
        assert result.language == Language.BANGLA
        assert result.bangla_ratio > 0.7

    def test_detects_bangla_legal_terms(self):
        result = detect_language("কনজিউমার রাইটস প্রোটেকশন আইন কী বলে?")
        assert result.language == Language.BANGLA

    # --- Banglish ---
    def test_detects_banglish(self):
        result = detect_language("amar malik betan dey nai ki korbo")
        assert result.language == Language.BANGLISH
        assert result.has_bangla_script is False
        assert result.has_latin_script is True

    def test_detects_banglish_legal(self):
        result = detect_language("ami jante chai ain ki bole shromik odhikar niye")
        assert result.language == Language.BANGLISH

    def test_detects_banglish_question(self):
        result = detect_language("korte parbo ki ami case file dorkar ki lagbe")
        assert result.language == Language.BANGLISH

    # --- Mixed language ---
    def test_detects_mixed_mostly_bangla(self):
        result = detect_language("Section 42 অনুযায়ী working hours কত?")
        assert result.has_bangla_script is True
        assert result.has_latin_script is True
        assert result.is_mixed is True

    def test_detects_mixed_mostly_english(self):
        result = detect_language("What are শ্রমিকের rights under the law?")
        assert result.is_mixed is True
        assert result.has_bangla_script is True

    # --- Edge cases ---
    def test_empty_string(self):
        result = detect_language("")
        assert result.language == Language.UNKNOWN
        assert result.confidence == 0.0

    def test_none_like_input(self):
        result = detect_language("   ")
        assert result.language == Language.UNKNOWN

    def test_numbers_only(self):
        result = detect_language("12345 67890")
        assert result.language in (Language.ENGLISH, Language.UNKNOWN)

    def test_single_bangla_word(self):
        result = detect_language("আইন")
        assert result.language == Language.BANGLA


class TestDetectionResult:
    """Tests for DetectionResult properties."""

    def test_is_bangla_for_bangla(self):
        result = detect_language("আইনের ব্যাখ্যা দিন")
        assert result.is_bangla is True

    def test_is_bangla_for_banglish(self):
        result = detect_language("amar malik betan dey nai ki korbo")
        assert result.is_bangla is True

    def test_is_bangla_false_for_english(self):
        result = detect_language("What is the law about theft?")
        assert result.is_bangla is False

    def test_needs_translation_bangla(self):
        result = detect_language("চুরির শাস্তি কী?")
        assert result.needs_translation is True

    def test_needs_translation_banglish(self):
        result = detect_language("amar malik betan dey nai ki korbo")
        assert result.needs_translation is True

    def test_needs_translation_english(self):
        result = detect_language("What is the penalty for theft?")
        assert result.needs_translation is False

    def test_response_language_bangla(self):
        result = detect_language("চুরির শাস্তি কী?")
        assert result.response_language == "bn"

    def test_response_language_english(self):
        result = detect_language("What is the penalty?")
        assert result.response_language == "en"


class TestBanglishDetection:
    """Tests for the Banglish marker detection."""

    def test_high_banglish_score(self):
        score = _detect_banglish("ami jante chai ain ki bole")
        assert score > 0.3

    def test_low_banglish_score(self):
        score = _detect_banglish("What is the penalty for theft")
        assert score < 0.3

    def test_empty_text(self):
        score = _detect_banglish("")
        assert score == 0.0

    def test_mixed_banglish_english(self):
        score = _detect_banglish("ami want to know about labor law")
        assert score > 0.0  # At least some Banglish detected

    def test_markers_exist(self):
        assert "ami" in BANGLISH_MARKERS
        assert "ain" in BANGLISH_MARKERS
        assert "odhikar" in BANGLISH_MARKERS


# ============================================
# Test: Query Translator (mocked)
# ============================================

class TestQueryTranslator:
    """Tests for the QueryTranslator with mocked OpenAI."""

    def _build_mock_translator(self):
        from src.language.translator import QueryTranslator

        translator = QueryTranslator.__new__(QueryTranslator)
        translator.model = "gpt-4o-mini"
        translator.total_translation_tokens = 0

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "What is the penalty for theft?"
        mock_response.usage.total_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        translator.client = mock_client

        return translator

    def test_translate_bangla_to_english(self):
        translator = self._build_mock_translator()
        result = translator.translate_query_to_english("চুরির শাস্তি কী?", "bn")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_translate_banglish_to_english(self):
        translator = self._build_mock_translator()
        result = translator.translate_query_to_english(
            "amar malik betan dey nai", "banglish"
        )
        assert isinstance(result, str)

    def test_translate_response_to_bangla(self):
        translator = self._build_mock_translator()
        translator.client.chat.completions.create.return_value.choices[0].message.content = (
            "ধারা ৪২ অনুযায়ী..."
        )
        result = translator.translate_response_to_bangla("According to Section 42...")
        assert isinstance(result, str)

    def test_translation_tracks_tokens(self):
        translator = self._build_mock_translator()
        translator.translate_query_to_english("test", "bn")
        assert translator.total_translation_tokens == 50

    def test_translator_rejects_empty_api_key(self):
        from src.language.translator import QueryTranslator
        with pytest.raises(ValueError, match="API key"):
            QueryTranslator(api_key="")


# ============================================
# Test: Bilingual RAG Chain (mocked)
# ============================================

class TestBilingualRAGChain:
    """Tests for the BilingualRAGChain with mocked components."""

    def _build_mock_bilingual_chain(self):
        from src.language.bilingual import BilingualRAGChain
        from src.chain.rag_chain import RAGResponse

        # Mock RAG chain
        mock_rag = MagicMock()
        mock_rag.query.return_value = RAGResponse(
            answer="According to Section 42, no worker shall work more than 8 hours.",
            sources=[{"citation": "Labour Act, Section 42", "similarity_score": 0.89, "rerank_score": 0.0}],
            query="working hours",
            retrieval_count=1,
            model="gpt-4o-mini",
        )

        # Mock translator
        mock_translator = MagicMock()
        mock_translator.translate_query_to_english.return_value = "What are the maximum working hours?"
        mock_translator.translate_response_to_bangla.return_value = (
            "ধারা ৪২ অনুযায়ী, কোনো শ্রমিক দৈনিক ৮ ঘণ্টার বেশি কাজ করবে না।"
        )
        mock_translator.total_translation_tokens = 100

        chain = BilingualRAGChain(
            rag_chain=mock_rag,
            translator=mock_translator,
            default_response_language="auto",
        )

        return chain, mock_rag, mock_translator

    def test_english_query_no_translation(self):
        chain, mock_rag, mock_translator = self._build_mock_bilingual_chain()

        response = chain.query("What are the working hours?")

        assert response.detected_language == "en"
        assert response.was_translated is False
        mock_translator.translate_query_to_english.assert_not_called()

    def test_bangla_query_translates(self):
        chain, mock_rag, mock_translator = self._build_mock_bilingual_chain()

        response = chain.query("শ্রমিকের সর্বোচ্চ কর্মঘণ্টা কত?")

        assert response.detected_language == "bn"
        assert response.was_translated is True
        mock_translator.translate_query_to_english.assert_called_once()

    def test_bangla_response_in_bangla(self):
        chain, mock_rag, mock_translator = self._build_mock_bilingual_chain()

        response = chain.query("শ্রমিকের সর্বোচ্চ কর্মঘণ্টা কত?")

        assert response.response_language == "bn"
        mock_translator.translate_response_to_bangla.assert_called_once()
        assert response.answer != response.answer_english

    def test_banglish_query_translates(self):
        chain, mock_rag, mock_translator = self._build_mock_bilingual_chain()

        response = chain.query("amar malik betan dey nai ki korbo")

        assert response.detected_language == "banglish"
        assert response.was_translated is True

    def test_english_keeps_answer_english(self):
        chain, mock_rag, mock_translator = self._build_mock_bilingual_chain()

        response = chain.query("What is the penalty for theft?")

        assert response.answer == response.answer_english
        mock_translator.translate_response_to_bangla.assert_not_called()

    def test_forced_language_override(self):
        chain, mock_rag, mock_translator = self._build_mock_bilingual_chain()

        # Force Bangla response for English query
        response = chain.query(
            "What are working hours?",
            response_language="bn",
        )

        # English query should not be translated for retrieval
        assert response.detected_language == "en"
        assert response.was_translated is False

    def test_sources_preserved(self):
        chain, mock_rag, mock_translator = self._build_mock_bilingual_chain()

        response = chain.query("test query")

        assert len(response.sources) > 0
        assert response.sources[0]["citation"] == "Labour Act, Section 42"

    def test_session_id_passed(self):
        chain, mock_rag, mock_translator = self._build_mock_bilingual_chain()

        response = chain.query("test", session_id="my_session")

        assert response.session_id == "my_session"
        mock_rag.query.assert_called_once()
        call_kwargs = mock_rag.query.call_args[1]
        assert call_kwargs["session_id"] == "my_session"

    def test_to_dict(self):
        chain, _, _ = self._build_mock_bilingual_chain()
        response = chain.query("What is the penalty?")
        d = response.to_dict()
        assert "answer" in d
        assert "answer_english" in d
        assert "detected_language" in d
        assert "was_translated" in d

    def test_detect_only(self):
        chain, _, _ = self._build_mock_bilingual_chain()
        detection = chain.detect_only("আমার প্রশ্ন")
        assert detection.language == Language.BANGLA

    def test_translation_tokens_tracked(self):
        chain, _, _ = self._build_mock_bilingual_chain()
        assert chain.translation_tokens_used == 100


# ============================================
# Test: Bilingual Response
# ============================================

class TestBilingualResponse:
    """Tests for the BilingualResponse dataclass."""

    def test_bilingual_response_creation(self):
        from src.language.bilingual import BilingualResponse

        response = BilingualResponse(
            answer="Bangla answer",
            answer_english="English answer",
            query_original="Bangla query",
            query_english="English query",
            detected_language="bn",
            response_language="bn",
            was_translated=True,
        )
        assert response.answer == "Bangla answer"
        assert response.answer_english == "English answer"
        assert response.was_translated is True

    def test_to_dict_complete(self):
        from src.language.bilingual import BilingualResponse

        response = BilingualResponse(
            answer="test",
            answer_english="test",
            detected_language="bn",
            response_language="bn",
            detection_confidence=0.95,
            was_translated=True,
        )
        d = response.to_dict()
        assert d["detected_language"] == "bn"
        assert d["was_translated"] is True
        assert d["detection_confidence"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
