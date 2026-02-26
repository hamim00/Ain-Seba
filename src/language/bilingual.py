"""
AinSeba - Bilingual Pipeline
Wraps the RAG chain with language detection and translation support.

Full bilingual flow:
1. Detect language (Bangla / English / Banglish)
2. If non-English: translate query to English for retrieval
3. Run RAG chain with English query
4. If original was Bangla/Banglish: translate response to Bangla
5. Return response in user's language with source tracking
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.language.detector import detect_language, Language, DetectionResult
from src.language.translator import QueryTranslator
from src.chain.rag_chain import LegalRAGChain, RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class BilingualResponse:
    """Extended response with bilingual metadata."""
    answer: str                            # Final answer in user's language
    answer_english: str                    # English version (always available)
    sources: list[dict] = field(default_factory=list)
    query_original: str = ""               # User's original query
    query_english: str = ""                # English translation (if translated)
    detected_language: str = "en"          # Detected language code
    response_language: str = "en"          # Language of the answer
    detection_confidence: float = 0.0
    was_translated: bool = False           # Whether translation was applied
    model: str = ""
    session_id: str = "default"

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "answer_english": self.answer_english,
            "sources": self.sources,
            "query_original": self.query_original,
            "query_english": self.query_english,
            "detected_language": self.detected_language,
            "response_language": self.response_language,
            "detection_confidence": self.detection_confidence,
            "was_translated": self.was_translated,
            "model": self.model,
            "session_id": self.session_id,
        }


class BilingualRAGChain:
    """
    Bilingual wrapper around the LegalRAGChain.

    Adds:
    - Automatic language detection on queries
    - Query translation (Bangla/Banglish -> English) for retrieval
    - Response translation (English -> Bangla) for user
    - Language override option
    - Translation token tracking
    """

    def __init__(
        self,
        rag_chain: LegalRAGChain,
        translator: QueryTranslator,
        default_response_language: str = "auto",
    ):
        """
        Args:
            rag_chain: The Phase 3 LegalRAGChain instance.
            translator: QueryTranslator instance.
            default_response_language: "auto" (detect), "en", or "bn".
        """
        self.rag_chain = rag_chain
        self.translator = translator
        self.default_response_language = default_response_language

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        response_language: Optional[str] = None,
        act_id: Optional[str] = None,
        category: Optional[str] = None,
        use_reranker: bool = True,
    ) -> BilingualResponse:
        """
        Process a legal question with bilingual support.

        Args:
            question: User's question in any supported language.
            session_id: Conversation session ID.
            response_language: Override response language ("en", "bn", or None for auto).
            act_id: Filter by specific act.
            category: Filter by category.
            use_reranker: Whether to use reranking.

        Returns:
            BilingualResponse with answer in the appropriate language.
        """
        session_id = session_id or "default"

        # Step 1: Detect language
        detection = detect_language(question)
        logger.info(
            f"Language detected: {detection.language.value} "
            f"(confidence={detection.confidence:.2f})"
        )

        # Step 2: Determine response language
        target_lang = response_language or self.default_response_language
        if target_lang == "auto":
            target_lang = detection.response_language

        # Step 3: Translate query if needed
        english_query = question
        was_translated = False

        if detection.needs_translation:
            english_query = self.translator.translate_query_to_english(
                query=question,
                source_language=detection.language.value,
            )
            was_translated = True
            logger.info(f"Translated query: '{english_query[:80]}...'")

        # Step 4: Run RAG chain with English query
        rag_response = self.rag_chain.query(
            question=english_query,
            session_id=session_id,
            act_id=act_id,
            category=category,
            use_reranker=use_reranker,
        )

        # Step 5: Translate response if needed
        answer_english = rag_response.answer
        final_answer = answer_english

        if target_lang == "bn" and was_translated:
            final_answer = self.translator.translate_response_to_bangla(answer_english)
            logger.info("Response translated to Bangla")

        # Step 6: Build bilingual response
        return BilingualResponse(
            answer=final_answer,
            answer_english=answer_english,
            sources=rag_response.sources,
            query_original=question,
            query_english=english_query,
            detected_language=detection.language.value,
            response_language=target_lang,
            detection_confidence=detection.confidence,
            was_translated=was_translated,
            model=rag_response.model,
            session_id=session_id,
        )

    def detect_only(self, text: str) -> DetectionResult:
        """Detect language without querying. Useful for debugging."""
        return detect_language(text)

    def translate_only(self, text: str, to_english: bool = True) -> str:
        """Translate text without querying. Useful for debugging."""
        if to_english:
            detection = detect_language(text)
            return self.translator.translate_query_to_english(
                text, source_language=detection.language.value
            )
        else:
            return self.translator.translate_response_to_bangla(text)

    @property
    def translation_tokens_used(self) -> int:
        """Total tokens used for translation."""
        return self.translator.total_translation_tokens
