"""
AinSeba - RAG Chain
Connects the retrieval pipeline to GPT-4o-mini for citation-grounded legal answers.

Uses LangChain's LCEL (LangChain Expression Language) for a clean,
composable pipeline: Query -> Retrieve -> Format -> LLM -> Parse.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Generator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.prompts.templates import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_context,
)
from src.chain.memory import ConversationMemory

logger = logging.getLogger(__name__)


# ============================================
# Response Data Model
# ============================================

@dataclass
class RAGResponse:
    """Structured response from the RAG chain."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    retrieval_count: int = 0
    model: str = ""
    session_id: str = "default"

    def format_sources(self) -> str:
        """Format sources as a readable string."""
        if not self.sources:
            return "No sources referenced."
        lines = []
        for i, src in enumerate(self.sources, 1):
            citation = src.get("citation", "Unknown source")
            score = src.get("similarity_score", 0)
            lines.append(f"  [{i}] {citation} (relevance: {score:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "retrieval_count": self.retrieval_count,
            "model": self.model,
            "session_id": self.session_id,
        }


# ============================================
# RAG Chain
# ============================================

class LegalRAGChain:
    """
    Full RAG chain for AinSeba legal assistant.
    
    Pipeline:
    1. Take user question
    2. Retrieve relevant legal context (via LegalRetriever from Phase 2)
    3. Format prompt with context + conversation history
    4. Send to GPT-4o-mini
    5. Parse and return structured response with source tracking
    6. Update conversation memory
    
    Features:
    - Citation-grounded answers with section references
    - Conversation memory for follow-up questions (sliding window, k=5)
    - Streaming support for real-time responses
    - Source document tracking
    - Graceful handling of out-of-scope questions
    """

    def __init__(
        self,
        retriever,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1500,
        memory_k: int = 5,
    ):
        """
        Args:
            retriever: LegalRetriever instance (from Phase 2).
            api_key: OpenAI API key.
            model: LLM model name.
            temperature: LLM temperature (low = more factual).
            max_tokens: Maximum response tokens.
            memory_k: Number of conversation exchanges to remember.
        """
        self.retriever = retriever
        self.memory = ConversationMemory(k=memory_k)
        self.model_name = model

        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

        # Output parser
        self.parser = StrOutputParser()

        logger.info(
            f"RAG chain initialized (model={model}, temp={temperature}, memory_k={memory_k})"
        )

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        act_id: Optional[str] = None,
        category: Optional[str] = None,
        use_reranker: bool = True,
    ) -> RAGResponse:
        """
        Process a legal question through the full RAG pipeline.

        Args:
            question: User's legal question.
            session_id: Conversation session ID (for memory).
            act_id: Optional filter by specific act.
            category: Optional filter by legal category.
            use_reranker: Whether to use cross-encoder reranking.

        Returns:
            RAGResponse with answer, sources, and metadata.
        """
        session_id = session_id or "default"

        logger.info(f"Processing query: '{question[:80]}...'")

        # Step 1: Retrieve relevant context
        retrieval_results = self.retriever.retrieve(
            query=question,
            act_id=act_id,
            category=category,
            use_reranker=use_reranker,
        )
        logger.info(f"  Retrieved {len(retrieval_results)} relevant chunks")

        # Step 2: Get conversation history
        chat_history = self.memory.get_history(session_id)

        # Step 3: Build prompt
        user_prompt = build_user_prompt(
            question=question,
            retrieval_results=retrieval_results,
            chat_history=chat_history,
        )

        # Step 4: Call LLM via LangChain LCEL
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        chain = self.llm | self.parser
        answer = chain.invoke(messages)

        # Step 5: Build source tracking
        sources = []
        for r in retrieval_results:
            sources.append({
                "chunk_id": r.chunk_id,
                "citation": r.citation,
                "act_name": r.act_name,
                "act_id": r.act_id,
                "section_number": r.section_number,
                "section_title": r.section_title,
                "chapter": r.chapter,
                "similarity_score": r.similarity_score,
                "rerank_score": r.rerank_score,
                "text_preview": r.text[:200],
            })

        # Step 6: Update conversation memory
        self.memory.add_user_message(question, session_id)
        self.memory.add_assistant_message(
            content=answer,
            sources=[s["citation"] for s in sources],
            session_id=session_id,
        )

        response = RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            retrieval_count=len(retrieval_results),
            model=self.model_name,
            session_id=session_id,
        )

        logger.info(f"  Response generated ({len(answer)} chars, {len(sources)} sources)")

        return response

    def stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        act_id: Optional[str] = None,
        category: Optional[str] = None,
        use_reranker: bool = True,
    ) -> Generator[str, None, RAGResponse]:
        """
        Stream a response token-by-token.

        Yields:
            Individual text chunks as they arrive.

        Returns:
            Final RAGResponse (accessible after generator completes).
        """
        session_id = session_id or "default"

        # Retrieve context
        retrieval_results = self.retriever.retrieve(
            query=question,
            act_id=act_id,
            category=category,
            use_reranker=use_reranker,
        )

        # Build prompt
        chat_history = self.memory.get_history(session_id)
        user_prompt = build_user_prompt(
            question=question,
            retrieval_results=retrieval_results,
            chat_history=chat_history,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # Stream from LLM
        full_answer = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content
            if token:
                full_answer += token
                yield token

        # Build sources
        sources = []
        for r in retrieval_results:
            sources.append({
                "chunk_id": r.chunk_id,
                "citation": r.citation,
                "act_name": r.act_name,
                "act_id": r.act_id,
                "section_number": r.section_number,
                "section_title": r.section_title,
                "chapter": r.chapter,
                "similarity_score": r.similarity_score,
                "rerank_score": r.rerank_score,
            })

        # Update memory
        self.memory.add_user_message(question, session_id)
        self.memory.add_assistant_message(
            content=full_answer,
            sources=[s["citation"] for s in sources],
            session_id=session_id,
        )

    def get_conversation_history(
        self, session_id: Optional[str] = None
    ) -> list[dict]:
        """Get the conversation history for a session."""
        return self.memory.get_history(session_id)

    def clear_conversation(self, session_id: Optional[str] = None) -> None:
        """Clear conversation history for a session."""
        self.memory.clear(session_id)
        logger.info(f"Conversation cleared for session '{session_id or 'default'}'")

    def get_context_preview(
        self,
        question: str,
        act_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        """
        Preview what context would be retrieved for a question,
        without calling the LLM. Useful for debugging retrieval quality.
        """
        results = self.retriever.retrieve(
            query=question,
            act_id=act_id,
            category=category,
        )
        return format_context(results)
