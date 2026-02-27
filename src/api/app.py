"""
AinSeba - FastAPI Backend
Production-ready REST API wrapping the entire RAG pipeline.

Endpoints:
    POST /api/query          — Submit a legal question (bilingual)
    POST /api/query/stream   — Streaming response (SSE)
    GET  /api/health         — Health check + vector store status
    GET  /api/sources        — List available law documents
    POST /api/feedback       — User feedback on response quality
    GET  /api/session/{id}   — Get conversation history for a session

Run with:
    uvicorn src.api.app:app --reload --port 8000
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

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
from src.api.rate_limiter import RateLimiter
from src.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    CHROMA_COLLECTION_NAME,
    LAW_REGISTRY,
)

logger = logging.getLogger(__name__)

# ============================================
# Global State (initialized at startup)
# ============================================
_bilingual_chain = None
_rag_chain = None
_feedback_store: list[dict] = []
_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)


def _get_bilingual_chain():
    """Lazy-load the bilingual chain."""
    global _bilingual_chain
    if _bilingual_chain is None:
        from src.chain.builder import build_bilingual_chain
        _bilingual_chain = build_bilingual_chain()
    return _bilingual_chain


def _get_rag_chain():
    """Lazy-load the base RAG chain (from bilingual wrapper)."""
    return _get_bilingual_chain().rag_chain


# ============================================
# App Lifespan
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("AinSeba API starting up...")
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set! API calls will fail.")
    yield
    logger.info("AinSeba API shutting down.")


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="AinSeba API",
    description=(
        "Bangladesh Legal Aid RAG Assistant API. "
        "Ask legal questions in English, Bangla, or Banglish and receive "
        "citation-grounded answers from Bangladesh law."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Middleware: Request Logging
# ============================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} ({duration:.2f}s)"
    )
    return response


# ============================================
# Helper: Rate Limit Check
# ============================================

def _check_rate_limit(request: Request):
    """Check rate limit and raise 429 if exceeded."""
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        remaining = _rate_limiter.remaining(client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {_rate_limiter.window_seconds} seconds.",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )


# ============================================
# Endpoints
# ============================================

@app.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Ask a legal question",
    description="Submit a legal question in English, Bangla, or Banglish. "
    "Returns a citation-grounded answer with source references.",
    tags=["Query"],
)
async def query(request: Request, body: QueryRequest):
    """Process a legal question through the bilingual RAG pipeline."""
    _check_rate_limit(request)

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="API key not configured")

    try:
        chain = _get_bilingual_chain()

        response = chain.query(
            question=body.question,
            session_id=body.session_id,
            response_language=body.language,
            act_id=body.act_id,
            category=body.category,
            use_reranker=body.use_reranker,
        )

        # Map sources to Pydantic model
        sources = []
        for src in response.sources:
            sources.append(SourceInfo(
                citation=src.get("citation", ""),
                act_name=src.get("act_name", ""),
                act_id=src.get("act_id", ""),
                section_number=src.get("section_number", ""),
                section_title=src.get("section_title", ""),
                chapter=src.get("chapter", ""),
                similarity_score=src.get("similarity_score", 0.0),
                rerank_score=src.get("rerank_score", 0.0),
            ))

        return QueryResponse(
            answer=response.answer,
            answer_english=response.answer_english,
            sources=sources,
            query_original=response.query_original,
            query_english=response.query_english,
            detected_language=response.detected_language,
            response_language=response.response_language,
            was_translated=response.was_translated,
            session_id=response.session_id,
            model=response.model,
            retrieval_count=len(response.sources),
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/query/stream",
    summary="Ask a legal question (streaming)",
    description="Submit a legal question and receive the answer as Server-Sent Events (SSE).",
    tags=["Query"],
)
async def query_stream(request: Request, body: QueryRequest):
    """Stream a legal answer using Server-Sent Events."""
    _check_rate_limit(request)

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="API key not configured")

    try:
        chain = _get_bilingual_chain()

        # For streaming, detect language and translate query first
        from src.language.detector import detect_language
        detection = detect_language(body.question)

        english_query = body.question
        if detection.needs_translation:
            english_query = chain.translator.translate_query_to_english(
                body.question, detection.language.value
            )

        rag_chain = chain.rag_chain

        async def event_generator():
            """Generate SSE events."""
            # Send metadata first
            metadata = {
                "type": "metadata",
                "detected_language": detection.language.value,
                "query_english": english_query,
                "was_translated": detection.needs_translation,
            }
            yield f"data: {json.dumps(metadata)}\n\n"

            # Stream answer tokens
            for token in rag_chain.stream(
                english_query,
                session_id=body.session_id,
                act_id=body.act_id,
                category=body.category,
                use_reranker=body.use_reranker,
            ):
                chunk = {"type": "token", "content": token}
                yield f"data: {json.dumps(chunk)}\n\n"

            # Send done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and vector store status.",
    tags=["System"],
)
async def health():
    """Health check endpoint."""
    doc_count = 0
    acts = []

    try:
        chain = _get_bilingual_chain()
        store = chain.rag_chain.retriever.store
        stats = store.get_stats()
        doc_count = stats.get("total_documents", 0)
        acts = stats.get("acts", [])
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="1.0.0",
        vector_store_documents=doc_count,
        vector_store_acts=acts,
        model=LLM_MODEL,
    )


@app.get(
    "/api/sources",
    response_model=SourceListResponse,
    summary="List available laws",
    description="List all law documents available in the system.",
    tags=["System"],
)
async def list_sources():
    """List available law documents and categories."""
    doc_count = 0
    acts_in_store = []

    try:
        chain = _get_bilingual_chain()
        store = chain.rag_chain.retriever.store
        stats = store.get_stats()
        doc_count = stats.get("total_documents", 0)
        acts_in_store = stats.get("acts", [])
    except Exception:
        pass

    # Combine registry info with vector store status
    acts = []
    for law in LAW_REGISTRY:
        acts.append({
            "id": law["id"],
            "name": law["name"],
            "category": law["category"],
            "year": law["year"],
            "priority": law["priority"],
            "indexed": law["id"] in acts_in_store,
        })

    categories = sorted(set(law["category"] for law in LAW_REGISTRY))

    return SourceListResponse(
        total_documents=doc_count,
        acts=acts,
        categories=categories,
    )


@app.post(
    "/api/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback",
    description="Submit feedback on a response quality (1-5 rating).",
    tags=["Feedback"],
)
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Store user feedback for quality tracking."""
    _check_rate_limit(request)

    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": body.query,
        "answer_preview": body.answer[:200],
        "rating": body.rating,
        "comment": body.comment,
        "session_id": body.session_id,
    }

    _feedback_store.append(feedback_entry)
    logger.info(f"Feedback received: rating={body.rating}, query='{body.query[:50]}'")

    return FeedbackResponse(
        status="received",
        message="Thank you for your feedback!",
    )


@app.get(
    "/api/session/{session_id}",
    summary="Get conversation history",
    description="Retrieve conversation history for a session.",
    tags=["Session"],
)
async def get_session(session_id: str):
    """Get conversation history for a session."""
    try:
        chain = _get_bilingual_chain()
        history = chain.rag_chain.get_conversation_history(session_id)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500,
        ).model_dump(),
    )
