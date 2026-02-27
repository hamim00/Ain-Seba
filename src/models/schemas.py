"""
AinSeba - Pydantic Request/Response Models
Defines all API data schemas for the FastAPI backend.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============================================
# Request Models
# ============================================

class QueryRequest(BaseModel):
    """Request body for legal question submission."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Legal question in English, Bangla, or Banglish",
        examples=["What is the penalty for theft?"],
    )
    session_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Conversation session ID for follow-up questions",
    )
    language: Optional[str] = Field(
        None,
        pattern=r"^(en|bn|auto)$",
        description="Force response language: 'en', 'bn', or 'auto' (detect)",
    )
    act_id: Optional[str] = Field(
        None,
        description="Filter by specific act (e.g., 'labour_act_2006')",
    )
    category: Optional[str] = Field(
        None,
        description="Filter by legal category (e.g., 'Employment')",
    )
    use_reranker: bool = Field(
        True,
        description="Whether to use cross-encoder reranking",
    )


class FeedbackRequest(BaseModel):
    """Request body for user feedback on a response."""
    query: str = Field(..., description="The original question")
    answer: str = Field(..., description="The answer that was given")
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1 (poor) to 5 (excellent)",
    )
    comment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional feedback comment",
    )
    session_id: Optional[str] = Field(None)


# ============================================
# Response Models
# ============================================

class SourceInfo(BaseModel):
    """A single source reference in a response."""
    citation: str = Field(..., description="Human-readable citation")
    act_name: str = Field("", description="Act name")
    act_id: str = Field("", description="Act identifier")
    section_number: str = Field("", description="Section number")
    section_title: str = Field("", description="Section title")
    chapter: str = Field("", description="Chapter")
    similarity_score: float = Field(0.0, description="Cosine similarity score")
    rerank_score: float = Field(0.0, description="Cross-encoder rerank score")


class QueryResponse(BaseModel):
    """Response body for a legal query."""
    answer: str = Field(..., description="The legal answer")
    answer_english: str = Field("", description="English version of the answer")
    sources: list[SourceInfo] = Field(
        default_factory=list,
        description="Source citations used in the answer",
    )
    query_original: str = Field("", description="Original user query")
    query_english: str = Field("", description="English translation of query")
    detected_language: str = Field("en", description="Detected query language")
    response_language: str = Field("en", description="Language of the answer")
    was_translated: bool = Field(False, description="Whether translation was applied")
    session_id: str = Field("default", description="Session ID")
    model: str = Field("", description="LLM model used")
    retrieval_count: int = Field(0, description="Number of chunks retrieved")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("ok", description="Service status")
    version: str = Field("1.0.0", description="API version")
    vector_store_documents: int = Field(0, description="Documents in vector store")
    vector_store_acts: list[str] = Field(
        default_factory=list,
        description="Available law acts",
    )
    model: str = Field("", description="LLM model")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )


class SourceListResponse(BaseModel):
    """Response listing available law documents."""
    total_documents: int = Field(0, description="Total chunks in vector store")
    acts: list[dict] = Field(
        default_factory=list,
        description="Available law acts with metadata",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Available legal categories",
    )


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    status: str = Field("received", description="Feedback status")
    message: str = Field("Thank you for your feedback!")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error info")
    status_code: int = Field(500, description="HTTP status code")
