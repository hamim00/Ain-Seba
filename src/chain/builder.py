"""
AinSeba - Chain Builder
Factory function that wires all components (embedder, store, retriever, reranker, chain)
into a ready-to-use LegalRAGChain instance.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
    RERANKER_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    CONVERSATION_MEMORY_K,
)
from src.vectorstore.embeddings import EmbeddingGenerator
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import LegalRetriever
from src.chain.rag_chain import LegalRAGChain

logger = logging.getLogger(__name__)


def build_rag_chain(
    api_key: str = OPENAI_API_KEY,
    use_reranker: bool = True,
    memory_k: int = CONVERSATION_MEMORY_K,
    llm_model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> LegalRAGChain:
    """
    Build a fully wired LegalRAGChain from configuration.
    
    This is the main entry point for creating a working chain.
    Initializes: Embedder -> ChromaStore -> Reranker -> Retriever -> RAGChain.
    
    Args:
        api_key: OpenAI API key.
        use_reranker: Whether to load the cross-encoder reranker.
        memory_k: Conversation memory window size.
        llm_model: LLM model name.
        temperature: LLM temperature.
        max_tokens: LLM max response tokens.
    
    Returns:
        Ready-to-use LegalRAGChain instance.
    """
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Set OPENAI_API_KEY in .env file."
        )

    logger.info("Building AinSeba RAG chain...")

    # 1. Embedding generator
    embedder = EmbeddingGenerator(api_key=api_key, model=EMBEDDING_MODEL)
    logger.info(f"  Embedder: {EMBEDDING_MODEL}")

    # 2. ChromaDB vector store
    store = ChromaStore(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    doc_count = store.collection.count()
    logger.info(f"  Vector store: {doc_count} documents in '{CHROMA_COLLECTION_NAME}'")

    if doc_count == 0:
        logger.warning(
            "  Vector store is EMPTY! Run Phase 2 population first:\n"
            "    python scripts/run_vectorstore.py --populate-sample"
        )

    # 3. Reranker (optional)
    reranker = None
    if use_reranker:
        try:
            reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)
            logger.info(f"  Reranker: {RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"  Reranker failed to load: {e}. Proceeding without.")

    # 4. Retriever
    retriever = LegalRetriever(
        chroma_store=store,
        embedding_generator=embedder,
        reranker=reranker,
        top_k=RETRIEVAL_TOP_K,
        rerank_top_n=RERANK_TOP_N,
    )
    logger.info(f"  Retriever: top_k={RETRIEVAL_TOP_K}, rerank_top_n={RERANK_TOP_N}")

    # 5. RAG Chain
    chain = LegalRAGChain(
        retriever=retriever,
        api_key=api_key,
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        memory_k=memory_k,
    )
    logger.info(f"  LLM: {llm_model} (temp={temperature})")
    logger.info("RAG chain ready!")

    return chain
