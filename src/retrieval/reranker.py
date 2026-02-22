"""
AinSeba - Cross-Encoder Reranker
Re-ranks retrieved chunks using a cross-encoder model for better precision.

The cross-encoder scores each (query, document) pair directly, which is more
accurate than cosine similarity alone but slower — hence used as a second stage.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved results using a cross-encoder model.
    
    Uses sentence-transformers CrossEncoder which processes (query, document)
    pairs and outputs a relevance score. Much more accurate than bi-encoder
    similarity but too slow for first-stage retrieval.
    
    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Fast (~50ms per query-doc pair on CPU)
    - Trained on MS MARCO passage ranking
    - Good balance of speed and accuracy
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: HuggingFace model name for the cross-encoder.
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"Reranker initialized (model: {model_name})")

    @property
    def model(self):
        """Lazy-load the model (downloads on first use)."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}...")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_n: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Rerank retrieval results using the cross-encoder.

        Args:
            query: The user's query string.
            results: List of result dicts from ChromaStore.query_with_text().
                     Each must have a "text" key.
            top_n: Return only the top N results after reranking.
                   If None, returns all results.
            score_threshold: Minimum reranker score to include a result.
                            If None, no threshold is applied.

        Returns:
            Reranked list of result dicts, sorted by cross-encoder score (desc).
            Each dict gets an added "rerank_score" field.
        """
        if not results:
            return []

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, r["text"]) for r in results]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores to results
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort by rerank score (highest = most relevant)
        reranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)

        # Apply threshold
        if score_threshold is not None:
            reranked = [r for r in reranked if r["rerank_score"] >= score_threshold]

        # Apply top_n
        if top_n is not None:
            reranked = reranked[:top_n]

        logger.debug(
            f"Reranked {len(results)} → {len(reranked)} results "
            f"(top score: {reranked[0]['rerank_score']:.3f})"
            if reranked else "No results after reranking."
        )

        return reranked
