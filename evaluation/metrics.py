"""
AinSeba - RAG Evaluation Metrics
RAGAS-inspired evaluation framework for measuring RAG pipeline quality.

Metrics:
    1. Faithfulness     — Is the answer grounded in retrieved context?
    2. Answer Relevancy — Does the answer address the question?
    3. Context Precision — Are the retrieved chunks relevant to the question?
    4. Context Recall    — Did retrieval capture the needed information?
    5. Citation Accuracy — Are section references correct?

Uses GPT-4o-mini as the evaluator LLM (LLM-as-judge pattern).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    name: str
    score: float            # 0.0 - 1.0
    reasoning: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Complete evaluation result for a single query."""
    eval_id: str
    question: str
    answer: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    query_type: str = ""
    category: str = ""

    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        if not self.metrics:
            return 0.0
        weights = {
            "faithfulness": 0.30,
            "answer_relevancy": 0.25,
            "context_precision": 0.20,
            "context_recall": 0.15,
            "citation_accuracy": 0.10,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for name, result in self.metrics.items():
            w = weights.get(name, 0.1)
            weighted_sum += result.score * w
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "eval_id": self.eval_id,
            "question": self.question,
            "answer": self.answer[:300],
            "ground_truth": self.ground_truth[:300],
            "query_type": self.query_type,
            "category": self.category,
            "overall_score": round(self.overall_score, 4),
            "metrics": {
                name: {
                    "score": round(r.score, 4),
                    "reasoning": r.reasoning,
                    **r.details,
                }
                for name, r in self.metrics.items()
            },
            "num_contexts": len(self.contexts),
            "num_sources": len(self.sources),
        }


class RAGEvaluator:
    """
    RAGAS-style evaluator for the AinSeba RAG pipeline.

    Uses LLM-as-judge for faithfulness and relevancy,
    and rule-based scoring for context precision, recall, and citations.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API key for LLM-based evaluation.
            model: Model to use as evaluator.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.total_eval_tokens = 0

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: list[str],
        sources: list[dict],
        ground_truth_sections: list[str] = None,
    ) -> dict[str, MetricResult]:
        """
        Run all metrics on a single Q&A pair.

        Args:
            question: The user's question.
            answer: The RAG pipeline's answer.
            ground_truth: The expected correct answer.
            contexts: List of retrieved context strings.
            sources: List of source metadata dicts.
            ground_truth_sections: Expected section references.

        Returns:
            Dict mapping metric name to MetricResult.
        """
        metrics = {}

        # 1. Faithfulness (LLM-based)
        metrics["faithfulness"] = self._eval_faithfulness(answer, contexts)

        # 2. Answer Relevancy (LLM-based)
        metrics["answer_relevancy"] = self._eval_answer_relevancy(question, answer)

        # 3. Context Precision (LLM-based)
        metrics["context_precision"] = self._eval_context_precision(question, contexts)

        # 4. Context Recall (LLM-based)
        metrics["context_recall"] = self._eval_context_recall(
            ground_truth, contexts
        )

        # 5. Citation Accuracy (rule-based)
        metrics["citation_accuracy"] = self._eval_citation_accuracy(
            answer, sources, ground_truth_sections or []
        )

        return metrics

    # ------------------------------------------
    # Metric 1: Faithfulness
    # ------------------------------------------

    def _eval_faithfulness(
        self, answer: str, contexts: list[str]
    ) -> MetricResult:
        """Is the answer grounded in the retrieved context?"""
        if not contexts:
            return MetricResult(
                name="faithfulness", score=0.0,
                reasoning="No context was retrieved.",
            )

        context_text = "\n\n".join(contexts[:5])

        prompt = f"""You are evaluating a legal RAG system. Determine if the answer is faithfully grounded in the provided context.

CONTEXT:
{context_text[:3000]}

ANSWER:
{answer[:1500]}

Rate faithfulness from 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context
- 0.7: Most claims are supported, minor unsupported details
- 0.5: Some claims are supported, some are not
- 0.3: Few claims are supported by the context
- 0.0: Answer contradicts or is completely unrelated to the context

Respond in JSON format: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

        return self._llm_eval("faithfulness", prompt)

    # ------------------------------------------
    # Metric 2: Answer Relevancy
    # ------------------------------------------

    def _eval_answer_relevancy(
        self, question: str, answer: str
    ) -> MetricResult:
        """Does the answer address the question?"""
        prompt = f"""You are evaluating a legal RAG system. Determine if the answer is relevant to the question asked.

QUESTION:
{question}

ANSWER:
{answer[:1500]}

Rate answer relevancy from 0.0 to 1.0:
- 1.0: Answer directly and completely addresses the question
- 0.7: Answer mostly addresses the question with minor gaps
- 0.5: Answer partially addresses the question
- 0.3: Answer is tangentially related
- 0.0: Answer does not address the question at all

Respond in JSON format: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

        return self._llm_eval("answer_relevancy", prompt)

    # ------------------------------------------
    # Metric 3: Context Precision
    # ------------------------------------------

    def _eval_context_precision(
        self, question: str, contexts: list[str]
    ) -> MetricResult:
        """Are the retrieved chunks relevant to the question?"""
        if not contexts:
            return MetricResult(
                name="context_precision", score=0.0,
                reasoning="No context retrieved.",
            )

        context_summaries = ""
        for i, ctx in enumerate(contexts[:5], 1):
            context_summaries += f"\n[Chunk {i}]: {ctx[:300]}...\n"

        prompt = f"""You are evaluating a legal RAG system's retrieval quality. For each retrieved chunk, determine if it is relevant to answering the question.

QUESTION:
{question}

RETRIEVED CHUNKS:
{context_summaries}

For each chunk, indicate if it is relevant (1) or irrelevant (0) to answering the question.
Then calculate precision as: number of relevant chunks / total chunks.

Respond in JSON format: {{"score": <float>, "reasoning": "<brief explanation>", "chunk_relevance": [1, 0, ...]}}"""

        return self._llm_eval("context_precision", prompt)

    # ------------------------------------------
    # Metric 4: Context Recall
    # ------------------------------------------

    def _eval_context_recall(
        self, ground_truth: str, contexts: list[str]
    ) -> MetricResult:
        """Did retrieval capture the needed information?"""
        if not ground_truth:
            return MetricResult(
                name="context_recall", score=1.0,
                reasoning="No ground truth to compare (out-of-scope query).",
            )

        if not contexts:
            return MetricResult(
                name="context_recall", score=0.0,
                reasoning="No context retrieved.",
            )

        context_text = "\n\n".join(contexts[:5])

        prompt = f"""You are evaluating a legal RAG system's retrieval completeness. Determine what fraction of the ground truth information is present in the retrieved context.

GROUND TRUTH ANSWER:
{ground_truth[:1500]}

RETRIEVED CONTEXT:
{context_text[:3000]}

Rate context recall from 0.0 to 1.0:
- 1.0: All key information from the ground truth is present in the context
- 0.7: Most key information is present
- 0.5: About half the key information is present
- 0.3: Little key information is present
- 0.0: None of the key information is found in the context

Respond in JSON format: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

        return self._llm_eval("context_recall", prompt)

    # ------------------------------------------
    # Metric 5: Citation Accuracy (rule-based)
    # ------------------------------------------

    def _eval_citation_accuracy(
        self,
        answer: str,
        sources: list[dict],
        expected_sections: list[str],
    ) -> MetricResult:
        """Are section references in the answer correct?"""
        if not expected_sections:
            return MetricResult(
                name="citation_accuracy", score=1.0,
                reasoning="No expected sections to verify.",
                details={"expected": [], "found": [], "matched": []},
            )

        # Extract section references from the answer
        found_sections = set()
        patterns = [
            r'Section\s+(\d+[A-Za-z]*)',
            r'ধারা\s+(\d+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            for m in matches:
                found_sections.add(f"Section {m}")

        # Also check source metadata
        for src in sources:
            sec = src.get("section_number", "")
            if sec:
                found_sections.add(f"Section {sec}")

        # Calculate overlap
        expected_set = set(expected_sections)
        matched = expected_set.intersection(found_sections)

        score = len(matched) / len(expected_set) if expected_set else 1.0

        return MetricResult(
            name="citation_accuracy",
            score=score,
            reasoning=f"Matched {len(matched)}/{len(expected_set)} expected sections.",
            details={
                "expected": list(expected_set),
                "found": list(found_sections),
                "matched": list(matched),
            },
        )

    # ------------------------------------------
    # LLM Evaluation Helper
    # ------------------------------------------

    def _llm_eval(self, metric_name: str, prompt: str) -> MetricResult:
        """Call LLM for evaluation and parse JSON response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            self.total_eval_tokens += response.usage.total_tokens
            content = response.choices[0].message.content.strip()

            data = json.loads(content)
            score = float(data.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # Clamp

            return MetricResult(
                name=metric_name,
                score=score,
                reasoning=data.get("reasoning", ""),
                details={k: v for k, v in data.items() if k not in ("score", "reasoning")},
            )

        except Exception as e:
            logger.error(f"LLM eval error for {metric_name}: {e}")
            return MetricResult(
                name=metric_name,
                score=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
            )
