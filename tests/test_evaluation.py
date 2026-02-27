"""
AinSeba - Unit Tests for Phase 7 (RAG Evaluation)

Run with: pytest tests/test_evaluation.py -v

All tests use mocked LLM calls — no API key needed.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    RAGEvaluator,
    MetricResult,
    EvalResult,
)


# ============================================
# Test: MetricResult
# ============================================

class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_basic_creation(self):
        result = MetricResult(name="faithfulness", score=0.85, reasoning="Well grounded")
        assert result.name == "faithfulness"
        assert result.score == 0.85
        assert result.reasoning == "Well grounded"

    def test_with_details(self):
        result = MetricResult(
            name="citation_accuracy", score=0.5,
            reasoning="Matched 1/2",
            details={"expected": ["Section 42"], "found": ["Section 42", "Section 10"]},
        )
        assert result.details["expected"] == ["Section 42"]

    def test_default_values(self):
        result = MetricResult(name="test", score=0.0)
        assert result.reasoning == ""
        assert result.details == {}


# ============================================
# Test: EvalResult
# ============================================

class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def _make_eval_result(self, scores: dict = None) -> EvalResult:
        er = EvalResult(
            eval_id="eval_001",
            question="What is theft?",
            answer="According to Section 378...",
            ground_truth="Theft is defined in Section 378...",
            query_type="factual",
            category="Criminal Law",
        )
        if scores:
            for name, score in scores.items():
                er.metrics[name] = MetricResult(name=name, score=score)
        return er

    def test_overall_score_empty(self):
        er = self._make_eval_result()
        assert er.overall_score == 0.0

    def test_overall_score_perfect(self):
        er = self._make_eval_result({
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
            "citation_accuracy": 1.0,
        })
        assert er.overall_score == 1.0

    def test_overall_score_mixed(self):
        er = self._make_eval_result({
            "faithfulness": 0.8,
            "answer_relevancy": 0.6,
            "context_precision": 0.7,
            "context_recall": 0.5,
            "citation_accuracy": 0.9,
        })
        score = er.overall_score
        assert 0.5 < score < 0.9

    def test_overall_score_weighted(self):
        """Faithfulness (0.30) should have more weight than citation (0.10)."""
        high_faith = self._make_eval_result({
            "faithfulness": 1.0,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "citation_accuracy": 0.0,
        })
        low_faith = self._make_eval_result({
            "faithfulness": 0.0,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "citation_accuracy": 1.0,
        })
        assert high_faith.overall_score > low_faith.overall_score

    def test_to_dict(self):
        er = self._make_eval_result({"faithfulness": 0.8})
        d = er.to_dict()
        assert d["eval_id"] == "eval_001"
        assert d["query_type"] == "factual"
        assert "faithfulness" in d["metrics"]
        assert d["metrics"]["faithfulness"]["score"] == 0.8

    def test_to_dict_has_counts(self):
        er = self._make_eval_result()
        er.contexts = ["ctx1", "ctx2"]
        er.sources = [{"citation": "src1"}]
        d = er.to_dict()
        assert d["num_contexts"] == 2
        assert d["num_sources"] == 1


# ============================================
# Test: Citation Accuracy (rule-based)
# ============================================

class TestCitationAccuracy:
    """Tests for the rule-based citation accuracy metric."""

    def _build_evaluator(self):
        evaluator = RAGEvaluator.__new__(RAGEvaluator)
        evaluator.client = MagicMock()
        evaluator.model = "gpt-4o-mini"
        evaluator.total_eval_tokens = 0
        return evaluator

    def test_perfect_citation(self):
        evaluator = self._build_evaluator()
        result = evaluator._eval_citation_accuracy(
            answer="According to Section 378 of the Penal Code...",
            sources=[{"section_number": "378"}],
            expected_sections=["Section 378"],
        )
        assert result.score == 1.0

    def test_partial_citation(self):
        evaluator = self._build_evaluator()
        result = evaluator._eval_citation_accuracy(
            answer="According to Section 378...",
            sources=[],
            expected_sections=["Section 378", "Section 379"],
        )
        assert result.score == 0.5

    def test_missing_citations(self):
        evaluator = self._build_evaluator()
        result = evaluator._eval_citation_accuracy(
            answer="Theft is a crime.",
            sources=[],
            expected_sections=["Section 378"],
        )
        assert result.score == 0.0

    def test_no_expected_sections(self):
        evaluator = self._build_evaluator()
        result = evaluator._eval_citation_accuracy(
            answer="Some answer.",
            sources=[],
            expected_sections=[],
        )
        assert result.score == 1.0

    def test_section_from_sources(self):
        evaluator = self._build_evaluator()
        result = evaluator._eval_citation_accuracy(
            answer="The law says...",  # No section in text
            sources=[{"section_number": "42"}],
            expected_sections=["Section 42"],
        )
        assert result.score == 1.0

    def test_citation_details(self):
        evaluator = self._build_evaluator()
        result = evaluator._eval_citation_accuracy(
            answer="Section 100 and Section 108 apply here.",
            sources=[],
            expected_sections=["Section 100", "Section 108", "Section 117"],
        )
        assert "expected" in result.details
        assert "found" in result.details
        assert "matched" in result.details
        assert len(result.details["matched"]) == 2


# ============================================
# Test: RAGEvaluator LLM Methods (mocked)
# ============================================

class TestRAGEvaluatorMocked:
    """Tests for LLM-based metrics with mocked OpenAI."""

    def _build_mock_evaluator(self, score: float = 0.8, reasoning: str = "Good"):
        evaluator = RAGEvaluator.__new__(RAGEvaluator)
        evaluator.model = "gpt-4o-mini"
        evaluator.total_eval_tokens = 0

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "score": score,
            "reasoning": reasoning,
        })
        mock_response.usage.total_tokens = 100
        mock_client.chat.completions.create.return_value = mock_response
        evaluator.client = mock_client

        return evaluator

    def test_faithfulness(self):
        evaluator = self._build_mock_evaluator(0.9, "Well grounded")
        result = evaluator._eval_faithfulness(
            answer="According to Section 378...",
            contexts=["Section 378 defines theft as..."],
        )
        assert result.name == "faithfulness"
        assert result.score == 0.9

    def test_faithfulness_no_context(self):
        evaluator = self._build_mock_evaluator()
        result = evaluator._eval_faithfulness("answer", [])
        assert result.score == 0.0

    def test_answer_relevancy(self):
        evaluator = self._build_mock_evaluator(0.85, "Addresses the question")
        result = evaluator._eval_answer_relevancy(
            question="What is theft?",
            answer="Theft is defined in Section 378...",
        )
        assert result.score == 0.85

    def test_context_precision(self):
        evaluator = self._build_mock_evaluator(0.8, "4 of 5 relevant")
        result = evaluator._eval_context_precision(
            question="What is theft?",
            contexts=["chunk1", "chunk2", "chunk3"],
        )
        assert result.score == 0.8

    def test_context_precision_no_context(self):
        evaluator = self._build_mock_evaluator()
        result = evaluator._eval_context_precision("question", [])
        assert result.score == 0.0

    def test_context_recall(self):
        evaluator = self._build_mock_evaluator(0.7, "Most info found")
        result = evaluator._eval_context_recall(
            ground_truth="Section 378 defines theft...",
            contexts=["Section 378 of the Penal Code..."],
        )
        assert result.score == 0.7

    def test_context_recall_no_ground_truth(self):
        evaluator = self._build_mock_evaluator()
        result = evaluator._eval_context_recall("", ["context"])
        assert result.score == 1.0  # Out-of-scope

    def test_context_recall_no_context(self):
        evaluator = self._build_mock_evaluator()
        result = evaluator._eval_context_recall("ground truth", [])
        assert result.score == 0.0

    def test_full_evaluate(self):
        evaluator = self._build_mock_evaluator(0.8, "Good")
        metrics = evaluator.evaluate(
            question="What is theft?",
            answer="Section 378 defines theft.",
            ground_truth="Theft is defined in Section 378.",
            contexts=["Section 378 of the Penal Code..."],
            sources=[{"section_number": "378"}],
            ground_truth_sections=["Section 378"],
        )
        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics
        assert "context_precision" in metrics
        assert "context_recall" in metrics
        assert "citation_accuracy" in metrics
        assert all(isinstance(v, MetricResult) for v in metrics.values())

    def test_token_tracking(self):
        evaluator = self._build_mock_evaluator()
        evaluator._eval_faithfulness("answer", ["context"])
        evaluator._eval_answer_relevancy("question", "answer")
        assert evaluator.total_eval_tokens == 200  # 100 per call

    def test_llm_eval_error_handling(self):
        evaluator = self._build_mock_evaluator()
        evaluator.client.chat.completions.create.side_effect = Exception("API error")
        result = evaluator._eval_faithfulness("answer", ["context"])
        assert result.score == 0.0
        assert "failed" in result.reasoning.lower()

    def test_score_clamping(self):
        evaluator = self._build_mock_evaluator(1.5, "Over max")
        result = evaluator._eval_faithfulness("answer", ["context"])
        assert result.score == 1.0

        evaluator2 = self._build_mock_evaluator(-0.5, "Under min")
        result2 = evaluator2._eval_answer_relevancy("q", "a")
        assert result2.score == 0.0


# ============================================
# Test: Dataset
# ============================================

class TestDataset:
    """Tests for the evaluation dataset."""

    def test_dataset_loads(self):
        dataset_path = Path(__file__).parent.parent / "evaluation" / "dataset.json"
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        assert len(dataset) == 40

    def test_dataset_has_required_fields(self):
        dataset_path = Path(__file__).parent.parent / "evaluation" / "dataset.json"
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        required = {"id", "question", "ground_truth", "query_type"}
        for item in dataset:
            assert required.issubset(item.keys()), f"Missing fields in {item.get('id')}"

    def test_dataset_ids_unique(self):
        dataset_path = Path(__file__).parent.parent / "evaluation" / "dataset.json"
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        ids = [item["id"] for item in dataset]
        assert len(ids) == len(set(ids)), "Duplicate evaluation IDs found"

    def test_dataset_query_types(self):
        dataset_path = Path(__file__).parent.parent / "evaluation" / "dataset.json"
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        types = set(item["query_type"] for item in dataset)
        assert "factual" in types
        assert "situational" in types
        assert "comparative" in types
        assert "out_of_scope" in types

    def test_dataset_categories(self):
        dataset_path = Path(__file__).parent.parent / "evaluation" / "dataset.json"
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        categories = set(item.get("category", "") for item in dataset if item.get("category"))
        assert "Employment" in categories
        assert "Criminal Law" in categories

    def test_out_of_scope_query_exists(self):
        dataset_path = Path(__file__).parent.parent / "evaluation" / "dataset.json"
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        oos = [d for d in dataset if d["query_type"] == "out_of_scope"]
        assert len(oos) >= 1


# ============================================
# Test: Report Generation
# ============================================

class TestReportGeneration:
    """Tests for the evaluation report generator."""

    def _make_results(self) -> list[dict]:
        return [
            {
                "eval_id": "eval_001",
                "question": "What is theft?",
                "overall_score": 0.85,
                "query_type": "factual",
                "category": "Criminal Law",
                "metrics": {
                    "faithfulness": {"score": 0.9},
                    "answer_relevancy": {"score": 0.8},
                    "context_precision": {"score": 0.8},
                    "context_recall": {"score": 0.7},
                    "citation_accuracy": {"score": 1.0},
                },
            },
            {
                "eval_id": "eval_002",
                "question": "Working hours?",
                "overall_score": 0.65,
                "query_type": "factual",
                "category": "Employment",
                "metrics": {
                    "faithfulness": {"score": 0.7},
                    "answer_relevancy": {"score": 0.6},
                    "context_precision": {"score": 0.5},
                    "context_recall": {"score": 0.5},
                    "citation_accuracy": {"score": 0.8},
                },
            },
            {
                "eval_id": "eval_003",
                "question": "My employer...",
                "overall_score": 0.75,
                "query_type": "situational",
                "category": "Employment",
                "metrics": {
                    "faithfulness": {"score": 0.8},
                    "answer_relevancy": {"score": 0.7},
                    "context_precision": {"score": 0.7},
                    "context_recall": {"score": 0.6},
                    "citation_accuracy": {"score": 0.9},
                },
            },
        ]

    def test_report_generation(self):
        from evaluation.run_evaluation import generate_report
        results = self._make_results()
        report = generate_report(results)
        assert "overall" in report
        assert "metrics" in report
        assert "by_query_type" in report
        assert "grade_distribution" in report

    def test_report_overall_score(self):
        from evaluation.run_evaluation import generate_report
        results = self._make_results()
        report = generate_report(results)
        mean = report["overall"]["mean"]
        assert 0.5 < mean < 1.0

    def test_report_by_query_type(self):
        from evaluation.run_evaluation import generate_report
        results = self._make_results()
        report = generate_report(results)
        assert "factual" in report["by_query_type"]
        assert "situational" in report["by_query_type"]
        assert report["by_query_type"]["factual"]["count"] == 2

    def test_report_by_category(self):
        from evaluation.run_evaluation import generate_report
        results = self._make_results()
        report = generate_report(results)
        assert "Employment" in report["by_category"]
        assert "Criminal Law" in report["by_category"]

    def test_report_grade_distribution(self):
        from evaluation.run_evaluation import generate_report
        results = self._make_results()
        report = generate_report(results)
        grades = report["grade_distribution"]
        total = sum(grades.values())
        assert total == 3

    def test_report_weakest_queries(self):
        from evaluation.run_evaluation import generate_report
        results = self._make_results()
        report = generate_report(results)
        weakest = report["weakest_queries"]
        assert len(weakest) <= 5
        assert weakest[0]["score"] <= weakest[-1]["score"]

    def test_empty_results(self):
        from evaluation.run_evaluation import generate_report
        report = generate_report([])
        assert "error" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
