#!/usr/bin/env python3
"""
AinSeba - RAG Evaluation Runner
Runs the full RAGAS-style evaluation pipeline.

Usage:
    python evaluation/run_evaluation.py                    # Full evaluation (all 40 queries)
    python evaluation/run_evaluation.py --limit 5          # Quick test (5 queries)
    python evaluation/run_evaluation.py --type factual     # Only factual queries
    python evaluation/run_evaluation.py --report-only      # Re-generate report from saved results
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OPENAI_API_KEY, PROCESSED_DATA_DIR


def setup_logging(level: str = "WARNING"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset(path: Path = None) -> list[dict]:
    """Load evaluation dataset."""
    if path is None:
        path = Path(__file__).parent / "dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    dataset: list[dict],
    limit: int = None,
    query_type: str = None,
    no_reranker: bool = False,
) -> list[dict]:
    """
    Run evaluation on the dataset.

    Returns list of EvalResult dicts.
    """
    from evaluation.metrics import RAGEvaluator, EvalResult
    from src.chain.builder import build_rag_chain

    # Filter dataset
    if query_type:
        dataset = [d for d in dataset if d.get("query_type") == query_type]
    if limit:
        dataset = dataset[:limit]

    print(f"\nRunning evaluation on {len(dataset)} queries...")
    print(f"{'='*70}")

    # Build chain
    chain = build_rag_chain(use_reranker=not no_reranker)
    evaluator = RAGEvaluator(api_key=OPENAI_API_KEY)

    results = []
    for i, item in enumerate(dataset, 1):
        eval_id = item["id"]
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        gt_sections = item.get("ground_truth_sections", [])
        q_type = item.get("query_type", "unknown")
        category = item.get("category", "")

        print(f"\n[{i}/{len(dataset)}] {eval_id}: {question[:60]}...")

        try:
            # Run RAG pipeline
            response = chain.query(question)

            # Extract contexts from retrieval
            contexts = []
            sources = response.sources
            for src in sources:
                preview = src.get("text_preview", "")
                if preview:
                    contexts.append(preview)

            # If no previews, use source metadata as context approximation
            if not contexts and sources:
                for src in sources:
                    citation = src.get("citation", "")
                    if citation:
                        contexts.append(citation)

            # Run evaluation metrics
            if q_type == "out_of_scope":
                # For out-of-scope queries, check if system correctly declines
                declined = any(
                    phrase in response.answer.lower()
                    for phrase in [
                        "don't have information",
                        "outside", "not within",
                        "cannot answer", "beyond",
                        "do not have", "unable",
                    ]
                )
                metrics_dict = {
                    "faithfulness": {"score": 1.0 if declined else 0.0,
                                     "reasoning": "Correctly declined" if declined else "Should have declined"},
                    "answer_relevancy": {"score": 1.0 if declined else 0.0,
                                          "reasoning": "Out-of-scope handling"},
                    "context_precision": {"score": 1.0, "reasoning": "N/A for out-of-scope"},
                    "context_recall": {"score": 1.0, "reasoning": "N/A for out-of-scope"},
                    "citation_accuracy": {"score": 1.0, "reasoning": "N/A for out-of-scope"},
                }
                eval_result = EvalResult(
                    eval_id=eval_id,
                    question=question,
                    answer=response.answer,
                    ground_truth=ground_truth,
                    contexts=contexts,
                    sources=sources,
                    query_type=q_type,
                    category=category,
                )
                from evaluation.metrics import MetricResult
                for name, data in metrics_dict.items():
                    eval_result.metrics[name] = MetricResult(
                        name=name, score=data["score"], reasoning=data["reasoning"]
                    )
            else:
                metrics = evaluator.evaluate(
                    question=question,
                    answer=response.answer,
                    ground_truth=ground_truth,
                    contexts=contexts,
                    sources=sources,
                    ground_truth_sections=gt_sections,
                )
                eval_result = EvalResult(
                    eval_id=eval_id,
                    question=question,
                    answer=response.answer,
                    ground_truth=ground_truth,
                    contexts=contexts,
                    sources=sources,
                    metrics=metrics,
                    query_type=q_type,
                    category=category,
                )

            score = eval_result.overall_score
            grade = "A" if score >= 0.8 else "B" if score >= 0.6 else "C" if score >= 0.4 else "D"
            print(f"  Score: {score:.2f} ({grade})")
            for name, m in eval_result.metrics.items():
                print(f"    {name}: {m.score:.2f}")

            results.append(eval_result.to_dict())

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "eval_id": eval_id,
                "question": question,
                "error": str(e),
                "overall_score": 0.0,
            })

    print(f"\nEvaluation tokens used: {evaluator.total_eval_tokens}")
    return results


def generate_report(results: list[dict]) -> dict:
    """Generate evaluation report with aggregate metrics."""
    if not results:
        return {"error": "No results to report"}

    # Filter out errors
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not valid:
        return {"error": "All evaluations failed", "errors": errors}

    # Aggregate metrics
    metric_names = ["faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "citation_accuracy"]

    aggregate = {}
    for metric in metric_names:
        scores = [r["metrics"][metric]["score"] for r in valid if metric in r.get("metrics", {})]
        if scores:
            aggregate[metric] = {
                "mean": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "count": len(scores),
            }

    # Overall scores
    overall_scores = [r["overall_score"] for r in valid]
    overall = {
        "mean": round(sum(overall_scores) / len(overall_scores), 4),
        "min": round(min(overall_scores), 4),
        "max": round(max(overall_scores), 4),
    }

    # By query type
    by_type = defaultdict(list)
    for r in valid:
        by_type[r.get("query_type", "unknown")].append(r["overall_score"])

    type_scores = {
        t: {"mean": round(sum(s) / len(s), 4), "count": len(s)}
        for t, s in by_type.items()
    }

    # By category
    by_category = defaultdict(list)
    for r in valid:
        cat = r.get("category", "Unknown")
        if cat:
            by_category[cat].append(r["overall_score"])

    category_scores = {
        c: {"mean": round(sum(s) / len(s), 4), "count": len(s)}
        for c, s in by_category.items()
    }

    # Grade distribution
    grades = {"A (>=0.8)": 0, "B (0.6-0.8)": 0, "C (0.4-0.6)": 0, "D (<0.4)": 0}
    for s in overall_scores:
        if s >= 0.8:
            grades["A (>=0.8)"] += 1
        elif s >= 0.6:
            grades["B (0.6-0.8)"] += 1
        elif s >= 0.4:
            grades["C (0.4-0.6)"] += 1
        else:
            grades["D (<0.4)"] += 1

    # Weakest queries
    weakest = sorted(valid, key=lambda r: r["overall_score"])[:5]
    weakest_summary = [
        {
            "id": r["eval_id"],
            "question": r["question"][:60],
            "score": r["overall_score"],
        }
        for r in weakest
    ]

    report = {
        "evaluation_date": datetime.now().isoformat(),
        "total_queries": len(results),
        "successful": len(valid),
        "errors": len(errors),
        "overall": overall,
        "metrics": aggregate,
        "by_query_type": type_scores,
        "by_category": category_scores,
        "grade_distribution": grades,
        "weakest_queries": weakest_summary,
    }

    return report


def print_report(report: dict):
    """Pretty-print the evaluation report."""
    print(f"\n{'='*70}")
    print("AinSeba RAG EVALUATION REPORT")
    print(f"{'='*70}")

    print(f"\nDate: {report.get('evaluation_date', 'N/A')}")
    print(f"Queries: {report.get('successful', 0)}/{report.get('total_queries', 0)} successful")

    overall = report.get("overall", {})
    print(f"\nOVERALL SCORE: {overall.get('mean', 0):.2f} "
          f"(min={overall.get('min', 0):.2f}, max={overall.get('max', 0):.2f})")

    print(f"\n--- Metric Breakdown ---")
    for name, data in report.get("metrics", {}).items():
        bar = "#" * int(data["mean"] * 20)
        print(f"  {name:25s} {data['mean']:.2f}  [{bar:20s}]")

    print(f"\n--- By Query Type ---")
    for t, data in report.get("by_query_type", {}).items():
        print(f"  {t:20s} {data['mean']:.2f}  (n={data['count']})")

    print(f"\n--- By Category ---")
    for c, data in report.get("by_category", {}).items():
        print(f"  {c:25s} {data['mean']:.2f}  (n={data['count']})")

    print(f"\n--- Grade Distribution ---")
    for grade, count in report.get("grade_distribution", {}).items():
        print(f"  {grade:15s} {count}")

    print(f"\n--- Weakest Queries ---")
    for w in report.get("weakest_queries", []):
        print(f"  [{w['id']}] {w['score']:.2f} - {w['question']}")

    # Analysis & recommendations
    metrics = report.get("metrics", {})
    print(f"\n--- Recommendations ---")
    if metrics.get("faithfulness", {}).get("mean", 1) < 0.7:
        print("  [!] Low faithfulness: Tighten system prompt grounding constraints")
    if metrics.get("context_precision", {}).get("mean", 1) < 0.7:
        print("  [!] Low context precision: Tune retrieval top_k or add reranker threshold")
    if metrics.get("context_recall", {}).get("mean", 1) < 0.7:
        print("  [!] Low context recall: Increase top_k or adjust chunk overlap")
    if metrics.get("answer_relevancy", {}).get("mean", 1) < 0.7:
        print("  [!] Low answer relevancy: Improve prompt template to focus on the question")
    if metrics.get("citation_accuracy", {}).get("mean", 1) < 0.7:
        print("  [!] Low citation accuracy: Strengthen citation instructions in system prompt")
    if all(m.get("mean", 0) >= 0.7 for m in metrics.values()):
        print("  All metrics above 0.7 - pipeline is performing well!")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="AinSeba - RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/run_evaluation.py                    # Full evaluation
  python evaluation/run_evaluation.py --limit 5          # Quick test
  python evaluation/run_evaluation.py --type factual     # Only factual queries
  python evaluation/run_evaluation.py --type situational # Only situational queries
  python evaluation/run_evaluation.py --report-only      # Re-print last report
        """,
    )

    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of queries to evaluate")
    parser.add_argument("--type", type=str, default=None,
                       choices=["factual", "situational", "comparative", "out_of_scope"],
                       help="Filter by query type")
    parser.add_argument("--no-reranker", action="store_true",
                       help="Skip cross-encoder reranking")
    parser.add_argument("--report-only", action="store_true",
                       help="Re-generate report from saved results")
    parser.add_argument("--log-level", default="WARNING",
                       choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory for output files")

    args = parser.parse_args()
    setup_logging(args.log_level)

    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "evaluation_results.json"
    report_path = output_dir / "evaluation_report.json"

    if args.report_only:
        if not results_path.exists():
            print(f"No results found at {results_path}. Run evaluation first.")
            sys.exit(1)
        with open(results_path, "r") as f:
            results = json.load(f)
        report = generate_report(results)
        print_report(report)
        return

    # Validate API key
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    # Load dataset
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} evaluation queries")

    # Run evaluation
    results = run_evaluation(
        dataset=dataset,
        limit=args.limit,
        query_type=args.type,
        no_reranker=args.no_reranker,
    )

    # Save results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")

    # Generate and save report
    report = generate_report(results)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}")

    # Print report
    print_report(report)


if __name__ == "__main__":
    main()
