"""
AinSeba - Retrieval Quality Testing
Tests retrieval with sample queries and logs results for inspection.

Usage:
    python scripts/test_retrieval.py                        # Run all test queries
    python scripts/test_retrieval.py --query "your question" # Test a single query
    python scripts/test_retrieval.py --interactive           # Interactive mode
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OPENAI_API_KEY,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
    RERANKER_MODEL,
    PROCESSED_DATA_DIR,
)
from src.vectorstore.embeddings import EmbeddingGenerator
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import LegalRetriever, print_results, format_results_table

logger = logging.getLogger(__name__)


# ============================================
# Sample Test Queries
# Covering diverse legal topics and query types
# ============================================

SAMPLE_QUERIES = [
    # --- Direct Factual Queries ---
    {
        "query": "What is the maximum working hours per day?",
        "type": "factual",
        "expected_act": "labour",
        "description": "Direct section lookup about working hours",
    },
    {
        "query": "What is the penalty for theft?",
        "type": "factual",
        "expected_act": "penal",
        "description": "Criminal law penalty lookup",
    },
    {
        "query": "What are the consumer rights for defective products?",
        "type": "factual",
        "expected_act": "consumer",
        "description": "Consumer protection query",
    },
    {
        "query": "What is the punishment for cybercrime?",
        "type": "factual",
        "expected_act": "cyber",
        "description": "Cyber security penalties",
    },

    # --- Situational Queries ---
    {
        "query": "My employer hasn't paid me for 3 months. What can I do?",
        "type": "situational",
        "expected_act": "labour",
        "description": "Practical labor dispute scenario",
    },
    {
        "query": "My landlord is trying to evict me without notice. What are my rights?",
        "type": "situational",
        "expected_act": "rent",
        "description": "Tenant rights scenario",
    },
    {
        "query": "Someone stole my credit card information online. What law protects me?",
        "type": "situational",
        "expected_act": "cyber",
        "description": "Digital crime scenario",
    },

    # --- Comparative Queries ---
    {
        "query": "What are the different types of leave an employee is entitled to?",
        "type": "comparative",
        "expected_act": "labour",
        "description": "Comparison of leave types",
    },
    {
        "query": "What are the penalties for workplace safety violations?",
        "type": "comparative",
        "expected_act": "labour",
        "description": "Safety violation penalties",
    },

    # --- Section-Specific Queries ---
    {
        "query": "What does Section 5 say about rest intervals?",
        "type": "section_specific",
        "expected_act": "labour",
        "description": "Direct section reference",
    },

    # --- Broad/General Queries ---
    {
        "query": "What are the fundamental rights of workers in Bangladesh?",
        "type": "general",
        "expected_act": "labour",
        "description": "Broad rights-based query",
    },
    {
        "query": "How does the law protect women in the workplace?",
        "type": "general",
        "expected_act": "labour",
        "description": "Gender-specific protections",
    },

    # --- Sample/Test Document Queries (for testing with sample PDF) ---
    {
        "query": "What are the maximum working hours allowed per day?",
        "type": "factual",
        "expected_act": "sample",
        "description": "Test with sample document - working hours",
    },
    {
        "query": "When must wages be paid to workers?",
        "type": "factual",
        "expected_act": "sample",
        "description": "Test with sample document - wage payment",
    },
    {
        "query": "What deductions can be made from worker wages?",
        "type": "factual",
        "expected_act": "sample",
        "description": "Test with sample document - wage deductions",
    },
]


def build_retriever(use_reranker: bool = True) -> LegalRetriever:
    """Initialize all retrieval components."""
    embedder = EmbeddingGenerator(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    store = ChromaStore(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    reranker = None
    if use_reranker:
        try:
            reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}. Proceeding without reranking.")

    retriever = LegalRetriever(
        chroma_store=store,
        embedding_generator=embedder,
        reranker=reranker,
        top_k=RETRIEVAL_TOP_K,
        rerank_top_n=RERANK_TOP_N,
    )

    return retriever


def run_test_queries(
    retriever: LegalRetriever,
    queries: list[dict] = None,
    save_results: bool = True,
) -> list[dict]:
    """
    Run all test queries and collect results.

    Args:
        retriever: Configured LegalRetriever instance.
        queries: List of query dicts to test. Defaults to SAMPLE_QUERIES.
        save_results: Whether to save results to JSON.

    Returns:
        List of test result dicts.
    """
    queries = queries or SAMPLE_QUERIES

    all_results = []
    passed = 0

    print(f"\n{'='*70}")
    print(f"AinSeba — Retrieval Quality Test")
    print(f"Running {len(queries)} test queries")
    print(f"{'='*70}")

    for i, query_info in enumerate(queries, 1):
        query = query_info["query"]
        print(f"\n--- Query {i}/{len(queries)}: [{query_info['type']}] ---")
        print(f"Q: {query}")

        try:
            results = retriever.retrieve(query=query)

            test_result = {
                "query_index": i,
                "query": query,
                "type": query_info["type"],
                "description": query_info["description"],
                "expected_act": query_info.get("expected_act", ""),
                "num_results": len(results),
                "top_results": [],
            }

            if results:
                for j, r in enumerate(results[:3]):  # Top 3
                    test_result["top_results"].append({
                        "rank": j + 1,
                        "chunk_id": r.chunk_id,
                        "citation": r.citation,
                        "similarity": r.similarity_score,
                        "rerank_score": r.rerank_score,
                        "text_preview": r.text[:200],
                    })

                top = results[0]
                print(f"  Top result: {top.citation}")
                print(f"  Similarity: {top.similarity_score:.4f}", end="")
                if top.rerank_score:
                    print(f" | Rerank: {top.rerank_score:.4f}", end="")
                print()
                print(f"  Preview: {top.text[:120]}...")

                # Simple pass/fail check
                expected = query_info.get("expected_act", "")
                if expected and expected in top.act_id:
                    print(f"  ✅ Matches expected act")
                    test_result["passed"] = True
                    passed += 1
                elif not expected:
                    test_result["passed"] = None  # Can't evaluate
                    passed += 1
                else:
                    print(f"  ⚠ Expected '{expected}' but got '{top.act_id}'")
                    test_result["passed"] = False
            else:
                print("  ❌ No results")
                test_result["passed"] = False

            all_results.append(test_result)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_results.append({
                "query_index": i,
                "query": query,
                "error": str(e),
                "passed": False,
            })

    # Summary
    total = len(queries)
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} queries returned relevant results")
    print(f"Pass rate: {passed/total*100:.0f}%")
    print(f"{'='*70}")

    # Save results
    if save_results:
        output_path = PROCESSED_DATA_DIR / "retrieval_test_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_date": datetime.now().isoformat(),
                "total_queries": total,
                "passed": passed,
                "pass_rate": round(passed / total * 100, 1),
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return all_results


def interactive_mode(retriever: LegalRetriever):
    """Interactive query mode — type queries and see results."""
    print(f"\n{'='*70}")
    print("AinSeba — Interactive Retrieval")
    print("Type a legal question and press Enter.")
    print("Commands: /quit, /stats, /filter act_id, /nofilter")
    print(f"{'='*70}")

    current_filter = {}

    while True:
        try:
            query = input("\n🔍 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query == "/quit":
            break
        elif query == "/stats":
            stats = retriever.store.get_stats()
            print(f"Documents: {stats['total_documents']}")
            print(f"Acts: {', '.join(stats['acts'])}")
            continue
        elif query.startswith("/filter "):
            act_id = query.split(" ", 1)[1].strip()
            current_filter = {"act_id": act_id}
            print(f"Filter set: act_id={act_id}")
            continue
        elif query == "/nofilter":
            current_filter = {}
            print("Filter cleared.")
            continue

        results = retriever.retrieve(query=query, **current_filter)
        print_results(results)


def main():
    parser = argparse.ArgumentParser(description="AinSeba — Retrieval Quality Testing")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", default=True, help="Run all test queries")
    group.add_argument("--query", type=str, help="Test a single query")
    group.add_argument("--interactive", action="store_true", help="Interactive query mode")

    parser.add_argument("--no-reranker", action="store_true", help="Skip cross-encoder reranking")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build retriever
    retriever = build_retriever(use_reranker=not args.no_reranker)
    if args.top_k:
        retriever.top_k = args.top_k

    if args.query:
        results = retriever.retrieve(query=args.query)
        print_results(results)
    elif args.interactive:
        interactive_mode(retriever)
    else:
        run_test_queries(retriever)


if __name__ == "__main__":
    main()
